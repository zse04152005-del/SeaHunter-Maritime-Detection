"""Thermal degradation and durable canary promotion/rollback state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class RuntimeMode(str, Enum):
    NORMAL = "normal"
    REDUCED_ROI = "reduced_roi"
    LOWER_RESOLUTION = "lower_resolution"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True, slots=True)
class RuntimeHealth:
    temperature_c: float | None
    gpu_memory_fraction: float | None
    inference_latency_ms: float
    timeout_streak: int = 0

    def __post_init__(self) -> None:
        if self.gpu_memory_fraction is not None and not 0.0 <= self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be within [0, 1]")
        if self.inference_latency_ms < 0.0 or self.timeout_streak < 0:
            raise ValueError("runtime latency and timeout streak must be non-negative")


@dataclass(frozen=True, slots=True)
class EdgeDegradationConfig:
    target_latency_ms: float = 50.0
    thermal_warning_c: float = 80.0
    thermal_critical_c: float = 90.0
    memory_warning_fraction: float = 0.9
    critical_timeout_streak: int = 3
    recovery_samples: int = 10

    def __post_init__(self) -> None:
        if self.target_latency_ms <= 0.0 or self.thermal_critical_c <= self.thermal_warning_c:
            raise ValueError("runtime latency and thermal thresholds are invalid")
        if not 0.0 < self.memory_warning_fraction <= 1.0:
            raise ValueError("memory_warning_fraction must be within (0, 1]")
        if self.critical_timeout_streak <= 0 or self.recovery_samples <= 0:
            raise ValueError("timeout and recovery controls must be positive")


@dataclass(frozen=True, slots=True)
class RuntimeDecision:
    mode: RuntimeMode
    changed: bool
    reasons: tuple[str, ...]


class EdgeDegradationController:
    def __init__(self, config: EdgeDegradationConfig | None = None) -> None:
        self.config = config or EdgeDegradationConfig()
        self._mode = RuntimeMode.NORMAL
        self._healthy_streak = 0

    def update(self, health: RuntimeHealth) -> RuntimeDecision:
        reasons: list[str] = []
        if health.temperature_c is not None and health.temperature_c >= self.config.thermal_warning_c:
            reasons.append("thermal")
        if health.gpu_memory_fraction is not None and health.gpu_memory_fraction >= self.config.memory_warning_fraction:
            reasons.append("gpu_memory")
        if health.inference_latency_ms > self.config.target_latency_ms:
            reasons.append("latency")
        if health.timeout_streak > 0:
            reasons.append("timeout")
        critical = (
            health.timeout_streak >= self.config.critical_timeout_streak
            or health.temperature_c is not None
            and health.temperature_c >= self.config.thermal_critical_c
        )
        if critical:
            target = RuntimeMode.SAFE_STOP
        elif len(reasons) >= 2:
            target = RuntimeMode.LOWER_RESOLUTION
        elif reasons:
            target = RuntimeMode.REDUCED_ROI
        else:
            self._healthy_streak += 1
            if self._healthy_streak < self.config.recovery_samples:
                return RuntimeDecision(self._mode, False, ("recovery_hysteresis",))
            target = RuntimeMode.NORMAL
        if reasons:
            self._healthy_streak = 0
        changed = target is not self._mode
        self._mode = target
        return RuntimeDecision(self._mode, changed, tuple(reasons))


class ReleasePhase(str, Enum):
    STABLE = "stable"
    STAGED = "staged"
    CANARY = "canary"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class ReleaseState:
    current_version: str
    previous_version: str | None
    candidate_version: str | None
    phase: ReleasePhase
    healthy_canary_samples: int = 0


class ReleaseManager:
    def __init__(self, state_path: Path, *, initial_version: str, required_healthy_samples: int = 5) -> None:
        if not initial_version.strip() or required_healthy_samples <= 0:
            raise ValueError("release initial_version and health gate are invalid")
        self.state_path = state_path
        self.required_healthy_samples = required_healthy_samples
        self.state = self._load() or ReleaseState(initial_version, None, None, ReleasePhase.STABLE)
        self._save()

    def stage(self, version: str) -> ReleaseState:
        if not version.strip() or version == self.state.current_version:
            raise ValueError("release candidate must be a new non-empty version")
        self.state = ReleaseState(
            self.state.current_version,
            self.state.previous_version,
            version,
            ReleasePhase.STAGED,
        )
        self._save()
        return self.state

    def begin_canary(self) -> ReleaseState:
        if self.state.phase is not ReleasePhase.STAGED or self.state.candidate_version is None:
            raise RuntimeError("a staged candidate is required before canary")
        self.state = ReleaseState(
            self.state.current_version,
            self.state.previous_version,
            self.state.candidate_version,
            ReleasePhase.CANARY,
        )
        self._save()
        return self.state

    def report_canary_health(self, healthy: bool) -> ReleaseState:
        candidate_version = self.state.candidate_version
        if self.state.phase is not ReleasePhase.CANARY or candidate_version is None:
            raise RuntimeError("canary health requires an active canary")
        if not healthy:
            self.state = ReleaseState(
                self.state.current_version,
                self.state.previous_version,
                None,
                ReleasePhase.ROLLED_BACK,
            )
        else:
            samples = self.state.healthy_canary_samples + 1
            if samples >= self.required_healthy_samples:
                self.state = ReleaseState(
                    candidate_version,
                    self.state.current_version,
                    None,
                    ReleasePhase.STABLE,
                )
            else:
                self.state = ReleaseState(
                    self.state.current_version,
                    self.state.previous_version,
                    self.state.candidate_version,
                    ReleasePhase.CANARY,
                    samples,
                )
        self._save()
        return self.state

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = asdict(self.state)
        payload["phase"] = self.state.phase.value
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def _load(self) -> ReleaseState | None:
        if not self.state_path.exists():
            return None
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return ReleaseState(
            current_version=str(payload["current_version"]),
            previous_version=None if payload["previous_version"] is None else str(payload["previous_version"]),
            candidate_version=None if payload["candidate_version"] is None else str(payload["candidate_version"]),
            phase=ReleasePhase(str(payload["phase"])),
            healthy_canary_samples=int(payload["healthy_canary_samples"]),
        )
