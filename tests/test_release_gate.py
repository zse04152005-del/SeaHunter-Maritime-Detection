from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from seahunter.acceptance import (
    ExternalReleaseGates,
    ReleaseAssets,
    ReviewApproval,
    ReviewDomain,
    SeaState,
    SeaTrialPlan,
    SeaTrialResult,
    TimeBand,
    audit_acceptance,
    freeze_release_manifest,
)
from seahunter.data import WeatherCondition


def plan() -> SeaTrialPlan:
    return SeaTrialPlan(
        "v1-trial",
        2,
        tuple(TimeBand),
        ("low", "medium", "high"),
        tuple(WeatherCondition),
        tuple(SeaState),
        5,
        1.0,
        0,
        0.8,
        20.0,
    )


def trials() -> tuple[SeaTrialResult, ...]:
    return tuple(
        SeaTrialResult(
            f"trial-{index}",
            f"location-{index % 2}",
            tuple(TimeBand)[index % len(TimeBand)],
            ("low", "medium", "high")[index % 3],
            weather,
            tuple(SeaState)[index % len(SeaState)],
            True,
            f"operator-{index}",
            "calibration.json",
            "sync.json",
            "zones-v1",
            "events.json",
            "evidence.zip",
            "a" * 64,
            0.5,
            0,
            0.9,
            10.0,
        )
        for index, weather in enumerate(WeatherCondition)
    )


def approvals() -> tuple[ReviewApproval, ...]:
    return tuple(ReviewApproval(domain, True, "reviewer", "review.pdf") for domain in ReviewDomain)


def assets() -> ReleaseAssets:
    return ReleaseAssets(
        version="1.0.0",
        git_commit="abc123",
        model_package_sha256="a" * 64,
        tensorrt_engine_sha256="b" * 64,
        edge_image_digest="sha256:" + "c" * 64,
        documentation_commit="def456",
        sea_trial_report_sha256="d" * 64,
    )


class ReleaseGateTests(unittest.TestCase):
    def test_external_hardware_gate_blocks_release_even_after_trial_contract_passes(self) -> None:
        acceptance = audit_acceptance(plan(), trials(), approvals())
        blocked = ExternalReleaseGates(True, True, False, False, False, False)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.json"
            with self.assertRaisesRegex(RuntimeError, "external hardware/data"):
                freeze_release_manifest(
                    output,
                    assets=assets(),
                    acceptance=acceptance,
                    external_gates=blocked,
                    frozen_at=datetime.now(timezone.utc),
                )
            self.assertFalse(output.exists())

    def test_failed_sea_trial_blocks_release(self) -> None:
        failed = audit_acceptance(plan(), (), ())
        passed_external = ExternalReleaseGates(True, True, True, True, True, True)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "sea-trial"):
            freeze_release_manifest(
                Path(directory) / "release.json",
                assets=assets(),
                acceptance=failed,
                external_gates=passed_external,
                frozen_at=datetime.now(timezone.utc),
            )

    def test_all_gates_atomically_freeze_exact_assets(self) -> None:
        acceptance = audit_acceptance(plan(), trials(), approvals())
        external = ExternalReleaseGates(True, True, True, True, True, True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.json"
            freeze_release_manifest(
                output,
                assets=assets(),
                acceptance=acceptance,
                external_gates=external,
                frozen_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["assets"]["version"], "1.0.0")
            self.assertTrue(payload["acceptance"]["passed"])
            self.assertTrue(all(payload["external_gates"].values()))


if __name__ == "__main__":
    unittest.main()
