"""Time-gated IMU/gimbal camera-motion prior for BoT-SORT."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import cos, isfinite, radians, sin, tan
from typing import Protocol

from seahunter.schemas import Detection, FramePacket, TelemetryPacket

from .global_motion import GlobalMotionEstimate, GlobalMotionEstimator


class TelemetryMotionEstimator(GlobalMotionEstimator, Protocol):
    """A global-motion estimator that consumes timestamped telemetry."""

    def add(self, packet: TelemetryPacket) -> None:
        """Add a telemetry packet to the bounded alignment history."""


@dataclass(frozen=True, slots=True)
class TelemetryMotionConfig:
    """Calibration and quality gates for a small-angle image-motion prior."""

    focal_length_px: float
    maximum_alignment_error_seconds: float = 0.1
    minimum_quality: float = 0.5
    maximum_angle_delta_degrees: float = 15.0
    maximum_packets_per_source: int = 512

    def __post_init__(self) -> None:
        if not isfinite(self.focal_length_px) or self.focal_length_px <= 0:
            raise ValueError("focal_length_px must be finite and positive")
        if not isfinite(self.maximum_alignment_error_seconds) or self.maximum_alignment_error_seconds <= 0:
            raise ValueError("maximum_alignment_error_seconds must be finite and positive")
        if not 0.0 <= self.minimum_quality <= 1.0:
            raise ValueError("minimum_quality must be within [0, 1]")
        if not 0.0 < self.maximum_angle_delta_degrees < 90.0:
            raise ValueError("maximum_angle_delta_degrees must be within (0, 90)")
        if self.maximum_packets_per_source < 2:
            raise ValueError("maximum_packets_per_source must be at least 2")


class TelemetryMotionPrior:
    """Convert aligned platform plus gimbal attitude deltas into an image affine.

    The transform models rotational camera motion only. It deliberately ignores
    platform translation and therefore acts as a gated prior, not a replacement
    for visual GMC or the calibrated geospatial model implemented in M3.
    """

    def __init__(self, config: TelemetryMotionConfig) -> None:
        self.config = config
        self._packets: dict[str, list[TelemetryPacket]] = {}
        self._estimator_id = (
            "imu-gimbal-prior:v1"
            f":focal={config.focal_length_px:g}"
            f":align={config.maximum_alignment_error_seconds:g}"
            f":quality={config.minimum_quality:g}"
            f":angle={config.maximum_angle_delta_degrees:g}"
        )

    @property
    def estimator_id(self) -> str:
        return self._estimator_id

    def add(self, packet: TelemetryPacket) -> None:
        """Insert one packet in timestamp order and retain a bounded history."""

        packets = self._packets.setdefault(packet.source_id, [])
        timestamps = [item.captured_at for item in packets]
        index = bisect_left(timestamps, packet.captured_at)
        if index < len(packets) and packets[index].captured_at == packet.captured_at:
            if packet.quality >= packets[index].quality:
                packets[index] = packet
        else:
            packets.insert(index, packet)
        overflow = len(packets) - self.config.maximum_packets_per_source
        if overflow > 0:
            del packets[:overflow]

    def estimate(
        self,
        previous_frame: FramePacket,
        current_frame: FramePacket,
        *,
        previous_detections: Sequence[Detection] = (),
        current_detections: Sequence[Detection] = (),
    ) -> GlobalMotionEstimate:
        del previous_detections, current_detections
        if previous_frame.source_id != current_frame.source_id:
            return GlobalMotionEstimate.identity("telemetry_source_changed", source="telemetry")
        packets = self._packets.get(current_frame.source_id, [])
        previous = self._nearest(packets, previous_frame.captured_at)
        current = self._nearest(packets, current_frame.captured_at)
        if previous is None or current is None:
            return GlobalMotionEstimate.identity("telemetry_alignment_miss", source="telemetry")
        if previous[0].captured_at >= current[0].captured_at:
            return GlobalMotionEstimate.identity("telemetry_samples_not_distinct", source="telemetry")

        previous_packet, previous_error = previous
        current_packet, current_error = current
        alignment_ratio = max(previous_error, current_error) / self.config.maximum_alignment_error_seconds
        quality = min(previous_packet.quality, current_packet.quality) * max(0.0, 1.0 - 0.5 * alignment_ratio)
        if quality < self.config.minimum_quality:
            return GlobalMotionEstimate.identity(
                "telemetry_quality_below_threshold",
                quality=quality,
                source="telemetry",
                prior_quality=quality,
            )

        roll_delta = _wrapped_delta(
            _camera_angle(previous_packet.platform_roll_deg, previous_packet.gimbal_roll_deg),
            _camera_angle(current_packet.platform_roll_deg, current_packet.gimbal_roll_deg),
        )
        pitch_delta = _wrapped_delta(
            _camera_angle(previous_packet.platform_pitch_deg, previous_packet.gimbal_pitch_deg),
            _camera_angle(current_packet.platform_pitch_deg, current_packet.gimbal_pitch_deg),
        )
        yaw_delta = _wrapped_delta(
            _camera_angle(previous_packet.platform_yaw_deg, previous_packet.gimbal_yaw_deg),
            _camera_angle(current_packet.platform_yaw_deg, current_packet.gimbal_yaw_deg),
        )
        if max(abs(roll_delta), abs(pitch_delta), abs(yaw_delta)) > self.config.maximum_angle_delta_degrees:
            return GlobalMotionEstimate.identity(
                "telemetry_rotation_delta_implausible",
                quality=quality,
                source="telemetry",
                prior_quality=quality,
            )

        scene_roll = radians(-roll_delta)
        a = cos(scene_roll)
        b = -sin(scene_roll)
        c = sin(scene_roll)
        d = cos(scene_roll)
        center_x = current_frame.width / 2.0
        center_y = current_frame.height / 2.0
        yaw_shift = -self.config.focal_length_px * tan(radians(yaw_delta))
        pitch_shift = self.config.focal_length_px * tan(radians(pitch_delta))
        translate_x = center_x - a * center_x - b * center_y + yaw_shift
        translate_y = center_y - c * center_x - d * center_y + pitch_shift
        return GlobalMotionEstimate(
            affine_2x3=(a, b, translate_x, c, d, translate_y),
            quality=quality,
            feature_points=0,
            tracked_points=0,
            inliers=0,
            applied=True,
            source="telemetry",
            prior_quality=quality,
        )

    def reset(self) -> None:
        self._packets.clear()

    def _nearest(self, packets: Sequence[TelemetryPacket], timestamp: datetime) -> tuple[TelemetryPacket, float] | None:
        if not packets:
            return None
        timestamps = [packet.captured_at for packet in packets]
        index = bisect_left(timestamps, timestamp)
        candidates = packets[max(0, index - 1) : min(len(packets), index + 1)]
        packet = min(
            candidates,
            key=lambda candidate: (abs((candidate.captured_at - timestamp).total_seconds()), candidate.captured_at),
        )
        error = abs((packet.captured_at - timestamp).total_seconds())
        if error > self.config.maximum_alignment_error_seconds:
            return None
        return packet, error


def _camera_angle(platform_degrees: float, gimbal_degrees: float) -> float:
    value = platform_degrees + gimbal_degrees
    if not isfinite(value):
        raise ValueError("telemetry attitude values must be finite")
    return value


def _wrapped_delta(previous_degrees: float, current_degrees: float) -> float:
    return (current_degrees - previous_degrees + 180.0) % 360.0 - 180.0
