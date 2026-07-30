"""Validated camera calibration contracts and pixel-ray projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, sqrt
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics with OpenCV radial/tangential distortion."""

    width: int
    height: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    calibration_rms_px: float = 0.0

    def __post_init__(self) -> None:
        values = (self.fx_px, self.fy_px, self.cx_px, self.cy_px, self.calibration_rms_px, *self.distortion)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("calibration image dimensions must be positive")
        if not all(isfinite(value) for value in values):
            raise ValueError("camera intrinsics must contain finite values")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if not 0.0 <= self.cx_px <= self.width or not 0.0 <= self.cy_px <= self.height:
            raise ValueError("principal point must lie within the calibration image")
        if self.calibration_rms_px < 0.0:
            raise ValueError("calibration_rms_px must be non-negative")


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """Rigid transform from camera optical coordinates to gimbal coordinates.

    Camera optical coordinates use x right, y down, z forward. Gimbal/body
    coordinates use aerospace FRD: x forward, y right, z down.
    """

    rotation_camera_to_gimbal: tuple[float, float, float, float, float, float, float, float, float]
    translation_camera_in_gimbal_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.rotation_camera_to_gimbal):
            raise ValueError("extrinsic rotation must contain finite values")
        if not all(isfinite(value) for value in self.translation_camera_in_gimbal_m):
            raise ValueError("extrinsic translation must contain finite values")
        rows = _matrix_rows(self.rotation_camera_to_gimbal)
        for row in rows:
            if abs(_dot(row, row) - 1.0) > 1e-3:
                raise ValueError("extrinsic rotation rows must have unit length")
        if any(abs(_dot(rows[left], rows[right])) > 1e-3 for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("extrinsic rotation rows must be orthogonal")
        if abs(_determinant(rows) - 1.0) > 1e-3:
            raise ValueError("extrinsic rotation must be right-handed with determinant one")


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Versioned camera/lens/mount calibration accepted by geometry modules."""

    calibration_id: str
    camera_serial: str
    lens_id: str
    calibrated_at: datetime
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    quality: float

    def __post_init__(self) -> None:
        if not self.calibration_id.strip() or not self.camera_serial.strip() or not self.lens_id.strip():
            raise ValueError("calibration_id, camera_serial, and lens_id must not be empty")
        if self.calibrated_at.tzinfo is None or self.calibrated_at.utcoffset() is None:
            raise ValueError("calibrated_at must be timezone-aware")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("calibration quality must be within [0, 1]")


def image_pixel_to_camera_ray(
    pixel_xy: tuple[float, float],
    intrinsics: CameraIntrinsics,
    *,
    undistortion_iterations: int = 8,
) -> tuple[float, float, float]:
    """Convert one distorted image pixel to a normalized optical ray."""

    u, v = pixel_xy
    if not isfinite(u) or not isfinite(v):
        raise ValueError("pixel coordinates must be finite")
    if undistortion_iterations < 0:
        raise ValueError("undistortion_iterations must be non-negative")
    distorted_x = (u - intrinsics.cx_px) / intrinsics.fx_px
    distorted_y = (v - intrinsics.cy_px) / intrinsics.fy_px
    x = distorted_x
    y = distorted_y
    k1, k2, p1, p2, k3 = intrinsics.distortion
    for _ in range(undistortion_iterations):
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2 + k3 * radius2 * radius2 * radius2
        if abs(radial) < 1e-12:
            raise ValueError("distortion inversion became singular")
        tangential_x = 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        tangential_y = p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (distorted_x - tangential_x) / radial
        y = (distorted_y - tangential_y) / radial
    return _normalize((x, y, 1.0))


def calibration_quality(calibration: CameraCalibration, *, maximum_rms_px: float = 1.0) -> float:
    """Combine declared calibration quality with reprojection error."""

    if maximum_rms_px <= 0.0 or not isfinite(maximum_rms_px):
        raise ValueError("maximum_rms_px must be finite and positive")
    rms_factor = max(0.0, 1.0 - calibration.intrinsics.calibration_rms_px / maximum_rms_px)
    return min(calibration.quality, rms_factor)


def load_camera_calibration(path: Path) -> CameraCalibration:
    """Load a strict, versioned camera calibration JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("camera calibration root must be an object")
    _require_keys(
        payload,
        {
            "schema_version",
            "calibration_id",
            "camera_serial",
            "lens_id",
            "calibrated_at",
            "quality",
            "intrinsics",
            "extrinsics",
        },
        "camera calibration",
    )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported camera calibration schema_version")
    intrinsics = _object(payload["intrinsics"], "intrinsics")
    _require_keys(
        intrinsics,
        {"width", "height", "fx_px", "fy_px", "cx_px", "cy_px", "distortion", "calibration_rms_px"},
        "intrinsics",
    )
    extrinsics = _object(payload["extrinsics"], "extrinsics")
    _require_keys(
        extrinsics,
        {"rotation_camera_to_gimbal", "translation_camera_in_gimbal_m"},
        "extrinsics",
    )
    calibrated_at = datetime.fromisoformat(str(payload["calibrated_at"]).replace("Z", "+00:00"))
    return CameraCalibration(
        calibration_id=str(payload["calibration_id"]),
        camera_serial=str(payload["camera_serial"]),
        lens_id=str(payload["lens_id"]),
        calibrated_at=calibrated_at,
        quality=float(payload["quality"]),
        intrinsics=CameraIntrinsics(
            width=int(intrinsics["width"]),
            height=int(intrinsics["height"]),
            fx_px=float(intrinsics["fx_px"]),
            fy_px=float(intrinsics["fy_px"]),
            cx_px=float(intrinsics["cx_px"]),
            cy_px=float(intrinsics["cy_px"]),
            distortion=_float_tuple5(intrinsics["distortion"], "distortion"),
            calibration_rms_px=float(intrinsics["calibration_rms_px"]),
        ),
        extrinsics=CameraExtrinsics(
            rotation_camera_to_gimbal=_float_tuple9(
                extrinsics["rotation_camera_to_gimbal"],
                "rotation_camera_to_gimbal",
            ),
            translation_camera_in_gimbal_m=_float_tuple3(
                extrinsics["translation_camera_in_gimbal_m"],
                "translation_camera_in_gimbal_m",
            ),
        ),
    )


def _matrix_rows(
    values: tuple[float, float, float, float, float, float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return ((values[0], values[1], values[2]), (values[3], values[4], values[5]), (values[6], values[7], values[8]))


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return sum(left * right for left, right in zip(first, second, strict=True))


def _determinant(
    rows: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> float:
    first, second, third = rows
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(_dot(vector, vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero-length ray")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _require_keys(payload: dict[str, Any], expected: set[str], field_name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{field_name} keys mismatch: missing={missing}, unknown={unknown}")


def _float_sequence(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    result = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{field_name} must contain finite values")
    return result


def _float_tuple3(value: Any, field_name: str) -> tuple[float, float, float]:
    result = _float_sequence(value, 3, field_name)
    return (result[0], result[1], result[2])


def _float_tuple5(value: Any, field_name: str) -> tuple[float, float, float, float, float]:
    result = _float_sequence(value, 5, field_name)
    return (result[0], result[1], result[2], result[3], result[4])


def _float_tuple9(
    value: Any,
    field_name: str,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    result = _float_sequence(value, 9, field_name)
    return (result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8])
