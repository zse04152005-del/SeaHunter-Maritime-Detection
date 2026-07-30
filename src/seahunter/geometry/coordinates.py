"""Explicit ENU/NED, body, gimbal, camera, and geodetic transforms."""

from __future__ import annotations

from math import atan2, cos, degrees, isfinite, radians, sin, sqrt

from seahunter.schemas import TelemetryPacket

from .calibration import CameraCalibration

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]

_WGS84_A_M = 6_378_137.0
_WGS84_E2 = 6.69437999014e-3


def enu_to_ned(vector_enu: Vector3) -> Vector3:
    east, north, up = vector_enu
    return (north, east, -up)


def ned_to_enu(vector_ned: Vector3) -> Vector3:
    north, east, down = vector_ned
    return (east, north, -down)


def rotation_from_roll_pitch_yaw(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Matrix3:
    """Return aerospace ZYX rotation from a local FRD frame to its parent."""

    if not all(isfinite(value) for value in (roll_deg, pitch_deg, yaw_deg)):
        raise ValueError("roll, pitch, and yaw must be finite")
    roll = radians(roll_deg)
    pitch = radians(pitch_deg)
    yaw = radians(yaw_deg)
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def camera_ray_to_enu(ray_camera: Vector3, telemetry: TelemetryPacket, calibration: CameraCalibration) -> Vector3:
    """Rotate an optical ray through mount, gimbal, platform, NED, then ENU."""

    camera_to_gimbal = _unflatten(calibration.extrinsics.rotation_camera_to_gimbal)
    gimbal_to_body = rotation_from_roll_pitch_yaw(
        telemetry.gimbal_roll_deg,
        telemetry.gimbal_pitch_deg,
        telemetry.gimbal_yaw_deg,
    )
    body_to_ned = rotation_from_roll_pitch_yaw(
        telemetry.platform_roll_deg,
        telemetry.platform_pitch_deg,
        telemetry.platform_yaw_deg,
    )
    ray_gimbal = matvec(camera_to_gimbal, ray_camera)
    ray_body = matvec(gimbal_to_body, ray_gimbal)
    ray_ned = matvec(body_to_ned, ray_body)
    return _normalize(ned_to_enu(ray_ned))


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> Vector3:
    if not all(isfinite(value) for value in (latitude_deg, longitude_deg, altitude_m)):
        raise ValueError("geodetic coordinates must be finite")
    if not -90.0 <= latitude_deg <= 90.0 or not -180.0 <= longitude_deg <= 180.0:
        raise ValueError("invalid geodetic coordinate range")
    latitude = radians(latitude_deg)
    longitude = radians(longitude_deg)
    normal = _WGS84_A_M / sqrt(1.0 - _WGS84_E2 * sin(latitude) ** 2)
    return (
        (normal + altitude_m) * cos(latitude) * cos(longitude),
        (normal + altitude_m) * cos(latitude) * sin(longitude),
        (normal * (1.0 - _WGS84_E2) + altitude_m) * sin(latitude),
    )


def ecef_delta_to_enu(delta_ecef: Vector3, latitude_deg: float, longitude_deg: float) -> Vector3:
    latitude = radians(latitude_deg)
    longitude = radians(longitude_deg)
    x, y, z = delta_ecef
    return (
        -sin(longitude) * x + cos(longitude) * y,
        -sin(latitude) * cos(longitude) * x - sin(latitude) * sin(longitude) * y + cos(latitude) * z,
        cos(latitude) * cos(longitude) * x + cos(latitude) * sin(longitude) * y + sin(latitude) * z,
    )


def enu_offset_to_geodetic(origin_latitude_deg: float, origin_longitude_deg: float, offset_enu_m: Vector3) -> Vector3:
    """Convert a local ENU offset to geodetic coordinates using WGS84 ECEF."""

    origin = geodetic_to_ecef(origin_latitude_deg, origin_longitude_deg, 0.0)
    latitude = radians(origin_latitude_deg)
    longitude = radians(origin_longitude_deg)
    east, north, up = offset_enu_m
    delta = (
        -sin(longitude) * east - sin(latitude) * cos(longitude) * north + cos(latitude) * cos(longitude) * up,
        cos(longitude) * east - sin(latitude) * sin(longitude) * north + cos(latitude) * sin(longitude) * up,
        cos(latitude) * north + sin(latitude) * up,
    )
    return ecef_to_geodetic((origin[0] + delta[0], origin[1] + delta[1], origin[2] + delta[2]))


def ecef_to_geodetic(ecef_m: Vector3) -> Vector3:
    x, y, z = ecef_m
    longitude = atan2(y, x)
    horizontal = sqrt(x * x + y * y)
    latitude = atan2(z, horizontal * (1.0 - _WGS84_E2))
    altitude = 0.0
    for _ in range(8):
        normal = _WGS84_A_M / sqrt(1.0 - _WGS84_E2 * sin(latitude) ** 2)
        altitude = horizontal / max(cos(latitude), 1e-12) - normal
        latitude = atan2(z, horizontal * (1.0 - _WGS84_E2 * normal / (normal + altitude)))
    return (degrees(latitude), degrees(longitude), altitude)


def matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return (
        sum(matrix[0][index] * vector[index] for index in range(3)),
        sum(matrix[1][index] * vector[index] for index in range(3)),
        sum(matrix[2][index] * vector[index] for index in range(3)),
    )


def _unflatten(values: tuple[float, float, float, float, float, float, float, float, float]) -> Matrix3:
    return ((values[0], values[1], values[2]), (values[3], values[4], values[5]), (values[6], values[7], values[8]))


def _normalize(vector: Vector3) -> Vector3:
    norm = sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)
