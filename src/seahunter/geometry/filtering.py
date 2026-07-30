"""Constant-velocity ENU extended-Kalman baseline with covariance output."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

from seahunter.schemas import GeoEstimate

from .coordinates import ecef_delta_to_enu, geodetic_to_ecef


@dataclass(frozen=True, slots=True)
class ENUPositionMeasurement:
    track_id: int
    captured_at: datetime
    east_m: float
    north_m: float
    covariance_en_m2: tuple[float, float, float, float]
    quality: float

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        if not all(isfinite(value) for value in (self.east_m, self.north_m, *self.covariance_en_m2)):
            raise ValueError("ENU measurement must contain finite values")
        if self.covariance_en_m2[0] <= 0.0 or self.covariance_en_m2[3] <= 0.0:
            raise ValueError("ENU measurement covariance diagonal must be positive")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("measurement quality must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class ENUKinematicState:
    track_id: int
    captured_at: datetime
    east_m: float
    north_m: float
    east_velocity_m_s: float
    north_velocity_m_s: float
    covariance: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    quality: float


@dataclass(frozen=True, slots=True)
class ENUFilterConfig:
    acceleration_sigma_m_s2: float = 2.0
    initial_velocity_sigma_m_s: float = 10.0
    maximum_gap_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.acceleration_sigma_m_s2 <= 0.0 or self.initial_velocity_sigma_m_s <= 0.0:
            raise ValueError("filter noise values must be positive")
        if self.maximum_gap_seconds <= 0.0:
            raise ValueError("maximum_gap_seconds must be positive")


@dataclass(slots=True)
class _FilterState:
    captured_at: datetime
    mean: Any
    covariance: Any


class ENUConstantVelocityEKF:
    """Track-independent linear EKF (a constant-velocity EKF special case)."""

    def __init__(self, config: ENUFilterConfig | None = None) -> None:
        self.config = config or ENUFilterConfig()
        self._np: Any = importlib.import_module("numpy")
        self._states: dict[int, _FilterState] = {}

    def update(self, measurement: ENUPositionMeasurement) -> ENUKinematicState:
        state = self._states.get(measurement.track_id)
        if state is None:
            state = self._initialize(measurement)
        else:
            delta_seconds = (measurement.captured_at - state.captured_at).total_seconds()
            if delta_seconds <= 0.0:
                raise ValueError("filter measurements must increase strictly per track")
            if delta_seconds > self.config.maximum_gap_seconds:
                state = self._initialize(measurement)
            else:
                state = self._predict_update(state, measurement, delta_seconds)
        self._states[measurement.track_id] = state
        return self._to_output(measurement, state)

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._states.clear()
        else:
            self._states.pop(track_id, None)

    def _initialize(self, measurement: ENUPositionMeasurement) -> _FilterState:
        np = self._np
        mean = np.array([measurement.east_m, measurement.north_m, 0.0, 0.0], dtype=float)
        covariance = np.zeros((4, 4), dtype=float)
        covariance[:2, :2] = np.array(measurement.covariance_en_m2, dtype=float).reshape(2, 2)
        covariance[2, 2] = self.config.initial_velocity_sigma_m_s**2
        covariance[3, 3] = self.config.initial_velocity_sigma_m_s**2
        return _FilterState(measurement.captured_at, mean, covariance)

    def _predict_update(
        self,
        state: _FilterState,
        measurement: ENUPositionMeasurement,
        delta_seconds: float,
    ) -> _FilterState:
        np = self._np
        transition = np.array(
            [
                [1.0, 0.0, delta_seconds, 0.0],
                [0.0, 1.0, 0.0, delta_seconds],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        acceleration_variance = self.config.acceleration_sigma_m_s2**2
        process_axis = (
            np.array(
                [
                    [delta_seconds**4 / 4.0, delta_seconds**3 / 2.0],
                    [delta_seconds**3 / 2.0, delta_seconds**2],
                ],
                dtype=float,
            )
            * acceleration_variance
        )
        process = np.zeros((4, 4), dtype=float)
        process[np.ix_([0, 2], [0, 2])] = process_axis
        process[np.ix_([1, 3], [1, 3])] = process_axis
        predicted_mean = transition @ state.mean
        predicted_covariance = transition @ state.covariance @ transition.T + process
        observation = np.array([measurement.east_m, measurement.north_m], dtype=float)
        observation_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        measurement_covariance = np.array(measurement.covariance_en_m2, dtype=float).reshape(2, 2)
        innovation = observation - observation_matrix @ predicted_mean
        innovation_covariance = (
            observation_matrix @ predicted_covariance @ observation_matrix.T + measurement_covariance
        )
        gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        updated_mean = predicted_mean + gain @ innovation
        identity = np.eye(4, dtype=float)
        residual_projection = identity - gain @ observation_matrix
        updated_covariance = (
            residual_projection @ predicted_covariance @ residual_projection.T + gain @ measurement_covariance @ gain.T
        )
        return _FilterState(measurement.captured_at, updated_mean, updated_covariance)

    def _to_output(self, measurement: ENUPositionMeasurement, state: _FilterState) -> ENUKinematicState:
        flattened = tuple(float(value) for value in state.covariance.reshape(-1).tolist())
        covariance = _tuple16(flattened)
        position_sigma = max(0.0, float(state.covariance[0, 0]) + float(state.covariance[1, 1])) ** 0.5
        quality = measurement.quality / (1.0 + position_sigma / 50.0)
        return ENUKinematicState(
            track_id=measurement.track_id,
            captured_at=state.captured_at,
            east_m=float(state.mean[0]),
            north_m=float(state.mean[1]),
            east_velocity_m_s=float(state.mean[2]),
            north_velocity_m_s=float(state.mean[3]),
            covariance=covariance,
            quality=quality,
        )


def geo_to_enu_measurement(
    estimate: GeoEstimate,
    *,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
) -> ENUPositionMeasurement | None:
    if (
        not estimate.absolute
        or estimate.latitude_deg is None
        or estimate.longitude_deg is None
        or estimate.captured_at is None
        or estimate.covariance_en_m2 is None
    ):
        return None
    origin = geodetic_to_ecef(origin_latitude_deg, origin_longitude_deg, 0.0)
    target = geodetic_to_ecef(estimate.latitude_deg, estimate.longitude_deg, 0.0)
    east, north, _ = ecef_delta_to_enu(
        (target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]),
        origin_latitude_deg,
        origin_longitude_deg,
    )
    return ENUPositionMeasurement(
        track_id=estimate.track_id,
        captured_at=estimate.captured_at,
        east_m=east,
        north_m=north,
        covariance_en_m2=estimate.covariance_en_m2,
        quality=estimate.quality,
    )


def _tuple16(
    values: tuple[float, ...],
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    if len(values) != 16:
        raise ValueError("filter covariance must contain 16 values")
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        values[13],
        values[14],
        values[15],
    )
