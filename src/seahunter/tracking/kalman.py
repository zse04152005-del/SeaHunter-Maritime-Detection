"""Constant-velocity Kalman filter for bounding boxes in center-XYWH form."""

from __future__ import annotations

import importlib
from typing import Any


class KalmanXYWH:
    """Eight-dimensional constant-velocity filter for ``(cx, cy, w, h)``."""

    def __init__(self) -> None:
        self._np: Any = importlib.import_module("numpy")
        self._measurement_matrix = self._np.eye(4, 8, dtype=float)

    def initiate(self, bbox_xyxy: tuple[float, float, float, float]) -> tuple[Any, Any]:
        measurement = self._xyxy_to_xywh(bbox_xyxy)
        mean = self._np.r_[measurement, self._np.zeros(4, dtype=float)]
        scale = max(float(measurement[2]), float(measurement[3]), 1.0)
        position_std = scale / 10.0
        velocity_std = scale / 16.0
        standard_deviation = self._np.array(
            [
                position_std,
                position_std,
                position_std,
                position_std,
                velocity_std,
                velocity_std,
                velocity_std,
                velocity_std,
            ],
            dtype=float,
        )
        covariance = self._np.diag(standard_deviation**2)
        return mean, covariance

    def predict(self, mean: Any, covariance: Any, *, delta_frames: int = 1) -> tuple[Any, Any]:
        if delta_frames <= 0:
            raise ValueError("delta_frames must be positive")
        motion_matrix = self._np.eye(8, dtype=float)
        for index in range(4):
            motion_matrix[index, index + 4] = float(delta_frames)

        scale = max(float(mean[2]), float(mean[3]), 1.0)
        position_std = scale / 20.0 * max(1.0, float(delta_frames))
        velocity_std = scale / 160.0 * max(1.0, float(delta_frames))
        motion_covariance = self._np.diag(
            self._np.array(
                [position_std] * 4 + [velocity_std] * 4,
                dtype=float,
            )
            ** 2
        )
        predicted_mean = motion_matrix @ mean
        predicted_mean[2:4] = self._np.maximum(predicted_mean[2:4], 1.0)
        predicted_covariance = motion_matrix @ covariance @ motion_matrix.T + motion_covariance
        return predicted_mean, predicted_covariance

    def update(
        self,
        mean: Any,
        covariance: Any,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> tuple[Any, Any]:
        measurement = self._xyxy_to_xywh(bbox_xyxy)
        projected_mean = self._measurement_matrix @ mean
        scale = max(float(mean[2]), float(mean[3]), 1.0)
        measurement_std = scale / 20.0
        innovation_covariance = self._np.diag(self._np.array([measurement_std] * 4, dtype=float) ** 2)
        projected_covariance = (
            self._measurement_matrix @ covariance @ self._measurement_matrix.T + innovation_covariance
        )
        kalman_gain = covariance @ self._measurement_matrix.T @ self._np.linalg.inv(projected_covariance)
        innovation = measurement - projected_mean
        updated_mean = mean + kalman_gain @ innovation
        updated_mean[2:4] = self._np.maximum(updated_mean[2:4], 1.0)
        identity = self._np.eye(8, dtype=float)
        updated_covariance = (identity - kalman_gain @ self._measurement_matrix) @ covariance
        return updated_mean, updated_covariance

    @staticmethod
    def to_xyxy(mean: Any) -> tuple[float, float, float, float]:
        cx, cy, width, height = (float(value) for value in mean[:4])
        half_width = max(width, 1.0) / 2.0
        half_height = max(height, 1.0) / 2.0
        return cx - half_width, cy - half_height, cx + half_width, cy + half_height

    def covariance_xy(self, covariance: Any) -> tuple[float, float, float, float]:
        return (
            float(covariance[0, 0]),
            float(covariance[0, 1]),
            float(covariance[1, 0]),
            float(covariance[1, 1]),
        )

    def _xyxy_to_xywh(self, bbox_xyxy: tuple[float, float, float, float]) -> Any:
        x1, y1, x2, y2 = bbox_xyxy
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        return self._np.array([x1 + width / 2.0, y1 + height / 2.0, width, height], dtype=float)
