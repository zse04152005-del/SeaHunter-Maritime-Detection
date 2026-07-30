"""Robust visual global-motion estimation for moving maritime cameras."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite, sqrt
from typing import Any, Protocol

from seahunter.schemas import Detection, FramePacket

IDENTITY_AFFINE: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True, slots=True)
class GlobalMotionEstimate:
    """One previous-frame to current-frame affine camera-motion estimate."""

    affine_2x3: tuple[float, float, float, float, float, float]
    quality: float
    feature_points: int
    tracked_points: int
    inliers: int
    applied: bool
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.affine_2x3) != 6 or not all(isfinite(value) for value in self.affine_2x3):
            raise ValueError("global-motion affine must contain six finite coefficients")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("global-motion quality must be within [0, 1]")
        if min(self.feature_points, self.tracked_points, self.inliers) < 0:
            raise ValueError("global-motion point counts must be non-negative")
        if self.inliers > self.tracked_points or self.tracked_points > self.feature_points:
            raise ValueError("global-motion counts must satisfy inliers <= tracked <= features")
        if self.applied and self.fallback_reason is not None:
            raise ValueError("an applied global-motion estimate cannot have a fallback reason")
        if not self.applied and (self.fallback_reason is None or not self.fallback_reason.strip()):
            raise ValueError("a rejected global-motion estimate requires a fallback reason")

    @property
    def translation_xy(self) -> tuple[float, float]:
        return self.affine_2x3[2], self.affine_2x3[5]

    @classmethod
    def identity(
        cls,
        reason: str,
        *,
        quality: float = 0.0,
        feature_points: int = 0,
        tracked_points: int = 0,
        inliers: int = 0,
    ) -> GlobalMotionEstimate:
        return cls(
            affine_2x3=IDENTITY_AFFINE,
            quality=quality,
            feature_points=feature_points,
            tracked_points=tracked_points,
            inliers=inliers,
            applied=False,
            fallback_reason=reason,
        )


class GlobalMotionEstimator(Protocol):
    """Contract for auditable visual camera-motion estimators."""

    @property
    def estimator_id(self) -> str:
        """Return a stable algorithm/configuration identifier."""

    def estimate(
        self,
        previous_frame: FramePacket,
        current_frame: FramePacket,
        *,
        previous_detections: Sequence[Detection] = (),
        current_detections: Sequence[Detection] = (),
    ) -> GlobalMotionEstimate:
        """Estimate a warp from the previous frame into the current frame."""

    def reset(self) -> None:
        """Clear estimator state, if any."""


@dataclass(frozen=True, slots=True)
class SparseOpticalFlowConfig:
    """Quality gates for sparse LK optical flow and RANSAC affine fitting."""

    downscale_factor: int = 2
    max_corners: int = 500
    quality_level: float = 0.01
    minimum_feature_distance: float = 8.0
    block_size: int = 3
    lk_window_size: int = 21
    lk_max_level: int = 3
    max_optical_flow_error: float = 30.0
    ransac_reprojection_threshold: float = 3.0
    minimum_inliers: int = 12
    minimum_inlier_ratio: float = 0.35
    exclusion_padding_px: float = 6.0
    maximum_translation_ratio: float = 0.35
    minimum_scale: float = 0.8
    maximum_scale: float = 1.2
    maximum_rotation_degrees: float = 20.0

    def __post_init__(self) -> None:
        if self.downscale_factor <= 0:
            raise ValueError("downscale_factor must be positive")
        if self.max_corners <= 0 or not 3 <= self.minimum_inliers <= self.max_corners:
            raise ValueError("minimum_inliers must be within [3, max_corners]")
        if not 0.0 < self.quality_level <= 1.0:
            raise ValueError("quality_level must be within (0, 1]")
        if self.minimum_feature_distance < 0 or self.exclusion_padding_px < 0:
            raise ValueError("feature distance and exclusion padding must be non-negative")
        if self.block_size <= 1 or self.block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer greater than 1")
        if self.lk_window_size <= 1 or self.lk_window_size % 2 == 0 or self.lk_max_level < 0:
            raise ValueError("LK window must be odd and greater than 1; pyramid level must be non-negative")
        if self.max_optical_flow_error <= 0 or self.ransac_reprojection_threshold <= 0:
            raise ValueError("optical-flow and RANSAC error thresholds must be positive")
        if not 0.0 < self.minimum_inlier_ratio <= 1.0:
            raise ValueError("minimum_inlier_ratio must be within (0, 1]")
        if not 0.0 < self.maximum_translation_ratio <= 1.0:
            raise ValueError("maximum_translation_ratio must be within (0, 1]")
        if not 0.0 < self.minimum_scale <= 1.0 <= self.maximum_scale:
            raise ValueError("scale bounds must satisfy 0 < minimum <= 1 <= maximum")
        if not 0.0 <= self.maximum_rotation_degrees <= 180.0:
            raise ValueError("maximum_rotation_degrees must be within [0, 180]")


class SparseOpticalFlowGMC:
    """Estimate global affine motion while excluding detected foreground boxes."""

    def __init__(self, config: SparseOpticalFlowConfig | None = None) -> None:
        self.config = config or SparseOpticalFlowConfig()
        self._np: Any = importlib.import_module("numpy")
        try:
            self._cv2: Any = importlib.import_module("cv2")
        except ModuleNotFoundError as exc:
            raise RuntimeError("visual GMC requires the 'video' project extra") from exc
        self._estimator_id = (
            "sparse-flow-gmc:v1"
            f":down={self.config.downscale_factor}"
            f":corners={self.config.max_corners}"
            f":inliers={self.config.minimum_inliers}"
            f":ratio={self.config.minimum_inlier_ratio:g}"
            f":translation={self.config.maximum_translation_ratio:g}"
            f":scale={self.config.minimum_scale:g}/{self.config.maximum_scale:g}"
            f":rotation={self.config.maximum_rotation_degrees:g}"
        )

    @property
    def estimator_id(self) -> str:
        return self._estimator_id

    def estimate(
        self,
        previous_frame: FramePacket,
        current_frame: FramePacket,
        *,
        previous_detections: Sequence[Detection] = (),
        current_detections: Sequence[Detection] = (),
    ) -> GlobalMotionEstimate:
        if previous_frame.width != current_frame.width or previous_frame.height != current_frame.height:
            return GlobalMotionEstimate.identity("frame_dimensions_changed")
        previous_gray = self._prepare_gray(previous_frame)
        current_gray = self._prepare_gray(current_frame)
        if previous_gray is None or current_gray is None:
            return GlobalMotionEstimate.identity("missing_or_invalid_payload")

        previous_small = self._downscale(previous_gray)
        current_small = self._downscale(current_gray)
        mask = self._feature_mask(previous_small.shape[:2], previous_detections)
        points = self._cv2.goodFeaturesToTrack(
            previous_small,
            mask=mask,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.minimum_feature_distance / self.config.downscale_factor,
            blockSize=self.config.block_size,
        )
        if points is None:
            return GlobalMotionEstimate.identity("no_features")
        feature_points = int(len(points))
        if feature_points < self.config.minimum_inliers:
            return GlobalMotionEstimate.identity("insufficient_features", feature_points=feature_points)

        current_points, status, errors = self._cv2.calcOpticalFlowPyrLK(
            previous_small,
            current_small,
            points,
            None,
            winSize=(self.config.lk_window_size, self.config.lk_window_size),
            maxLevel=self.config.lk_max_level,
            criteria=(
                self._cv2.TERM_CRITERIA_EPS | self._cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if current_points is None or status is None:
            return GlobalMotionEstimate.identity("optical_flow_failed", feature_points=feature_points)

        previous_xy = points.reshape(-1, 2)
        current_xy = current_points.reshape(-1, 2)
        valid = status.reshape(-1).astype(bool)
        valid &= self._np.isfinite(previous_xy).all(axis=1) & self._np.isfinite(current_xy).all(axis=1)
        if errors is not None:
            flow_error = errors.reshape(-1)
            valid &= self._np.isfinite(flow_error) & (flow_error <= self.config.max_optical_flow_error)
        valid &= self._points_inside_frame(current_xy, current_small.shape[:2])
        valid &= self._points_outside_detections(current_xy, current_detections)
        previous_xy = previous_xy[valid]
        current_xy = current_xy[valid]
        tracked_points = int(len(previous_xy))
        if tracked_points < self.config.minimum_inliers:
            return GlobalMotionEstimate.identity(
                "insufficient_tracked_points",
                feature_points=feature_points,
                tracked_points=tracked_points,
            )

        affine, inlier_mask = self._cv2.estimateAffinePartial2D(
            previous_xy,
            current_xy,
            method=self._cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reprojection_threshold,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inlier_mask is None or not bool(self._np.isfinite(affine).all()):
            return GlobalMotionEstimate.identity(
                "affine_estimation_failed",
                feature_points=feature_points,
                tracked_points=tracked_points,
            )
        inliers = int(inlier_mask.reshape(-1).astype(bool).sum())
        inlier_ratio = inliers / tracked_points
        if inliers < self.config.minimum_inliers or inlier_ratio < self.config.minimum_inlier_ratio:
            return GlobalMotionEstimate.identity(
                "insufficient_ransac_support",
                quality=inlier_ratio,
                feature_points=feature_points,
                tracked_points=tracked_points,
                inliers=inliers,
            )

        affine = self._to_original_resolution(affine)
        rejection_reason = self._plausibility_rejection(affine, current_frame)
        if rejection_reason is not None:
            return GlobalMotionEstimate.identity(
                rejection_reason,
                quality=inlier_ratio,
                feature_points=feature_points,
                tracked_points=tracked_points,
                inliers=inliers,
            )
        coefficients = (
            float(affine[0, 0]),
            float(affine[0, 1]),
            float(affine[0, 2]),
            float(affine[1, 0]),
            float(affine[1, 1]),
            float(affine[1, 2]),
        )
        return GlobalMotionEstimate(
            affine_2x3=coefficients,
            quality=inlier_ratio,
            feature_points=feature_points,
            tracked_points=tracked_points,
            inliers=inliers,
            applied=True,
        )

    def reset(self) -> None:
        return None

    def _prepare_gray(self, frame: FramePacket) -> Any | None:
        if frame.payload is None:
            return None
        array = self._np.asarray(frame.payload)
        if array.ndim not in (2, 3) or array.shape[0] != frame.height or array.shape[1] != frame.width:
            return None
        if not bool(self._np.isfinite(array).all()):
            return None
        if array.ndim == 3:
            if array.shape[2] == 3:
                array = self._cv2.cvtColor(array, self._cv2.COLOR_BGR2GRAY)
            elif array.shape[2] == 4:
                array = self._cv2.cvtColor(array, self._cv2.COLOR_BGRA2GRAY)
            else:
                return None
        if array.dtype != self._np.uint8:
            array = self._cv2.normalize(array, None, 0, 255, self._cv2.NORM_MINMAX).astype(self._np.uint8)
        return self._np.ascontiguousarray(array)

    def _downscale(self, gray: Any) -> Any:
        if self.config.downscale_factor == 1:
            return gray
        width = max(1, gray.shape[1] // self.config.downscale_factor)
        height = max(1, gray.shape[0] // self.config.downscale_factor)
        return self._cv2.resize(gray, (width, height), interpolation=self._cv2.INTER_AREA)

    def _feature_mask(self, shape: tuple[int, int], detections: Sequence[Detection]) -> Any:
        mask = self._np.full(shape, 255, dtype=self._np.uint8)
        scale = float(self.config.downscale_factor)
        padding = self.config.exclusion_padding_px / scale
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            left = max(0, int(x1 / scale - padding))
            top = max(0, int(y1 / scale - padding))
            right = min(shape[1] - 1, int(x2 / scale + padding))
            bottom = min(shape[0] - 1, int(y2 / scale + padding))
            if right >= left and bottom >= top:
                self._cv2.rectangle(mask, (left, top), (right, bottom), 0, thickness=-1)
        return mask

    def _points_inside_frame(self, points: Any, shape: tuple[int, int]) -> Any:
        return (
            (points[:, 0] >= 0.0)
            & (points[:, 0] < float(shape[1]))
            & (points[:, 1] >= 0.0)
            & (points[:, 1] < float(shape[0]))
        )

    def _points_outside_detections(self, points: Any, detections: Sequence[Detection]) -> Any:
        keep = self._np.ones(len(points), dtype=bool)
        scale = float(self.config.downscale_factor)
        padding = self.config.exclusion_padding_px / scale
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            inside = (
                (points[:, 0] >= x1 / scale - padding)
                & (points[:, 0] <= x2 / scale + padding)
                & (points[:, 1] >= y1 / scale - padding)
                & (points[:, 1] <= y2 / scale + padding)
            )
            keep &= ~inside
        return keep

    def _to_original_resolution(self, affine: Any) -> Any:
        affine = self._np.asarray(affine, dtype=float).copy()
        affine[:, 2] *= float(self.config.downscale_factor)
        return affine

    def _plausibility_rejection(self, affine: Any, frame: FramePacket) -> str | None:
        a, _, tx = (float(value) for value in affine[0])
        c, _, ty = (float(value) for value in affine[1])
        scale = sqrt(a * a + c * c)
        rotation = abs(degrees(atan2(c, a)))
        translation_ratio = hypot(tx, ty) / hypot(float(frame.width), float(frame.height))
        if not self.config.minimum_scale <= scale <= self.config.maximum_scale:
            return "implausible_scale"
        if rotation > self.config.maximum_rotation_degrees:
            return "implausible_rotation"
        if translation_ratio > self.config.maximum_translation_ratio:
            return "implausible_translation"
        return None
