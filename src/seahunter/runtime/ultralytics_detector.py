"""Framework adapter from the legacy Ultralytics model to SeaHunter contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seahunter.schemas import Detection, FramePacket

from .legacy_ultralytics import activate_legacy_ultralytics


class UltralyticsDetector:
    """Run one-frame-at-a-time inference without retaining unbounded video results."""

    def __init__(
        self,
        weights: Path,
        *,
        repo_root: Path | None = None,
        imgsz: int = 1024,
        confidence: float = 0.25,
        iou: float = 0.70,
        device: str = "0",
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be within [0, 1]")

        ultralytics = activate_legacy_ultralytics(repo_root)
        self._model: Any = ultralytics.YOLO(str(weights))
        self._weights = weights.resolve()
        self._imgsz = imgsz
        self._confidence = confidence
        self._iou = iou
        self._device = device
        self._detector_id = f"ultralytics-8.3.234:{self._weights.stem}:{device}"

    @property
    def detector_id(self) -> str:
        return self._detector_id

    def infer(self, frame: FramePacket) -> list[Detection]:
        """Infer one decoded frame and normalize all outputs."""

        if frame.payload is None:
            raise ValueError("FramePacket.payload is required for inference")

        results = self._model.predict(
            source=frame.payload,
            imgsz=self._imgsz,
            conf=self._confidence,
            iou=self._iou,
            device=self._device,
            verbose=False,
            save=False,
            stream=False,
        )
        if len(results) != 1:
            raise RuntimeError(f"single-frame inference returned {len(results)} results")

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        names: dict[int, str] = result.names

        return [
            Detection(
                bbox_xyxy=tuple(float(value) for value in xyxy),  # type: ignore[arg-type]
                class_id=int(class_id),
                class_name=names[int(class_id)],
                confidence=float(score),
                detector_id=self.detector_id,
            )
            for xyxy, score, class_id in zip(coordinates, confidences, classes, strict=True)
        ]
