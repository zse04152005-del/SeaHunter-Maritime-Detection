"""Video source parsing without treating network streams as local paths."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

from seahunter.schemas import SourceKind


@dataclass(frozen=True, slots=True)
class VideoSource:
    """A normalized video source specification."""

    raw: str
    kind: SourceKind
    location: str

    @property
    def is_network(self) -> bool:
        return self.kind in {SourceKind.RTSP, SourceKind.SRT, SourceKind.HTTP}

    @property
    def is_live(self) -> bool:
        """Return whether transient read failures should trigger reconnection."""

        return self.kind is not SourceKind.FILE

    def suggested_id(self) -> str:
        """Return a stable identifier without exposing URL credentials."""

        if self.kind is SourceKind.FILE:
            return Path(self.location).stem or "video-file"
        if self.kind is SourceKind.DEVICE:
            return f"device-{self.location}"

        parsed = urlparse(self.location)
        host = (parsed.hostname or "stream").replace(".", "-")
        digest = sha256(self.location.encode("utf-8")).hexdigest()[:8]
        return f"{self.kind.value}-{host}-{digest}"


def parse_video_source(value: str) -> VideoSource:
    """Classify a video source without probing the network or filesystem."""

    cleaned = value.strip()
    if not cleaned:
        raise ValueError("video source must not be empty")

    if cleaned.isdecimal():
        return VideoSource(raw=value, kind=SourceKind.DEVICE, location=cleaned)

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme == "rtsp":
        return VideoSource(raw=value, kind=SourceKind.RTSP, location=cleaned)
    if scheme == "srt":
        return VideoSource(raw=value, kind=SourceKind.SRT, location=cleaned)
    if scheme in {"http", "https"}:
        return VideoSource(raw=value, kind=SourceKind.HTTP, location=cleaned)
    if scheme and len(scheme) > 1:
        raise ValueError(f"unsupported video source scheme: {scheme}")

    return VideoSource(raw=value, kind=SourceKind.FILE, location=str(Path(cleaned)))
