"""Video source and ingestion primitives."""

from .buffer import BufferClosed, BufferStats, LatestItemBuffer
from .ingestion import FrameIngestWorker, FrameReader, IngestionFailed, IngestionStats
from .reader import (
    ExponentialBackoff,
    OpenCVFrameReader,
    OpenCVReaderConfig,
    ReaderClosed,
    ReaderStats,
    StreamUnavailable,
    VideoReaderError,
    VideoSourceOpenError,
)
from .sources import VideoSource, parse_video_source

__all__ = [
    "BufferClosed",
    "BufferStats",
    "ExponentialBackoff",
    "FrameIngestWorker",
    "FrameReader",
    "IngestionFailed",
    "IngestionStats",
    "LatestItemBuffer",
    "OpenCVFrameReader",
    "OpenCVReaderConfig",
    "ReaderClosed",
    "ReaderStats",
    "StreamUnavailable",
    "VideoReaderError",
    "VideoSource",
    "VideoSourceOpenError",
    "parse_video_source",
]
