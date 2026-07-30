"""Video source and ingestion primitives."""

from .buffer import BufferClosed, BufferStats, LatestItemBuffer
from .capabilities import DecoderBuildReport, probe_opencv_decoder_build
from .ingestion import FrameIngestWorker, FrameReader, IngestionFailed, IngestionStats
from .reader import (
    DecoderSelectionReport,
    ExponentialBackoff,
    OpenCVFrameReader,
    OpenCVReaderConfig,
    ReaderClosed,
    ReaderStats,
    StreamUnavailable,
    VideoReaderError,
    VideoSourceOpenError,
    create_opencv_capture,
)
from .sources import VideoSource, parse_video_source

__all__ = [
    "BufferClosed",
    "BufferStats",
    "DecoderBuildReport",
    "DecoderSelectionReport",
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
    "create_opencv_capture",
    "parse_video_source",
    "probe_opencv_decoder_build",
]
