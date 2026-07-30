"""Export, TensorRT build, calibration, and high-performance runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from seahunter.data import WeatherCondition


class InferencePrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


@dataclass(frozen=True, slots=True)
class ONNXExportSpec:
    output_name: str
    opset: int
    static_shape: tuple[int, int, int, int]
    dynamic_batch: bool = False
    dynamic_spatial: bool = False

    def __post_init__(self) -> None:
        if not self.output_name.strip() or self.opset < 17:
            raise ValueError("ONNX output name must be set and opset must be at least 17")
        if len(self.static_shape) != 4 or any(value <= 0 for value in self.static_shape):
            raise ValueError("ONNX static_shape must contain four positive dimensions")
        if self.static_shape[1] != 3:
            raise ValueError("ONNX detector input must have three channels")

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        axes: dict[int, str] = {}
        if self.dynamic_batch:
            axes[0] = "batch"
        if self.dynamic_spatial:
            axes.update({2: "height", 3: "width"})
        return {"images": axes, "output0": {0: "batch"}} if axes else {}


@dataclass(frozen=True, slots=True)
class INT8CalibrationSet:
    data_version: str
    sample_count: int
    real_only: bool
    weather_samples: tuple[tuple[WeatherCondition, int], ...]

    def __post_init__(self) -> None:
        if not self.data_version.strip() or self.sample_count <= 0:
            raise ValueError("INT8 calibration data version and sample_count are required")
        if len({weather for weather, _ in self.weather_samples}) != len(self.weather_samples):
            raise ValueError("INT8 weather entries must be unique")
        if any(count < 0 for _, count in self.weather_samples):
            raise ValueError("INT8 weather counts must be non-negative")

    def validate_for_production(self, *, minimum_per_weather: int) -> None:
        if minimum_per_weather <= 0:
            raise ValueError("minimum_per_weather must be positive")
        if not self.real_only:
            raise ValueError("production INT8 calibration must use real maritime samples only")
        counts = dict(self.weather_samples)
        missing = [weather.value for weather in WeatherCondition if counts.get(weather, 0) < minimum_per_weather]
        if missing:
            raise ValueError(f"INT8 calibration weather coverage is insufficient: {', '.join(missing)}")
        if sum(counts.values()) != self.sample_count:
            raise ValueError("INT8 calibration sample_count does not match weather counts")


@dataclass(frozen=True, slots=True)
class TensorRTBuildSpec:
    onnx_path: str
    engine_path: str
    precision: InferencePrecision
    minimum_shape: tuple[int, int, int, int]
    optimum_shape: tuple[int, int, int, int]
    maximum_shape: tuple[int, int, int, int]
    workspace_mb: int = 2048
    calibration: INT8CalibrationSet | None = None

    def __post_init__(self) -> None:
        if not self.onnx_path.strip() or not self.engine_path.strip() or self.workspace_mb <= 0:
            raise ValueError("TensorRT paths and workspace are invalid")
        for shape in (self.minimum_shape, self.optimum_shape, self.maximum_shape):
            if len(shape) != 4 or any(value <= 0 for value in shape):
                raise ValueError("TensorRT profile shapes must contain four positive dimensions")
        for lower, optimum, upper in zip(self.minimum_shape, self.optimum_shape, self.maximum_shape, strict=True):
            if not lower <= optimum <= upper:
                raise ValueError("TensorRT profile shapes must satisfy min <= opt <= max")
        if self.precision is InferencePrecision.INT8 and self.calibration is None:
            raise ValueError("TensorRT INT8 builds require a calibration set")
        if self.precision is not InferencePrecision.INT8 and self.calibration is not None:
            raise ValueError("calibration metadata is only valid for TensorRT INT8 builds")

    def trtexec_arguments(self) -> tuple[str, ...]:
        arguments = (
            "trtexec",
            f"--onnx={self.onnx_path}",
            f"--saveEngine={self.engine_path}",
            f"--memPoolSize=workspace:{self.workspace_mb}",
            f"--minShapes=images:{_shape(self.minimum_shape)}",
            f"--optShapes=images:{_shape(self.optimum_shape)}",
            f"--maxShapes=images:{_shape(self.maximum_shape)}",
        )
        if self.precision is InferencePrecision.FP16:
            return (*arguments, "--fp16")
        if self.precision is InferencePrecision.INT8:
            return (*arguments, "--int8")
        return arguments


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    nvdec_verified: bool
    cuda_streams: bool
    preallocated_buffers: bool
    tensorrt_available: bool


@dataclass(frozen=True, slots=True)
class HighPerformanceRuntimeSpec:
    decoder: str = "nvdec"
    inference_backend: str = "tensorrt"
    cuda_stream_count: int = 2
    preallocated_frame_buffers: int = 8
    implementation: str = "deepstream"

    def __post_init__(self) -> None:
        if not self.decoder.strip() or not self.inference_backend.strip() or not self.implementation.strip():
            raise ValueError("runtime backend names must not be empty")
        if self.cuda_stream_count <= 0 or self.preallocated_frame_buffers <= 0:
            raise ValueError("runtime CUDA streams and frame buffers must be positive")

    def missing_capabilities(self, capabilities: RuntimeCapabilities) -> tuple[str, ...]:
        missing: list[str] = []
        if self.decoder == "nvdec" and not capabilities.nvdec_verified:
            missing.append("nvdec")
        if self.cuda_stream_count > 0 and not capabilities.cuda_streams:
            missing.append("cuda_streams")
        if self.preallocated_frame_buffers > 0 and not capabilities.preallocated_buffers:
            missing.append("preallocated_buffers")
        if self.inference_backend == "tensorrt" and not capabilities.tensorrt_available:
            missing.append("tensorrt")
        return tuple(missing)


@dataclass(frozen=True, slots=True)
class BackendConsistencyResult:
    backend: str
    output_shape_matches: bool
    maximum_absolute_error: float
    mean_absolute_error: float
    maximum_threshold: float
    mean_threshold: float

    @property
    def passed(self) -> bool:
        return (
            self.output_shape_matches
            and self.maximum_absolute_error <= self.maximum_threshold
            and self.mean_absolute_error <= self.mean_threshold
        )


def _shape(shape: tuple[int, int, int, int]) -> str:
    return "x".join(str(value) for value in shape)
