"""Fixed spatial/frequency anti-alias downsampling reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

Matrix = tuple[tuple[float, ...], ...]


class DownsampleVariant(str, Enum):
    STANDARD_STRIDE = "standard_stride"
    SPDCONV = "spdconv"
    JOINT_ANTIALIAS = "joint_antialias"


@dataclass(frozen=True, slots=True)
class DownsampleOperatorManifest:
    variant: DownsampleVariant
    onnx_primitives: tuple[str, ...]
    uses_fft: bool
    uses_complex: bool
    fixed_kernel: bool
    tensorrt_plugin_required: bool


def joint_spatial_frequency_downsample(
    image: Matrix, *, stride: int = 2, high_frequency_weight: float = 0.15
) -> Matrix:
    """Fuse fixed low-pass and Laplacian energy using export-friendly real operators."""

    height, width = _validate_matrix(image)
    if stride <= 0 or not isfinite(high_frequency_weight) or not 0.0 <= high_frequency_weight <= 1.0:
        raise ValueError("downsample stride and high-frequency weight are invalid")
    low_kernel = ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))
    high_kernel = ((0.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 0.0))
    low = _convolve(image, low_kernel, divisor=16.0)
    high = _convolve(image, high_kernel, divisor=1.0)
    output: list[tuple[float, ...]] = []
    for row in range(0, height, stride):
        output.append(
            tuple(
                low[row][column] + high_frequency_weight * abs(high[row][column]) for column in range(0, width, stride)
            )
        )
    return tuple(output)


def operator_manifest(variant: DownsampleVariant) -> DownsampleOperatorManifest:
    if variant is DownsampleVariant.JOINT_ANTIALIAS:
        return DownsampleOperatorManifest(variant, ("Conv", "Abs", "Mul", "Add"), False, False, True, False)
    if variant is DownsampleVariant.SPDCONV:
        return DownsampleOperatorManifest(variant, ("Reshape", "Transpose", "Conv"), False, False, False, False)
    return DownsampleOperatorManifest(variant, ("Conv",), False, False, False, False)


def _validate_matrix(image: Matrix) -> tuple[int, int]:
    if not image or not image[0]:
        raise ValueError("downsample image must be non-empty")
    width = len(image[0])
    if any(len(row) != width for row in image):
        raise ValueError("downsample image rows must have equal length")
    if any(not isfinite(value) for row in image for value in row):
        raise ValueError("downsample image values must be finite")
    return len(image), width


def _convolve(image: Matrix, kernel: Matrix, *, divisor: float) -> Matrix:
    height = len(image)
    width = len(image[0])
    rows: list[tuple[float, ...]] = []
    for row in range(height):
        values: list[float] = []
        for column in range(width):
            total = 0.0
            for kernel_row in range(3):
                source_row = min(height - 1, max(0, row + kernel_row - 1))
                for kernel_column in range(3):
                    source_column = min(width - 1, max(0, column + kernel_column - 1))
                    total += image[source_row][source_column] * kernel[kernel_row][kernel_column]
            values.append(total / divisor)
        rows.append(tuple(values))
    return tuple(rows)
