# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import cupy as cp
import pytest
from numba_cuda_mlir import cuda, types

import cudf.core.udf.mlir_backend.strings_lowering  # noqa: F401  registers len
from cudf.core.udf.utils import DEPRECATED_SM_REGEX
from cudf.core.udf.mlir_backend.strings_typing import (
    ManagedStrArrayWrapper,
    mlir_string,
    mlir_string_arg_handler,
)

from .utils import MLIRNumbaCudaConfig

pytestmark = [
    pytest.mark.filterwarnings(f"ignore:{DEPRECATED_SM_REGEX}:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:Grid size:"
        "numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


class _DeviceBuf:
    """Minimal ``.ptr``-exposing wrapper over a cupy array (test marshaller)."""

    def __init__(self, arr):
        self._arr = arr

    @property
    def ptr(self):
        return int(self._arr.data.ptr)


def _make_mlir_string_array(pystrings):
    """Build a device ``mlir_string`` array (borrowed data, null meminfo).

    Returns the ``ManagedStrArrayWrapper`` plus the backing device arrays, which
    must be kept alive for the duration of the kernel launch.
    """
    encoded = [s.encode("utf-8") for s in pystrings]
    chars = b"".join(encoded) or b"\x00"
    chars_dev = cp.asarray(np.frombuffer(chars, dtype=np.uint8))
    base = int(chars_dev.data.ptr)
    # struct layout {u64 meminfo, u64 data, i64 nbytes} == 3 x 8 bytes
    structs = np.zeros(len(pystrings) * 3, dtype=np.uint64)
    offset = 0
    for i, e in enumerate(encoded):
        structs[i * 3 + 1] = base + offset  # data
        structs[i * 3 + 2] = len(e)         # nbytes
        offset += len(e)
    structs_dev = cp.asarray(structs)
    wrapper = ManagedStrArrayWrapper(_DeviceBuf(structs_dev))
    return wrapper, (chars_dev, structs_dev)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", 0),
        ("a", 1),
        ("abc", 3),
        ("h\u00e9llo", 5),      # é is 2 UTF-8 bytes, 1 char
        ("\U0001F600x", 2),     # emoji is 4 UTF-8 bytes, 1 char
        ("na\u00efve", 5),
    ],
)
def test_len_counts_characters(value, expected):
    """``len(mlir_string)`` returns the UTF-8 character count, not byte count."""
    arr, _keep = _make_mlir_string_array([value])
    out = cp.zeros(1, dtype=np.int64)

    @cuda.jit(
        types.void(types.int64[::1], types.CPointer(mlir_string)),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, s):
        o[0] = len(s[0])

    with MLIRNumbaCudaConfig():
        k[1, 1](out, arr)
    cuda.synchronize()
    assert int(out.get()[0]) == expected


def test_len_over_array():
    """``len`` over a multi-element ``mlir_string`` array, one thread per row."""
    strings = ["", "a", "abc", "h\u00e9llo", "\U0001F600x"]
    arr, _keep = _make_mlir_string_array(strings)
    out = cp.zeros(len(strings), dtype=np.int64)

    @cuda.jit(
        types.void(types.int64[::1], types.CPointer(mlir_string)),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, s):
        i = cuda.grid(1)
        if i < o.size:
            o[i] = len(s[i])

    with MLIRNumbaCudaConfig():
        k[1, len(strings)](out, arr)
    cuda.synchronize()
    assert out.get().tolist() == [len(s) for s in strings]
