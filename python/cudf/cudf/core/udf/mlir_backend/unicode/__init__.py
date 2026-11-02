# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
Device-resident Unicode lookup tables for MLIR string operations.

Tables are extracted from libcudf's Unicode Character Database and embedded
as Python byte blobs.  On first use per CUDA context, the bytes are copied
to device memory and cached.

Public API
----------
get_character_flags_table_ptr() -> int
get_character_cases_table_ptr() -> int
get_special_case_mapping_table_ptr() -> int
"""

from __future__ import annotations

import ctypes
import threading

from cudf.core.udf.mlir_backend.unicode.tables_data import (
    CHARACTER_CASES,
    CHARACTER_FLAGS,
    SPECIAL_CASE_MAPPINGS,
)


_lock = threading.Lock()
_cache: dict[int, tuple[int, int, int]] = {}


def _upload_tables() -> tuple[int, int, int]:
    """Allocate device memory and copy the three tables.  Returns device pointers."""
    from numba_cuda_mlir.numba_cuda.cudadrv.driver import driver

    flags_size = len(CHARACTER_FLAGS)
    cases_size = len(CHARACTER_CASES)
    special_size = len(SPECIAL_CASE_MAPPINGS)

    d_flags = driver.cuMemAlloc(flags_size)
    d_cases = driver.cuMemAlloc(cases_size)
    d_special = driver.cuMemAlloc(special_size)

    flags_host = (ctypes.c_char * flags_size).from_buffer_copy(CHARACTER_FLAGS)
    cases_host = (ctypes.c_char * cases_size).from_buffer_copy(CHARACTER_CASES)
    special_host = (ctypes.c_char * special_size).from_buffer_copy(SPECIAL_CASE_MAPPINGS)

    driver.cuMemcpyHtoD(d_flags, flags_host, flags_size)
    driver.cuMemcpyHtoD(d_cases, cases_host, cases_size)
    driver.cuMemcpyHtoD(d_special, special_host, special_size)

    return int(d_flags), int(d_cases), int(d_special)


def _ensure_tables() -> tuple[int, int, int]:
    """Return (flags_ptr, cases_ptr, special_ptr) for the current CUDA context."""
    from numba_cuda_mlir.numba_cuda.cudadrv.driver import driver

    ctx = driver.cuCtxGetCurrent()
    ctx_handle = int(ctx)

    ptrs = _cache.get(ctx_handle)
    if ptrs is not None:
        return ptrs

    with _lock:
        ptrs = _cache.get(ctx_handle)
        if ptrs is not None:
            return ptrs
        ptrs = _upload_tables()
        _cache[ctx_handle] = ptrs
        return ptrs


def get_character_flags_table_ptr() -> int:
    """Return device pointer to the character flags table (65536 x uint8)."""
    return _ensure_tables()[0]


def get_character_cases_table_ptr() -> int:
    """Return device pointer to the character cases table (65536 x uint16)."""
    return _ensure_tables()[1]


def get_special_case_mapping_table_ptr() -> int:
    """Return device pointer to the special case mapping table (499 x 16 bytes)."""
    return _ensure_tables()[2]
