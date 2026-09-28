# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import operator

import cupy as cp
import numpy as np
import pytest
from numba_cuda_mlir import cuda, types

import cudf.core.udf.mlir_backend.strings_lowering  # noqa: F401  registers len
from cudf.core.udf.api import Masked
from cudf.core.udf.mlir_backend.strings_typing import (
    ManagedStrArrayWrapper,
    mlir_string,
    mlir_string_arg_handler,
)
from cudf.core.udf.utils import DEPRECATED_SM_REGEX

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

    ``None`` entries produce null rows: a null ``data`` pointer with
    ``nbytes == 0`` (mirroring how a null string row is marshalled), which the
    length loop must never dereference.

    Returns the ``ManagedStrArrayWrapper`` plus the backing device arrays, which
    must be kept alive for the duration of the kernel launch.
    """
    encoded = [None if s is None else s.encode("utf-8") for s in pystrings]
    chars = b"".join(e for e in encoded if e) or b"\x00"
    chars_dev = cp.asarray(np.frombuffer(chars, dtype=np.uint8))
    base = int(chars_dev.data.ptr)
    # struct layout {u64 meminfo, u64 data, i64 nbytes} == 3 x 8 bytes
    structs = np.zeros(len(pystrings) * 3, dtype=np.uint64)
    offset = 0
    for i, e in enumerate(encoded):
        if e is None:
            # null row: data==0 (null ptr), nbytes==0 (already zeroed)
            continue
        structs[i * 3 + 1] = base + offset  # data
        structs[i * 3 + 2] = len(e)  # nbytes
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
        ("h\u00e9llo", 5),  # é is 2 UTF-8 bytes, 1 char
        ("\U0001f600x", 2),  # emoji is 4 UTF-8 bytes, 1 char
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
    strings = ["", "a", "abc", "h\u00e9llo", "\U0001f600x"]
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


@pytest.mark.parametrize("valid", [True, False])
def test_masked_len_propagates_validity(valid):
    """``len(Masked(mlir_string))`` -> ``Masked(int64)`` carrying validity.

    The ``valid=False`` case uses a genuine null-row payload (null ``data``,
    ``nbytes == 0``) so the scan is exercised against the null representation,
    not a stand-in valid payload.
    """
    payload = "abc" if valid else None
    arr, _keep = _make_mlir_string_array([payload])
    out = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.CPointer(mlir_string),
            types.boolean[::1],
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, ov, s, sv):
        m = len(Masked(s[0], sv[0]))
        o[0] = m.value
        ov[0] = m.valid

    with MLIRNumbaCudaConfig():
        k[1, 1](out, out_valid, arr, cp.array([valid], dtype=np.bool_))
    cuda.synchronize()
    assert bool(out_valid.get()[0]) is valid
    # Payload of an invalid Masked is not an API guarantee; only assert it when
    # the result is valid.
    if valid:
        assert int(out.get()[0]) == 3


def test_masked_len_mixed_null_rows():
    """``len`` over a ``Masked(mlir_string)`` array with interleaved null rows.

    Null rows carry a null ``data`` pointer with ``nbytes == 0``; the kernel
    must not crash on them and must propagate ``valid=False``.
    """
    strings = ["abc", None, "h\u00e9llo", None, ""]
    valid = [s is not None for s in strings]
    arr, _keep = _make_mlir_string_array(strings)
    n = len(strings)
    out = cp.zeros(n, dtype=np.int64)
    out_valid = cp.zeros(n, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.CPointer(mlir_string),
            types.boolean[::1],
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, ov, s, sv):
        i = cuda.grid(1)
        if i < o.size:
            m = len(Masked(s[i], sv[i]))
            o[i] = m.value
            ov[i] = m.valid

    with MLIRNumbaCudaConfig():
        k[1, n](out, out_valid, arr, cp.array(valid, dtype=np.bool_))
    cuda.synchronize()
    got_valid = out_valid.get().tolist()
    got_value = out.get().tolist()
    assert got_valid == valid
    for s, v, value in zip(strings, got_valid, got_value, strict=True):
        if v:
            assert value == len(s)


_CMP_OPS = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]


@pytest.mark.parametrize("op", _CMP_OPS)
def test_string_comparison_arrays(op):
    """``str <cmp> str`` element-wise over two ``mlir_string`` arrays."""
    left = ["abc", "abc", "abd", "ab", "abcd", "", "h\u00e9llo"]
    right = ["abc", "abd", "abc", "abc", "abc", "", "h\u00e9llo"]
    la, _k1 = _make_mlir_string_array(left)
    ra, _k2 = _make_mlir_string_array(right)
    n = len(left)
    out = cp.zeros(n, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.boolean[::1],
            types.CPointer(mlir_string),
            types.CPointer(mlir_string),
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, a, b):
        i = cuda.grid(1)
        if i < o.size:
            o[i] = op(a[i], b[i])

    with MLIRNumbaCudaConfig():
        k[1, n](out, la, ra)
    cuda.synchronize()
    expected = [op(x, y) for x, y in zip(left, right, strict=True)]
    assert out.get().tolist() == expected


@pytest.mark.parametrize("op", _CMP_OPS)
def test_string_comparison_literal(op):
    """``str <cmp> "literal"`` (and the reflected form) over an array."""
    strings = ["foo", "fop", "fon", "fo", "foobar", ""]
    arr, _k = _make_mlir_string_array(strings)
    n = len(strings)
    out = cp.zeros(n, dtype=np.bool_)
    out_r = cp.zeros(n, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.boolean[::1],
            types.boolean[::1],
            types.CPointer(mlir_string),
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, orev, s):
        i = cuda.grid(1)
        if i < o.size:
            o[i] = op(s[i], "foo")
            orev[i] = op("foo", s[i])

    with MLIRNumbaCudaConfig():
        k[1, n](out, out_r, arr)
    cuda.synchronize()
    assert out.get().tolist() == [op(s, "foo") for s in strings]
    assert out_r.get().tolist() == [op("foo", s) for s in strings]


def test_string_affix_and_contains():
    """``startswith`` / ``endswith`` / ``in`` against a literal, over an array."""
    strings = ["abc", "abcd", "xabc", "ab", "", "cab"]
    arr, _k = _make_mlir_string_array(strings)
    n = len(strings)
    starts = cp.zeros(n, dtype=np.bool_)
    ends = cp.zeros(n, dtype=np.bool_)
    has = cp.zeros(n, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.boolean[::1],
            types.boolean[::1],
            types.boolean[::1],
            types.CPointer(mlir_string),
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(so, eo, ho, s):
        i = cuda.grid(1)
        if i < so.size:
            so[i] = s[i].startswith("ab")
            eo[i] = s[i].endswith("bc")
            ho[i] = "ab" in s[i]

    with MLIRNumbaCudaConfig():
        k[1, n](starts, ends, has, arr)
    cuda.synchronize()
    assert starts.get().tolist() == [s.startswith("ab") for s in strings]
    assert ends.get().tolist() == [s.endswith("bc") for s in strings]
    assert has.get().tolist() == ["ab" in s for s in strings]


def test_string_search():
    """``find`` / ``rfind`` / ``count`` against a literal, over an array."""
    strings = ["hello", "world", "abc", "lll", "", "l"]
    arr, _k = _make_mlir_string_array(strings)
    n = len(strings)
    finds = cp.zeros(n, dtype=np.int32)
    rfinds = cp.zeros(n, dtype=np.int32)
    counts = cp.zeros(n, dtype=np.int32)

    @cuda.jit(
        types.void(
            types.int32[::1],
            types.int32[::1],
            types.int32[::1],
            types.CPointer(mlir_string),
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(fo, ro, co, s):
        i = cuda.grid(1)
        if i < fo.size:
            fo[i] = s[i].find("l")
            ro[i] = s[i].rfind("l")
            co[i] = s[i].count("l")

    with MLIRNumbaCudaConfig():
        k[1, n](finds, rfinds, counts, arr)
    cuda.synchronize()
    assert finds.get().tolist() == [s.find("l") for s in strings]
    assert rfinds.get().tolist() == [s.rfind("l") for s in strings]
    assert counts.get().tolist() == [s.count("l") for s in strings]


def test_string_find_multibyte_char_position():
    """``find`` returns a character (not byte) position for multibyte input."""
    strings = ["h\u00e9llo", "\U0001f600x!"]  # é is 2 bytes; emoji is 4 bytes
    arr, _k = _make_mlir_string_array(strings)
    n = len(strings)
    out = cp.zeros(n, dtype=np.int32)

    @cuda.jit(
        types.void(types.int32[::1], types.CPointer(mlir_string)),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, s):
        i = cuda.grid(1)
        if i < o.size:
            o[i] = s[i].find("!")

    with MLIRNumbaCudaConfig():
        k[1, n](out, arr)
    cuda.synchronize()
    # "héllo".find("!") == -1 ; "\U0001f600x!".find("!") == 2 (char position)
    assert out.get().tolist() == [s.find("!") for s in strings]


_IS_PREDICATES = [
    "isalpha",
    "isalnum",
    "isdecimal",
    "isdigit",
    "isupper",
    "islower",
    "isspace",
    "isnumeric",
    "istitle",
]


def test_string_is_predicates():
    """Character-class predicates match Python's ``str`` methods for ASCII.

    A single kernel computes all nine predicates into separate columns (numba
    can't take the method name dynamically), each compared to the ``str`` oracle.
    """
    strings = [
        "abc",
        "ABC",
        "Abc",
        "abc123",
        "123",
        "  ",
        "a b",
        "",
        "Hello World",
        "hello world",
        "3.14",
        "ABC123",
    ]
    arr, _k = _make_mlir_string_array(strings)
    n = len(strings)
    outs = {name: cp.zeros(n, dtype=np.bool_) for name in _IS_PREDICATES}

    @cuda.jit(
        types.void(
            types.boolean[::1],  # isalpha
            types.boolean[::1],  # isalnum
            types.boolean[::1],  # isdecimal
            types.boolean[::1],  # isdigit
            types.boolean[::1],  # isupper
            types.boolean[::1],  # islower
            types.boolean[::1],  # isspace
            types.boolean[::1],  # isnumeric
            types.boolean[::1],  # istitle
            types.CPointer(mlir_string),
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o_alpha, o_alnum, o_dec, o_dig, o_up, o_low, o_sp, o_num, o_tit, s):
        i = cuda.grid(1)
        if i < o_alpha.size:
            e = s[i]
            o_alpha[i] = e.isalpha()
            o_alnum[i] = e.isalnum()
            o_dec[i] = e.isdecimal()
            o_dig[i] = e.isdigit()
            o_up[i] = e.isupper()
            o_low[i] = e.islower()
            o_sp[i] = e.isspace()
            o_num[i] = e.isnumeric()
            o_tit[i] = e.istitle()

    with MLIRNumbaCudaConfig():
        k[1, n](
            outs["isalpha"],
            outs["isalnum"],
            outs["isdecimal"],
            outs["isdigit"],
            outs["isupper"],
            outs["islower"],
            outs["isspace"],
            outs["isnumeric"],
            outs["istitle"],
            arr,
        )
    cuda.synchronize()
    for name in _IS_PREDICATES:
        expected = [getattr(s, name)() for s in strings]
        assert outs[name].get().tolist() == expected, name
