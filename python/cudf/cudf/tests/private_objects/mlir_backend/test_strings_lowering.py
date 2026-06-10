# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-level tests for the MLIR backend strings lowering module (PR 2).

Each test compiles a small ``cuda.jit`` kernel that exercises a single
shim entry point, launches it against a ``string_view`` array materialized
from a ``cudf.Series``, and verifies the result matches a Python reference.

Tests are organized by shim family:

* primitive-returning shims (``len``, ``is*``, ``find/rfind/count``,
  ``startswith/endswith``, ``contains``, scalar cmpops): full value match.
* managed-string-returning shims (``+``, ``upper/lower``, ``strip*``,
  ``replace``): length match; deeper content correctness lives in the
  integration suite (``tests/dataframe/methods/test_apply.py``).

The ``managed_udf_string`` operations require NRT linkage; that is
enabled per-kernel via ``cudf.core.udf.nrt_utils.nrt_enabled``.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext

import cupy as cp
import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import cuda  # noqa: E402
from numba_cuda_mlir.types import (  # noqa: E402
    CPointer,
    boolean,
    int32,
    void,
)

import cudf  # noqa: E402
from cudf._lib import strings_udf  # noqa: E402
from cudf.core.buffer import as_buffer  # noqa: E402
from cudf.core.udf.mlir_backend.strings_typing import (  # noqa: E402
    StrViewArrayWrapper,
    str_view_arg_handler,
    string_view,
)
from cudf.core.udf.nrt_utils import nrt_enabled  # noqa: E402
from cudf.core.udf.utils import (  # noqa: E402
    DEPRECATED_SM_REGEX,
    UDF_SHIM_FILE,
)
from cudf.utils._numba import _CUDFNumbaConfig  # noqa: E402

# Shim registrations are loaded by importing the module; we don't call
# anything from it directly here.
import cudf.core.udf.mlir_backend.strings_lowering  # noqa: E402, F401


# --- helpers ----------------------------------------------------------------


SV_PTR = CPointer(string_view)


def _str_view_array(strings) -> StrViewArrayWrapper:
    """Return a ``StrViewArrayWrapper`` over the device string-view array.

    Note: the returned ``string_view`` structs hold raw pointers into the
    underlying ``cudf.Series`` data. We attach the source Series to the
    wrapper as ``_owner`` so it isn't GC'd until the wrapper is - otherwise
    the kernel reads dangling pointers and the failures look like spooky
    cross-input pollution.
    """
    sr = cudf.Series(strings)
    wrapper = StrViewArrayWrapper(
        as_buffer(strings_udf.column_to_string_view_array(sr._column.plc_column))
    )
    wrapper._owner = sr
    return wrapper


def _jit(sig, *, nrt=False):
    """Decorator that compiles a kernel with the production link/handler stack."""
    ctx = nrt_enabled() if nrt else nullcontext()

    def decorator(fn):
        with ctx:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=DEPRECATED_SM_REGEX,
                    category=UserWarning,
                    module=r"^numba\.cuda(\.|$)",
                )
                return cuda.jit(
                    sig,
                    link=[UDF_SHIM_FILE],
                    extensions=[str_view_arg_handler],
                )(fn)

    return decorator


def _launch(kernel, *args):
    """Launch ``kernel[1,1](*args)`` and synchronize.

    Wrapped in ``_CUDFNumbaConfig`` to suppress
    ``NumbaPerformanceWarning`` (raised as error) that would otherwise
    fire on the 1x1 grids these tests use.
    """
    with _CUDFNumbaConfig():
        kernel[1, 1](*args)
    cuda.synchronize()


def _run_one(kernel, out, sv_arg):
    """Convenience wrapper for the common ``out, sv`` launch shape."""
    _launch(kernel, out, sv_arg)


# --- len --------------------------------------------------------------------


@pytest.mark.parametrize(
    "s,expected_chars",
    [
        ("", 0),
        ("a", 1),
        ("abc", 3),
        ("héllo", 5),  # `len` returns code-point count, matching Python.
        ("\U0001F600", 1),
    ],
)
def test_len(s, expected_chars):
    """``len(string_view)`` returns the character count as size_type (i32)."""

    @_jit(void(int32[::1], SV_PTR))
    def k(out, sv):
        out[0] = len(sv[0])

    sv = _str_view_array([s])
    out = cp.zeros(1, dtype=np.int32)
    _run_one(k, out, sv)
    assert int(out.get()[0]) == expected_chars


# --- is* predicates ---------------------------------------------------------


_IS_INPUTS = ["abc", "ABC", "AbC", "123", "abc123", " ", "", "Hello World"]


def _make_is_kernel(method_name):
    """Build an `out[0] = sv[0].<method>()` kernel."""
    src = f"""
def _k(out, sv):
    out[0] = sv[0].{method_name}()
"""
    ns = {"cuda": cuda}
    exec(src, ns)
    return _jit(void(boolean[::1], SV_PTR))(ns["_k"])


@pytest.mark.parametrize("s", _IS_INPUTS)
@pytest.mark.parametrize(
    "method",
    [
        "isalpha",
        "isalnum",
        "isdecimal",
        "isdigit",
        "isupper",
        "islower",
        "isspace",
        "isnumeric",
        "istitle",
    ],
)
def test_is_predicate(method, s):
    """Each ``string_view.is*()`` predicate matches Python's behavior."""
    k = _make_is_kernel(method)
    sv = _str_view_array([s])
    out = cp.zeros(1, dtype=np.bool_)
    _run_one(k, out, sv)
    assert bool(out.get()[0]) == getattr(s, method)()


# --- find / rfind / count ---------------------------------------------------


@pytest.mark.parametrize(
    "haystack,needle,expected",
    [
        ("hello world", "world", 6),
        ("hello world", "x", -1),
        ("aaa", "a", 0),
        ("", "x", -1),
        ("abc", "", 0),
    ],
)
def test_find(haystack, needle, expected):
    @_jit(void(int32[::1], SV_PTR, SV_PTR))
    def k(out, hay, ndl):
        out[0] = hay[0].find(ndl[0])

    hay = _str_view_array([haystack])
    ndl = _str_view_array([needle])
    out = cp.zeros(1, dtype=np.int32)
    _launch(k, out, hay, ndl)
    assert int(out.get()[0]) == expected


@pytest.mark.parametrize(
    "haystack,needle,expected",
    [
        ("hello world", "o", 7),
        ("aaaa", "a", 3),
        ("abc", "x", -1),
    ],
)
def test_rfind(haystack, needle, expected):
    @_jit(void(int32[::1], SV_PTR, SV_PTR))
    def k(out, hay, ndl):
        out[0] = hay[0].rfind(ndl[0])

    hay = _str_view_array([haystack])
    ndl = _str_view_array([needle])
    out = cp.zeros(1, dtype=np.int32)
    _launch(k, out, hay, ndl)
    assert int(out.get()[0]) == expected


@pytest.mark.parametrize(
    "haystack,needle,expected",
    [
        ("aaaa", "a", 4),
        ("aaaa", "aa", 2),
        ("hello world", "o", 2),
        ("abc", "x", 0),
    ],
)
def test_count(haystack, needle, expected):
    @_jit(void(int32[::1], SV_PTR, SV_PTR))
    def k(out, hay, ndl):
        out[0] = hay[0].count(ndl[0])

    hay = _str_view_array([haystack])
    ndl = _str_view_array([needle])
    out = cp.zeros(1, dtype=np.int32)
    _launch(k, out, hay, ndl)
    assert int(out.get()[0]) == expected


# --- startswith / endswith --------------------------------------------------


@pytest.mark.parametrize(
    "s,prefix,expected",
    [
        ("hello", "he", True),
        ("hello", "lo", False),
        ("hello", "", True),
        ("", "x", False),
    ],
)
def test_startswith(s, prefix, expected):
    @_jit(void(boolean[::1], SV_PTR, SV_PTR))
    def k(out, x, p):
        out[0] = x[0].startswith(p[0])

    x = _str_view_array([s])
    p = _str_view_array([prefix])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, x, p)
    assert bool(out.get()[0]) == expected


@pytest.mark.parametrize(
    "s,suffix,expected",
    [
        ("hello", "lo", True),
        ("hello", "he", False),
        ("hello", "", True),
        ("", "x", False),
    ],
)
def test_endswith(s, suffix, expected):
    @_jit(void(boolean[::1], SV_PTR, SV_PTR))
    def k(out, x, p):
        out[0] = x[0].endswith(p[0])

    x = _str_view_array([s])
    p = _str_view_array([suffix])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, x, p)
    assert bool(out.get()[0]) == expected


# --- comparison ops ---------------------------------------------------------
#
# Each cmpop is its own test function (no ``exec``) so the failure
# fingerprint is unambiguous.

_CMP_INPUTS = [("abc", "abc"), ("abc", "abd"), ("abc", "ab"), ("", ""), ("a", "")]


@pytest.mark.parametrize("lhs,rhs", _CMP_INPUTS)
def test_cmp_eq(lhs, rhs):
    @_jit(void(boolean[::1], SV_PTR, SV_PTR))
    def k(out, a, b):
        out[0] = a[0] == b[0]

    a = _str_view_array([lhs])
    b = _str_view_array([rhs])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, a, b)
    assert bool(out.get()[0]) == (lhs == rhs)


@pytest.mark.parametrize("lhs,rhs", _CMP_INPUTS)
def test_cmp_ne(lhs, rhs):
    @_jit(void(boolean[::1], SV_PTR, SV_PTR))
    def k(out, a, b):
        out[0] = a[0] != b[0]

    a = _str_view_array([lhs])
    b = _str_view_array([rhs])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, a, b)
    assert bool(out.get()[0]) == (lhs != rhs)


@pytest.mark.parametrize("lhs,rhs", _CMP_INPUTS)
def test_cmp_lt(lhs, rhs):
    @_jit(void(boolean[::1], SV_PTR, SV_PTR))
    def k(out, a, b):
        out[0] = a[0] < b[0]

    a = _str_view_array([lhs])
    b = _str_view_array([rhs])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, a, b)
    assert bool(out.get()[0]) == (lhs < rhs)


@pytest.mark.parametrize("lhs,rhs", _CMP_INPUTS)
def test_cmp_le(lhs, rhs):
    @_jit(void(boolean[::1], SV_PTR, SV_PTR))
    def k(out, a, b):
        out[0] = a[0] <= b[0]

    a = _str_view_array([lhs])
    b = _str_view_array([rhs])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, a, b)
    assert bool(out.get()[0]) == (lhs <= rhs)


@pytest.mark.parametrize("lhs,rhs", _CMP_INPUTS)
def test_cmp_gt(lhs, rhs):
    @_jit(void(boolean[::1], SV_PTR, SV_PTR))
    def k(out, a, b):
        out[0] = a[0] > b[0]

    a = _str_view_array([lhs])
    b = _str_view_array([rhs])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, a, b)
    assert bool(out.get()[0]) == (lhs > rhs)


@pytest.mark.parametrize("lhs,rhs", _CMP_INPUTS)
def test_cmp_ge(lhs, rhs):
    @_jit(void(boolean[::1], SV_PTR, SV_PTR))
    def k(out, a, b):
        out[0] = a[0] >= b[0]

    a = _str_view_array([lhs])
    b = _str_view_array([rhs])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, a, b)
    assert bool(out.get()[0]) == (lhs >= rhs)


# --- operator.contains ('in') -----------------------------------------------


@pytest.mark.parametrize(
    "needle,haystack,expected",
    [
        ("ell", "hello", True),
        ("xyz", "hello", False),
        ("", "hello", True),
        ("a", "", False),
    ],
)
def test_contains_sv_in_sv(needle, haystack, expected):
    @_jit(void(boolean[::1], SV_PTR, SV_PTR))
    def k(out, hay, ndl):
        out[0] = ndl[0] in hay[0]

    hay = _str_view_array([haystack])
    ndl = _str_view_array([needle])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, hay, ndl)
    assert bool(out.get()[0]) == expected


# --- string-returning ops: predicate-based correctness checks ---------------
#
# At this layer (no Masked typing yet), there is no ``len(managed_udf_string)``
# overload, so we can't compare lengths directly on the kernel side. We
# instead check predicates on the produced ``managed_udf_string`` (which
# delegate through ``lower_managed_to_sv_value`` to the same shims used for
# ``string_view``). Length-level checks are exercised end-to-end in the
# integration suite once Masked typing is in place.


@pytest.mark.parametrize(
    "lhs,rhs",
    [("abc", "def"), ("hello", " world"), ("a", "z")],
)
def test_concat_startswith(lhs, rhs):
    """``(a + b).startswith(a)`` is True when ``a`` is non-empty."""

    @_jit(void(boolean[::1], SV_PTR, SV_PTR), nrt=True)
    def k(out, a, b):
        out[0] = (a[0] + b[0]).startswith(a[0])

    a = _str_view_array([lhs])
    b = _str_view_array([rhs])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, a, b)
    assert bool(out.get()[0]) is True


@pytest.mark.parametrize(
    "lhs,rhs",
    [("abc", "def"), ("hello", " world"), ("a", "z")],
)
def test_concat_endswith(lhs, rhs):
    """``(a + b).endswith(b)`` is True when ``b`` is non-empty."""

    @_jit(void(boolean[::1], SV_PTR, SV_PTR), nrt=True)
    def k(out, a, b):
        out[0] = (a[0] + b[0]).endswith(b[0])

    a = _str_view_array([lhs])
    b = _str_view_array([rhs])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, a, b)
    assert bool(out.get()[0]) is True


@pytest.mark.parametrize("s", ["abc", "MiXeD", "hello"])
def test_upper_isupper(s):
    """``sv.upper().isupper()`` is True for inputs with at least one cased letter."""

    @_jit(void(boolean[::1], SV_PTR), nrt=True)
    def k(out, sv):
        out[0] = sv[0].upper().isupper()

    sv = _str_view_array([s])
    out = cp.zeros(1, dtype=np.bool_)
    _run_one(k, out, sv)
    assert bool(out.get()[0]) is True


@pytest.mark.parametrize("s", ["ABC", "MiXeD", "Hello"])
def test_lower_islower(s):
    """``sv.lower().islower()`` is True for inputs with at least one cased letter."""

    @_jit(void(boolean[::1], SV_PTR), nrt=True)
    def k(out, sv):
        out[0] = sv[0].lower().islower()

    sv = _str_view_array([s])
    out = cp.zeros(1, dtype=np.bool_)
    _run_one(k, out, sv)
    assert bool(out.get()[0]) is True


@pytest.mark.parametrize("s,chars", [("xxhelloxx", "x"), ("aaabbbccc", "a")])
def test_lstrip_drops_leading(s, chars):
    """After ``sv.lstrip(c)``, the result no longer starts with ``c``."""

    @_jit(void(boolean[::1], SV_PTR, SV_PTR), nrt=True)
    def k(out, sv, ch):
        out[0] = not sv[0].lstrip(ch[0]).startswith(ch[0])

    sv = _str_view_array([s])
    ch = _str_view_array([chars])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, sv, ch)
    assert bool(out.get()[0]) is True


@pytest.mark.parametrize("s,chars", [("xxhelloxx", "x"), ("aaabbbccc", "c")])
def test_rstrip_drops_trailing(s, chars):
    """After ``sv.rstrip(c)``, the result no longer ends with ``c``."""

    @_jit(void(boolean[::1], SV_PTR, SV_PTR), nrt=True)
    def k(out, sv, ch):
        out[0] = not sv[0].rstrip(ch[0]).endswith(ch[0])

    sv = _str_view_array([s])
    ch = _str_view_array([chars])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, sv, ch)
    assert bool(out.get()[0]) is True


@pytest.mark.parametrize("s,chars", [("xxhelloxx", "x"), ("  hello  ", " ")])
def test_strip_drops_both(s, chars):
    """After ``sv.strip(c)``, the result no longer starts or ends with ``c``."""

    @_jit(void(boolean[::1], SV_PTR, SV_PTR), nrt=True)
    def k(out, sv, ch):
        result = sv[0].strip(ch[0])
        out[0] = (not result.startswith(ch[0])) and (
            not result.endswith(ch[0])
        )

    sv = _str_view_array([s])
    ch = _str_view_array([chars])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, sv, ch)
    assert bool(out.get()[0]) is True


@pytest.mark.parametrize(
    "s,old,new",
    [
        ("hello", "l", "X"),
        ("hello", "ll", "rr"),
        ("aaa", "a", "bb"),
    ],
)
def test_replace_old_no_longer_present(s, old, new):
    """After ``sv.replace(old, new)``, ``old`` no longer appears in the result.

    This holds for the chosen cases since ``old`` is not a substring of
    ``new``. The check uses ``managed.find(old) == -1`` rather than
    ``old not in managed`` because ``operator.contains(managed, sv)`` is
    not registered (only ``contains(sv, sv)`` is) at this layer.

    Requires the numba-cuda-mlir extsi-widening fix
    (see https://github.com/NVIDIA/numba-cuda-mlir, branch
    ``fix/extsi-binary-ops``); without it, ``find()`` returns int32 ``-1``
    which is then zero-extended to int64 ``0xFFFFFFFF`` and compares
    incorrectly against the int64 literal ``-1``.
    """

    @_jit(void(boolean[::1], SV_PTR, SV_PTR, SV_PTR), nrt=True)
    def k(out, sv, o, n):
        out[0] = sv[0].replace(o[0], n[0]).find(o[0]) == -1

    sv = _str_view_array([s])
    o = _str_view_array([old])
    n = _str_view_array([new])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, sv, o, n)
    assert bool(out.get()[0]) is True


# --- managed_udf_string method delegation -----------------------------------


def test_managed_method_delegation_concat_then_upper_isupper():
    """``(a + b).upper().isupper()`` exercises managed_udf_string -> string_view
    delegation for the ``upper`` shim and the ``isupper`` predicate."""

    @_jit(void(boolean[::1], SV_PTR, SV_PTR), nrt=True)
    def k(out, a, b):
        out[0] = (a[0] + b[0]).upper().isupper()

    a = _str_view_array(["abc"])
    b = _str_view_array(["DeF"])
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, a, b)
    assert bool(out.get()[0]) is True
