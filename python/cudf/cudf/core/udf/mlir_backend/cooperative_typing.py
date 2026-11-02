"""Typing for CooperativeArrayType: attributes, binary ops, reductions.

cusimt owns the CooperativeArrayType and all its operations.
cudf maps GroupType columns to CooperativeArray views.
"""

from __future__ import annotations

import operator

from cudf.core.udf.mlir_backend.cooperative_types import CooperativeArrayType
from numba_cuda_mlir.extending import typing_registry as registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.typing import signature as nb_signature
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
)


class CooperativeArray:
    """Sentinel for constructing CooperativeArrayType from array + size."""
    pass


# ---------------------------------------------------------------------------
# Constructor: CooperativeArray(array, size) -> CooperativeArrayType(dtype)
# ---------------------------------------------------------------------------

@registry.register_global(CooperativeArray)
class CooperativeArrayConstructorTemplate(AbstractTemplate):
    key = CooperativeArray

    def generic(self, args, kws):
        if len(args) == 2 and not kws:
            arr, size = args
            if isinstance(arr, types.Array) and isinstance(size, types.Integer):
                return nb_signature(
                    CooperativeArrayType(arr.dtype), arr, size
                )


# ---------------------------------------------------------------------------
# Attribute typing: .size, .sum()
# ---------------------------------------------------------------------------

class CoopSumTemplate(AbstractTemplate):
    key = "CooperativeArray.sum"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(self.this.dtype, recvr=self.this)


class CoopMeanTemplate(AbstractTemplate):
    key = "CooperativeArray.mean"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(types.float64, recvr=self.this)


class CoopMinTemplate(AbstractTemplate):
    key = "CooperativeArray.min"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(self.this.dtype, recvr=self.this)


class CoopMaxTemplate(AbstractTemplate):
    key = "CooperativeArray.max"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(self.this.dtype, recvr=self.this)


class CoopExpTemplate(AbstractTemplate):
    key = "CooperativeArray.exp"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(self.this, recvr=self.this)


class CoopStdTemplate(AbstractTemplate):
    key = "CooperativeArray.std"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(types.float64, recvr=self.this)


class CoopVarTemplate(AbstractTemplate):
    key = "CooperativeArray.var"

    def generic(self, args, kws):
        if len(args) == 0 and not kws:
            return nb_signature(types.float64, recvr=self.this)



_SUPPORTED_DTYPES = (types.float64, types.float32, types.int64, types.int32)


def _make_attrs_class(dtype):
    ca_ty = CooperativeArrayType(dtype)

    @registry.register_attr
    class _Attrs(AttributeTemplate):
        key = ca_ty

        def resolve_size(self, mod):
            return types.int64

        def resolve_sum(self, mod):
            return types.BoundFunction(CoopSumTemplate, ca_ty)

        def resolve_mean(self, mod):
            return types.BoundFunction(CoopMeanTemplate, ca_ty)

        def resolve_min(self, mod):
            return types.BoundFunction(CoopMinTemplate, ca_ty)

        def resolve_max(self, mod):
            return types.BoundFunction(CoopMaxTemplate, ca_ty)

        def resolve_exp(self, mod):
            return types.BoundFunction(CoopExpTemplate, ca_ty)

        def resolve_std(self, mod):
            return types.BoundFunction(CoopStdTemplate, ca_ty)

        def resolve_var(self, mod):
            return types.BoundFunction(CoopVarTemplate, ca_ty)

    _Attrs.__name__ = f"CooperativeArray{dtype}Attrs"
    _Attrs.__qualname__ = _Attrs.__name__
    return _Attrs


for _dt in _SUPPORTED_DTYPES:
    _make_attrs_class(_dt)


# ---------------------------------------------------------------------------
# Binary ops: +, -, * between two CooperativeArrays of matching dtype
# ---------------------------------------------------------------------------

_BINOPS = (operator.add, operator.sub, operator.mul, operator.truediv)

_ca_f64 = CooperativeArrayType(types.float64)


def _make_binop_template(binop):
    class BinOpTemplate(AbstractTemplate):
        key = binop

        def generic(self, args, kws):
            if len(args) != 2 or kws:
                return None
            a, b = args
            # array op array
            if isinstance(a, CooperativeArrayType) and isinstance(b, CooperativeArrayType):
                if a.dtype == b.dtype:
                    return nb_signature(a, a, b)
            # array op scalar
            if isinstance(a, CooperativeArrayType) and isinstance(b, (types.Integer, types.Float)):
                return nb_signature(a, a, b)
            # scalar op array
            if isinstance(a, (types.Integer, types.Float)) and isinstance(b, CooperativeArrayType):
                return nb_signature(b, a, b)

    return BinOpTemplate


for _op in _BINOPS:
    registry.register_global(_op)(_make_binop_template(_op))


# ---------------------------------------------------------------------------
# Factory stub: cooperative_array_from_ptr(data_ptr, size) -> CooperativeArray
# ---------------------------------------------------------------------------

def _cooperative_array_from_ptr_stub(data_ptr, size):
    """Sentinel: construct a non-owning CooperativeArray from raw pointer and element count."""
    raise NotImplementedError("device-only")


class CoopFromPtrTemplate(AbstractTemplate):
    key = _cooperative_array_from_ptr_stub

    def generic(self, args, kws):
        if len(args) == 2 and not kws:
            if isinstance(args[0], types.Integer) and isinstance(args[1], types.Integer):
                return nb_signature(_ca_f64, types.int64, types.int64)


registry.register_global(
    _cooperative_array_from_ptr_stub,
    types.Function(CoopFromPtrTemplate),
)

cooperative_array_from_ptr = _cooperative_array_from_ptr_stub
