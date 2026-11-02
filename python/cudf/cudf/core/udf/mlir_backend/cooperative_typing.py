"""Typing for CooperativeArrayType: reduction attributes.

Grouped columns are modelled as ``CooperativeArrayType`` views. Only
scalar-returning reductions (min/max/sum/mean/var/std/idxmin/idxmax/corr/
count/size) are supported inside a JIT ``GroupBy.apply`` UDF; operators
applied directly to a grouped column are rejected (see below).
"""

from __future__ import annotations

from cudf.core.udf._ops import arith_ops, bitwise_ops, comparison_ops, unary_ops
from cudf.core.udf.mlir_backend.cooperative_types import CooperativeArrayType
from cudf.core.udf.utils import UDFError
from numba_cuda_mlir.extending import typing_registry as registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.typing import signature as nb_signature
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
)

_UDF_DOC_URL = "https://docs.rapids.ai/api/cudf/stable/cudf/guide-to-udfs/"


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

        def resolve_std(self, mod):
            return types.BoundFunction(CoopStdTemplate, ca_ty)

        def resolve_var(self, mod):
            return types.BoundFunction(CoopVarTemplate, ca_ty)

    _Attrs.__name__ = f"CooperativeArray{dtype}Attrs"
    _Attrs.__qualname__ = _Attrs.__name__
    return _Attrs


for _dt in _SUPPORTED_DTYPES:
    _make_attrs_class(_dt)


_ca_f64 = CooperativeArrayType(types.float64)


# ---------------------------------------------------------------------------
# Unsupported operators on grouped columns.
#
# A JIT GroupBy.apply UDF may only produce scalar reduction results. Applying
# arithmetic, comparison, bitwise, or unary operators directly to a grouped
# column (e.g. ``group['x'] + group['y']`` or ``-group['x']``) has no scalar
# meaning and is rejected with a UDFError, matching the legacy backend. The
# array-returning operators are intentionally not implemented.
# ---------------------------------------------------------------------------

def _make_unsupported_op_template(op):
    class UnsupportedCoopOpTemplate(AbstractTemplate):
        key = op

        def generic(self, args, kws):
            # Only intercept when a grouped column participates; scalar
            # operations (e.g. on reduction results) type normally.
            if not any(isinstance(a, CooperativeArrayType) for a in args):
                return None
            fname = getattr(op, "__name__", str(op))
            sr_err = ", ".join("Series" for _ in args)
            raise UDFError(
                f"{fname}({sr_err}) is not supported by JIT GroupBy apply. "
                f"Supported features are listed at: {_UDF_DOC_URL}"
            )

    return UnsupportedCoopOpTemplate


for _op in (*arith_ops, *comparison_ops, *bitwise_ops, *unary_ops):
    registry.register_global(_op)(_make_unsupported_op_template(_op))


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
