# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing and lowering for ``mlir_string`` operations.

Currently only ``len`` (UTF-8 character count). Registered with
``numba_cuda_mlir`` at import time via :func:`_register`.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm
from numba_cuda_mlir.extending import lowering_registry, typing_registry
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    Signature,
)
from numba_cuda_mlir.typing import signature as nb_signature

from cudf.core.udf.mlir_backend import string_lowering_impl as _impl
from cudf.core.udf.mlir_backend.masked_lowering import (
    _extract_masked_value_valid,
    _pack_masked,
)
from cudf.core.udf.mlir_backend.masked_typing import MaskedType
from cudf.core.udf.mlir_backend.strings_typing import (
    MLIRStringType,
    mlir_string,
)

if TYPE_CHECKING:
    from numba_cuda_mlir.mlir_lowering import MLIRLower
    from numba_cuda_mlir.numba_cuda.core.ir import Var

# Comparison operators -> the arith.cmpi predicate applied to the memcmp-style
# result (-1/0/1) of a lexicographic byte comparison.
_CMP_PREDICATES = {
    operator.eq: arith.CmpIPredicate.eq,
    operator.ne: arith.CmpIPredicate.ne,
    operator.lt: arith.CmpIPredicate.slt,
    operator.le: arith.CmpIPredicate.sle,
    operator.gt: arith.CmpIPredicate.sgt,
    operator.ge: arith.CmpIPredicate.sge,
}


def _is_string_operand(ty: types.Type) -> bool:
    """Whether ``ty`` is an ``mlir_string`` or a compile-time string literal."""
    return isinstance(ty, (MLIRStringType, types.StringLiteral))


def _operand_data_nbytes(
    builder: MLIRLower, var: Var
) -> tuple[ir.Value, ir.Value]:
    """``(data_ptr, nbytes)`` for an ``mlir_string`` or string-literal operand.

    String literals are materialized as UTF-8 device globals so both operand
    kinds reduce to a raw ``(i8* data, i64 nbytes)`` pair.
    """
    nb_ty = builder.get_numba_type(var.name)
    if isinstance(nb_ty, types.StringLiteral):
        return _impl._materialize_utf8_literal(
            builder.mlir_gpu_module, nb_ty.literal_value
        )
    return _impl._ms_data_nbytes(builder.load_var(var))


class LenMLIRStringTemplate(AbstractTemplate):
    """``len`` over strings.

    ``len(mlir_string)`` -> ``int64`` (UTF-8 character count), and
    ``len(Masked(mlir_string))`` -> ``Masked(int64)`` (validity carried from the
    operand).
    """

    key = len

    def generic(
        self, args: tuple[types.Type, ...], kws: dict
    ) -> Signature | None:
        """Resolve ``len`` over a (masked) ``mlir_string``.

        Parameters
        ----------
        args : tuple of types.Type
            Positional argument types.
        kws : dict
            Keyword argument types (must be empty).

        Returns
        -------
        Signature or None
            ``int64`` for a bare ``mlir_string``, ``Masked(int64)`` for a
            ``Masked(mlir_string)``, else ``None``.
        """
        if len(args) != 1 or kws:
            return None
        arg = args[0]
        if isinstance(arg, MLIRStringType):
            return nb_signature(types.int64, mlir_string)
        if isinstance(arg, MaskedType) and isinstance(
            arg.value_type, MLIRStringType
        ):
            return nb_signature(MaskedType(types.int64), arg)
        return None


def _lower_len(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len(mlir_string)``: count UTF-8 characters, returned as ``int64``."""
    ms_val = builder.load_var(args[0])
    builder.store_var(target, _impl._lower_len(ms_val))


class StringComparisonTemplate(AbstractTemplate):
    """Typing for ``str <cmp> str`` -> ``boolean``.

    Registered for ``==``/``!=``/``<``/``<=``/``>``/``>=``; accepts any pair of
    ``mlir_string``/string-literal operands where at least one is an
    ``mlir_string`` (two literals are constant-folded by the compiler).
    """

    def generic(
        self, args: tuple[types.Type, ...], kws: dict
    ) -> Signature | None:
        """Resolve a string comparison signature.

        Parameters
        ----------
        args : tuple of types.Type
            Positional argument types.
        kws : dict
            Keyword argument types (must be empty).

        Returns
        -------
        Signature or None
            ``boolean(a, b)`` for a valid string operand pair, else ``None``.
        """
        if len(args) != 2 or kws:
            return None
        a, b = args
        if (
            _is_string_operand(a)
            and _is_string_operand(b)
            and (
                isinstance(a, MLIRStringType) or isinstance(b, MLIRStringType)
            )
        ):
            return nb_signature(types.boolean, a, b)
        return None


def _make_lower_comparison(
    predicate: arith.CmpIPredicate,
) -> object:
    """Build a lowering for a string comparison with the given ``cmpi`` predicate.

    Parameters
    ----------
    predicate : arith.CmpIPredicate
        Predicate applied to the ``memcmp``-style result against zero.

    Returns
    -------
    callable
        A lowering ``(builder, target, args, kwargs) -> None``.
    """

    def _lower(
        builder: MLIRLower, target: Var, args: list[Var], kwargs: list
    ) -> None:
        a_data, a_nbytes = _operand_data_nbytes(builder, args[0])
        b_data, b_nbytes = _operand_data_nbytes(builder, args[1])
        cmp = _impl._lower_bytes_compare(a_data, a_nbytes, b_data, b_nbytes)
        zero = arith.constant(ir.IntegerType.get_signless(32), 0)
        builder.store_var(target, arith.cmpi(predicate, cmp, zero))

    return _lower


def _lower_masked_len(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len(Masked(mlir_string))``: character count packed with the operand's
    validity bit (``Masked(int64)``).

    The payload is scanned unconditionally; this is safe because null rows carry
    ``nbytes == 0`` (the marshaller leaves a null ``data`` pointer with zero
    length), so the count loop never dereferences ``data`` for a null row. The
    resulting count is discarded anyway when ``m_valid`` is false.
    """
    m = builder.load_var(args[0])
    st = llvm.StructType(m.type)
    ms_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
    count_i64 = _impl._lower_len(ms_val)
    target_type = builder.get_numba_type(target.name)
    packed = _pack_masked(builder, target_type, count_i64, m_valid)
    builder.store_var(target, packed)


def _register() -> None:
    """Register ``mlir_string`` op typing and lowering with ``numba_cuda_mlir``."""
    typing_registry.register_global(len)(LenMLIRStringTemplate)
    lowering_registry.lower(len, mlir_string)(_lower_len)
    lowering_registry.lower(len, MaskedType)(_lower_masked_len)

    # Comparisons: str <cmp> str -> boolean, over mlir_string / string literals.
    for cmp_op, predicate in _CMP_PREDICATES.items():
        typing_registry.register_global(cmp_op)(StringComparisonTemplate)
        lower_cmp = _make_lower_comparison(predicate)
        for lhs_ty, rhs_ty in (
            (MLIRStringType, MLIRStringType),
            (MLIRStringType, types.StringLiteral),
            (types.StringLiteral, MLIRStringType),
        ):
            lowering_registry.lower(cmp_op, lhs_ty, rhs_ty)(lower_cmp)


_register()
