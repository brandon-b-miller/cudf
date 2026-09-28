# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing and lowering for read-only ``mlir_string`` operations.

Covers the non-NRT, non-string-producing ops: ``len``, comparisons
(``==``/``!=``/``<``/``<=``/``>``/``>=``), ``in`` (``operator.contains``), and the
methods ``find``/``rfind``/``count`` (-> ``int32``) and
``startswith``/``endswith``/``isalpha``/... (-> ``boolean``). All lowerings are
pure MLIR (see :mod:`cudf.core.udf.mlir_backend.string_lowering_impl`); string
literals are handled without allocation via an internal view. Registered with
``numba_cuda_mlir`` at import time via :func:`_register`.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm
from numba_cuda_mlir.extending import lowering_registry, typing_registry
from numba_cuda_mlir.lowering_utilities import DeferredMethodCall
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    Signature,
)
from numba_cuda_mlir.typing import signature as nb_signature

from cudf._lib.strings_udf import get_character_flags_table_ptr
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

# libcudf size_type; the width of len / find / rfind / count results.
size_type = types.int32

# comparison operator -> the arith.cmpi predicate applied to the signed
# ``_impl._lower_compare`` result against zero.
_CMPOP_PREDICATES = {
    operator.eq: arith.CmpIPredicate.eq,
    operator.ne: arith.CmpIPredicate.ne,
    operator.lt: arith.CmpIPredicate.slt,
    operator.le: arith.CmpIPredicate.sle,
    operator.gt: arith.CmpIPredicate.sgt,
    operator.ge: arith.CmpIPredicate.sge,
}

# character-class predicate method name -> its pure-MLIR impl.
_IS_METHODS = {
    "isalpha": _impl._lower_isalpha,
    "isalnum": _impl._lower_isalnum,
    "isdecimal": _impl._lower_isdecimal,
    "isdigit": _impl._lower_isdigit,
    "isupper": _impl._lower_isupper,
    "islower": _impl._lower_islower,
    "isspace": _impl._lower_isspace,
    "isnumeric": _impl._lower_isnumeric,
    "istitle": _impl._lower_istitle,
}

# method name -> pure-MLIR impl for the two-string ops.
_INT_METHODS = {
    "find": _impl._lower_find,
    "rfind": _impl._lower_rfind,
    "count": _impl._lower_count,
}
_BOOL_METHODS = {
    "startswith": _impl._lower_startswith,
    "endswith": _impl._lower_endswith,
}


# --- typing -----------------------------------------------------------------
def _is_string_operand(ty: types.Type) -> bool:
    """Whether ``ty`` is an ``mlir_string`` or a compile-time string literal."""
    return isinstance(ty, (MLIRStringType, types.StringLiteral))


class LenMLIRStringTemplate(AbstractTemplate):
    """``len`` over strings.

    ``len(mlir_string)`` -> ``int32`` (UTF-8 character count), and
    ``len(Masked(mlir_string))`` -> ``Masked(int32)`` (validity carried through).
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
            ``int32`` for ``mlir_string``, ``Masked(int32)`` for a
            ``Masked(mlir_string)``, else ``None``.
        """
        if len(args) != 1 or kws:
            return None
        arg = args[0]
        if isinstance(arg, MLIRStringType):
            return nb_signature(size_type, mlir_string)
        if isinstance(arg, MaskedType) and isinstance(
            arg.value_type, MLIRStringType
        ):
            return nb_signature(MaskedType(size_type), arg)
        return None


def _make_cmpop_template(cmpop: object) -> type[AbstractTemplate]:
    """Build the typing template for a string comparison operator.

    Parameters
    ----------
    cmpop : object
        The comparison operator (e.g. ``operator.eq``).

    Returns
    -------
    type
        An ``AbstractTemplate`` subclass resolving the operator to ``boolean``
        for any ``mlir_string``/string-literal pair with at least one
        ``mlir_string``.
    """

    class _CmpOpTemplate(AbstractTemplate):
        key = cmpop

        def generic(self, args, kws):
            """Resolve ``a <cmp> b`` -> ``boolean`` for string operands."""
            if len(args) != 2 or kws:
                return None
            a, b = args
            if (
                _is_string_operand(a)
                and _is_string_operand(b)
                and (
                    isinstance(a, MLIRStringType)
                    or isinstance(b, MLIRStringType)
                )
            ):
                return nb_signature(types.boolean, a, b)
            return None

    return _CmpOpTemplate


class ContainsMLIRStringTemplate(AbstractTemplate):
    """``substr in string`` (``operator.contains``) -> ``boolean``."""

    key = operator.contains

    def generic(
        self, args: tuple[types.Type, ...], kws: dict
    ) -> Signature | None:
        """Resolve ``operator.contains`` for string operands.

        Parameters
        ----------
        args : tuple of types.Type
            ``(container, item)`` types.
        kws : dict
            Keyword argument types (must be empty).

        Returns
        -------
        Signature or None
            ``boolean(container, item)`` for string operands, else ``None``.
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


def _make_method_attr(attrname: str, retty: types.Type) -> object:
    """Build a ``resolve_<method>`` returning a bound one-arg string method.

    Parameters
    ----------
    attrname : str
        Method name (e.g. ``"find"``).
    retty : types.Type
        The method's return type.

    Returns
    -------
    callable
        A ``resolve_`` function for :class:`MLIRStringAttrs`.
    """

    class _MethodTemplate(AbstractTemplate):
        key = f"MLIRString.{attrname}"

        def generic(self, args, kws):
            """Resolve ``s.<method>(other)`` for a string ``other``."""
            if len(args) == 1 and not kws:
                return nb_signature(retty, args[0], recvr=self.this)
            return None

    def resolve(self, mod):
        return types.BoundFunction(_MethodTemplate, mlir_string)

    return resolve


def _make_is_attr(attrname: str) -> object:
    """Build a ``resolve_<isX>`` returning a bound no-arg predicate method.

    Parameters
    ----------
    attrname : str
        Predicate name (e.g. ``"isalpha"``).

    Returns
    -------
    callable
        A ``resolve_`` function for :class:`MLIRStringAttrs`.
    """

    class _IsTemplate(AbstractTemplate):
        key = f"MLIRString.{attrname}"

        def generic(self, args, kws):
            """Resolve ``s.<isX>()`` -> ``boolean``."""
            return nb_signature(types.boolean, recvr=self.this)

    def resolve(self, mod):
        return types.BoundFunction(_IsTemplate, mlir_string)

    return resolve


@typing_registry.register_attr
class MLIRStringAttrs(AttributeTemplate):
    """Attribute typing for ``mlir_string`` methods."""

    key = mlir_string


for _name in _IS_METHODS:
    setattr(MLIRStringAttrs, f"resolve_{_name}", _make_is_attr(_name))
for _name in _INT_METHODS:
    setattr(
        MLIRStringAttrs,
        f"resolve_{_name}",
        _make_method_attr(_name, size_type),
    )
for _name in _BOOL_METHODS:
    setattr(
        MLIRStringAttrs,
        f"resolve_{_name}",
        _make_method_attr(_name, types.boolean),
    )


# --- lowering helpers -------------------------------------------------------
def _flags_table_ptr_const() -> ir.Value:
    """MLIR pointer constant for the libcudf character-flags device table."""
    addr = int(get_character_flags_table_ptr())
    if addr >= (1 << 63):
        addr -= 1 << 64
    i64 = ir.IntegerType.get_signless(64)
    return llvm.inttoptr(llvm.PointerType.get(), arith.constant(i64, addr))


def _literal_to_view(builder: MLIRLower, literal_var: Var) -> ir.Value:
    """Materialize a ``StringLiteral`` var as a view ``{ptr, i32 nbytes, i32 len}``."""
    literal_ty = builder.get_numba_type(literal_var.name)
    literal_value = literal_ty.literal_value
    data_struct = builder.load_var(literal_var)
    ptr_ty = llvm.PointerType.get()
    i32 = ir.IntegerType.get_signless(32)
    view_ty = llvm.StructType.get_literal([ptr_ty, i32, i32])
    nbytes = len(literal_value.encode("utf-8"))
    length = len(literal_value)
    val = llvm.UndefOp(view_ty)
    val = llvm.insertvalue(
        container=val,
        value=llvm.extractvalue(ptr_ty, data_struct, [0]),
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    val = llvm.insertvalue(
        container=val,
        value=arith.constant(i32, nbytes),
        position=ir.DenseI64ArrayAttr.get([1]),
    )
    return llvm.insertvalue(
        container=val,
        value=arith.constant(i32, length),
        position=ir.DenseI64ArrayAttr.get([2]),
    )


def _any_var_to_view(builder: MLIRLower, var: Var) -> ir.Value:
    """View SSA value for an ``mlir_string`` or ``StringLiteral`` operand var."""
    nb_ty = builder.get_numba_type(var.name)
    if isinstance(nb_ty, types.StringLiteral):
        return _literal_to_view(builder, var)
    return _impl._mlir_string_to_view(builder.load_var(var))


# --- lowering impls ---------------------------------------------------------
def _lower_len_op(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len(mlir_string)`` -> ``int32`` character count."""
    view = _impl._mlir_string_to_view(builder.load_var(args[0]))
    builder.store_var(target, _impl._lower_len(view))


def _lower_masked_len(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len(Masked(mlir_string))`` -> ``Masked(int32)`` carrying validity.

    The payload is scanned unconditionally; this is safe because null rows carry
    ``nbytes == 0`` (a null ``data`` pointer with zero length), so the count loop
    never dereferences ``data``. The count is discarded when the operand is null.
    """
    m = builder.load_var(args[0])
    st = llvm.StructType(m.type)
    ms_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
    count = _impl._lower_len(_impl._mlir_string_to_view(ms_val))
    target_type = builder.get_numba_type(target.name)
    builder.store_var(
        target, _pack_masked(builder, target_type, count, m_valid)
    )


def _make_lower_cmpop(predicate: arith.CmpIPredicate) -> object:
    """Build a comparison-operator lowering for the given ``cmpi`` predicate."""

    def _lower(builder, target, args, kwargs):
        lhs = _any_var_to_view(builder, args[0])
        rhs = _any_var_to_view(builder, args[1])
        cmp = _impl._lower_compare(lhs, rhs)
        zero = arith.constant(ir.IntegerType.get_signless(32), 0)
        builder.store_var(target, arith.cmpi(predicate, cmp, zero))

    return _lower


def _lower_contains_op(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``item in container`` -> ``boolean`` for string operands."""
    container = _any_var_to_view(builder, args[0])
    item = _any_var_to_view(builder, args[1])
    builder.store_var(target, _impl._lower_contains(container, item))


def _make_lower_is(impl_fn: object) -> object:
    """Build the ``lower_getattr`` for a no-arg predicate method (e.g. isalpha)."""

    def _impl_lower(builder, target, args, kwargs):
        view = _impl._mlir_string_to_view(builder.load_var(args[0]))
        builder.store_var(target, impl_fn(view, _flags_table_ptr_const()))

    def _getattr(context, builder, target, value, attr=None):
        builder.store_var(target, DeferredMethodCall(value, _impl_lower))

    return _getattr


def _make_lower_binary(impl_fn: object) -> object:
    """Build the ``lower_getattr`` for a one-string-arg method (find/startswith)."""

    def _impl_lower(builder, target, args, kwargs):
        recv = _impl._mlir_string_to_view(builder.load_var(args[0]))
        other = _any_var_to_view(builder, args[1])
        builder.store_var(target, impl_fn(recv, other))

    def _getattr(context, builder, target, value, attr=None):
        builder.store_var(target, DeferredMethodCall(value, _impl_lower))

    return _getattr


def _register() -> None:
    """Register ``mlir_string`` op typing and lowering with ``numba_cuda_mlir``."""
    lower = lowering_registry.lower
    lower_getattr = lowering_registry.lower_getattr

    # len
    typing_registry.register_global(len)(LenMLIRStringTemplate)
    lower(len, mlir_string)(_lower_len_op)
    lower(len, MaskedType)(_lower_masked_len)

    # comparisons
    _str_combos = (
        (MLIRStringType, MLIRStringType),
        (MLIRStringType, types.StringLiteral),
        (types.StringLiteral, MLIRStringType),
    )
    for cmp_op, predicate in _CMPOP_PREDICATES.items():
        typing_registry.register_global(cmp_op)(_make_cmpop_template(cmp_op))
        impl = _make_lower_cmpop(predicate)
        for lhs_ty, rhs_ty in _str_combos:
            lower(cmp_op, lhs_ty, rhs_ty)(impl)

    # contains (``in``)
    typing_registry.register_global(operator.contains)(
        ContainsMLIRStringTemplate
    )
    for lhs_ty, rhs_ty in _str_combos:
        lower(operator.contains, lhs_ty, rhs_ty)(_lower_contains_op)

    # methods
    for name, impl_fn in _IS_METHODS.items():
        lower_getattr(mlir_string, name)(_make_lower_is(impl_fn))
    for name, impl_fn in {**_INT_METHODS, **_BOOL_METHODS}.items():
        lower_getattr(mlir_string, name)(_make_lower_binary(impl_fn))


_register()
