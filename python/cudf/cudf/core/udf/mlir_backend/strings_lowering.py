# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing and lowering for read-only ``mlir_string`` operations.

Covers the non-NRT, non-string-producing ops on ``mlir_string`` **and** on
``Masked(mlir_string)`` (validity propagated): ``len``, comparisons
(``==``/``!=``/``<``/``<=``/``>``/``>=``), ``in`` (``operator.contains``), the
methods ``find``/``rfind``/``count`` (-> ``int32``) and
``startswith``/``endswith`` (-> ``boolean``), and the character-class predicates
``isalpha``/... (-> ``boolean``). All lowerings are pure MLIR (see
:mod:`cudf.core.udf.mlir_backend.string_lowering_impl`); each operand is reduced
to an internal view ``{ptr, i32 nbytes, i32 length}`` and a validity bit, so
``mlir_string``, ``Masked(mlir_string)`` and string-literal operands share one
allocation-free path. Registered with ``numba_cuda_mlir`` via :func:`_register`.
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

from pylibcudf.strings.char_types import get_character_flags_table_ptr

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

# The single ``Masked(mlir_string)`` instance used as a typing/lowering key.
_masked_string = MaskedType(mlir_string)

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

# two-string method name -> (pure-MLIR impl, return type).
_BINARY_METHODS = {
    "find": (_impl._lower_find, size_type),
    "rfind": (_impl._lower_rfind, size_type),
    "count": (_impl._lower_count, size_type),
    "startswith": (_impl._lower_startswith, types.boolean),
    "endswith": (_impl._lower_endswith, types.boolean),
}

# Operand-type combinations to register op lowerings for (scalar + masked).
_STR_COMBOS = (
    (MLIRStringType, MLIRStringType),
    (MLIRStringType, types.StringLiteral),
    (types.StringLiteral, MLIRStringType),
    (_masked_string, _masked_string),
    (_masked_string, types.StringLiteral),
    (types.StringLiteral, _masked_string),
    (_masked_string, MLIRStringType),
    (MLIRStringType, _masked_string),
)


# --- typing helpers ---------------------------------------------------------
def _is_plain_string(ty: types.Type) -> bool:
    """Whether ``ty`` is an ``mlir_string`` or a compile-time string literal."""
    return isinstance(ty, (MLIRStringType, types.StringLiteral))


def _is_masked_string(ty: types.Type) -> bool:
    """Whether ``ty`` is a ``Masked(mlir_string)``."""
    return isinstance(ty, MaskedType) and isinstance(
        ty.value_type, MLIRStringType
    )


def _is_string_arg(ty: types.Type) -> bool:
    """Whether ``ty`` is any string operand: plain, literal or masked."""
    return _is_plain_string(ty) or _is_masked_string(ty)


# --- typing templates -------------------------------------------------------
class LenMLIRStringTemplate(AbstractTemplate):
    """``len`` over (masked) strings.

    ``len(mlir_string)`` -> ``int32``; ``len(Masked(mlir_string))`` ->
    ``Masked(int32)`` (validity carried through).
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
        if _is_masked_string(arg):
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
        (all-plain operands) or ``Masked(boolean)`` (any masked operand).
    """

    class _CmpOpTemplate(AbstractTemplate):
        key = cmpop

        def generic(self, args, kws):
            """Resolve ``a <cmp> b`` for string operands."""
            return _resolve_string_binary(args, kws, types.boolean)

    return _CmpOpTemplate


class ContainsMLIRStringTemplate(AbstractTemplate):
    """``item in container`` (``operator.contains``) over (masked) strings."""

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
            ``boolean`` (all-plain) or ``Masked(boolean)`` (any masked), else
            ``None``.
        """
        return _resolve_string_binary(args, kws, types.boolean)


def _resolve_string_binary(
    args: tuple[types.Type, ...], kws: dict, retty: types.Type
) -> Signature | None:
    """Resolve a two-operand string op: ``retty`` or ``Masked(retty)``.

    Parameters
    ----------
    args : tuple of types.Type
        The two operand types.
    kws : dict
        Keyword argument types (must be empty).
    retty : types.Type
        The unmasked result type.

    Returns
    -------
    Signature or None
        ``retty(a, b)`` when both operands are plain strings (at least one
        ``mlir_string``), ``Masked(retty)(a, b)`` when any operand is masked,
        else ``None``.
    """
    if len(args) != 2 or kws:
        return None
    a, b = args
    if not (_is_string_arg(a) and _is_string_arg(b)):
        return None
    if _is_masked_string(a) or _is_masked_string(b):
        return nb_signature(MaskedType(retty), a, b)
    if isinstance(a, MLIRStringType) or isinstance(b, MLIRStringType):
        return nb_signature(retty, a, b)
    return None


def _make_method_attr(
    attrname: str, retty: types.Type, masked: bool
) -> object:
    """Build a ``resolve_<method>`` for a one-string-arg method.

    Parameters
    ----------
    attrname : str
        Method name (e.g. ``"find"``).
    retty : types.Type
        The (unmasked) return type.
    masked : bool
        Whether the receiver is ``Masked(mlir_string)`` (wraps the result).

    Returns
    -------
    callable
        A ``resolve_`` function.
    """
    result_ty = MaskedType(retty) if masked else retty

    class _MethodTemplate(AbstractTemplate):
        key = f"MLIRString.{attrname}.{'m' if masked else 's'}"

        def generic(self, args, kws):
            """Resolve ``s.<method>(other)`` for a string ``other``."""
            if len(args) == 1 and not kws:
                return nb_signature(result_ty, args[0], recvr=self.this)
            return None

    recvr = _masked_string if masked else mlir_string

    def resolve(self, mod):
        return types.BoundFunction(_MethodTemplate, recvr)

    return resolve


def _make_is_attr(masked: bool) -> object:
    """Build a ``resolve_<isX>`` for a no-arg predicate method.

    Parameters
    ----------
    masked : bool
        Whether the receiver is ``Masked(mlir_string)`` (wraps the result).

    Returns
    -------
    callable
        A ``resolve_`` function.
    """
    result_ty = MaskedType(types.boolean) if masked else types.boolean

    class _IsTemplate(AbstractTemplate):
        key = f"MLIRString.is.{'m' if masked else 's'}"

        def generic(self, args, kws):
            """Resolve ``s.<isX>()``."""
            return nb_signature(result_ty, recvr=self.this)

    recvr = _masked_string if masked else mlir_string

    def resolve(self, mod):
        return types.BoundFunction(_IsTemplate, recvr)

    return resolve


@typing_registry.register_attr
class MLIRStringAttrs(AttributeTemplate):
    """Attribute typing for ``mlir_string`` methods."""

    key = mlir_string


@typing_registry.register_attr
class MaskedMLIRStringAttrs(AttributeTemplate):
    """Attribute typing for ``Masked(mlir_string)`` methods."""

    key = _masked_string


for _name in _IS_METHODS:
    setattr(MLIRStringAttrs, f"resolve_{_name}", _make_is_attr(masked=False))
    setattr(
        MaskedMLIRStringAttrs, f"resolve_{_name}", _make_is_attr(masked=True)
    )
for _name, (_impl_fn, _rty) in _BINARY_METHODS.items():
    setattr(
        MLIRStringAttrs,
        f"resolve_{_name}",
        _make_method_attr(_name, _rty, masked=False),
    )
    setattr(
        MaskedMLIRStringAttrs,
        f"resolve_{_name}",
        _make_method_attr(_name, _rty, masked=True),
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
    literal_value = builder.get_numba_type(literal_var.name).literal_value
    data_struct = builder.load_var(literal_var)
    ptr_ty = llvm.PointerType.get()
    i32 = ir.IntegerType.get_signless(32)
    view_ty = llvm.StructType.get_literal([ptr_ty, i32, i32])
    val = llvm.UndefOp(view_ty)
    val = llvm.insertvalue(
        container=val,
        value=llvm.extractvalue(ptr_ty, data_struct, [0]),
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    val = llvm.insertvalue(
        container=val,
        value=arith.constant(i32, len(literal_value.encode("utf-8"))),
        position=ir.DenseI64ArrayAttr.get([1]),
    )
    return llvm.insertvalue(
        container=val,
        value=arith.constant(i32, len(literal_value)),
        position=ir.DenseI64ArrayAttr.get([2]),
    )


def _view_and_valid(
    builder: MLIRLower, var: Var, true_val: ir.Value
) -> tuple[ir.Value, ir.Value]:
    """``(view, valid)`` for a string operand var (plain, literal or masked).

    Non-masked operands report validity ``true_val``; a ``Masked(mlir_string)``
    reports its stored validity bit.
    """
    ty = builder.get_numba_type(var.name)
    if isinstance(ty, types.StringLiteral):
        return _literal_to_view(builder, var), true_val
    val = builder.load_var(var)
    if _is_masked_string(ty):
        st = llvm.StructType(val.type)
        ms_val, valid = _extract_masked_value_valid(
            val, st.body[0], st.body[1]
        )
        return _impl._mlir_string_to_view(ms_val), valid
    return _impl._mlir_string_to_view(val), true_val


def _store_maybe_masked(
    builder: MLIRLower, target: Var, result: ir.Value, valid: ir.Value
) -> None:
    """Store ``result`` at ``target``, packing a ``Masked`` value if required."""
    target_type = builder.get_numba_type(target.name)
    if isinstance(target_type, MaskedType):
        builder.store_var(
            target, _pack_masked(builder, target_type, result, valid)
        )
    else:
        builder.store_var(target, result)


# --- lowering impls (unified over plain + masked operands) ------------------
def _lower_len_op(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len`` -> ``int32`` / ``Masked(int32)`` character count."""
    valid_ty = builder.get_mlir_type(types.boolean)
    view, valid = _view_and_valid(
        builder, args[0], arith.constant(valid_ty, 1)
    )
    _store_maybe_masked(builder, target, _impl._lower_len(view), valid)


def _make_lower_cmpop(predicate: arith.CmpIPredicate) -> object:
    """Build a comparison lowering for the given ``cmpi`` predicate."""

    def _lower(builder, target, args, kwargs):
        valid_ty = builder.get_mlir_type(types.boolean)
        true_val = arith.constant(valid_ty, 1)
        lhs, vl = _view_and_valid(builder, args[0], true_val)
        rhs, vr = _view_and_valid(builder, args[1], true_val)
        zero = arith.constant(ir.IntegerType.get_signless(32), 0)
        result = arith.cmpi(predicate, _impl._lower_compare(lhs, rhs), zero)
        _store_maybe_masked(builder, target, result, arith.andi(vl, vr))

    return _lower


def _lower_contains_op(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``item in container`` -> ``boolean`` / ``Masked(boolean)``."""
    valid_ty = builder.get_mlir_type(types.boolean)
    true_val = arith.constant(valid_ty, 1)
    container, vc = _view_and_valid(builder, args[0], true_val)
    item, vi = _view_and_valid(builder, args[1], true_val)
    result = _impl._lower_contains(container, item)
    _store_maybe_masked(builder, target, result, arith.andi(vc, vi))


def _make_lower_is(impl_fn: object) -> object:
    """Build the ``lower_getattr`` for a no-arg predicate method (e.g. isalpha)."""

    def _impl_lower(builder, target, args, kwargs):
        valid_ty = builder.get_mlir_type(types.boolean)
        view, valid = _view_and_valid(
            builder, args[0], arith.constant(valid_ty, 1)
        )
        result = impl_fn(view, _flags_table_ptr_const())
        _store_maybe_masked(builder, target, result, valid)

    def _getattr(context, builder, target, value, attr=None):
        builder.store_var(target, DeferredMethodCall(value, _impl_lower))

    return _getattr


def _make_lower_binary(impl_fn: object) -> object:
    """Build the ``lower_getattr`` for a one-string-arg method (find/startswith)."""

    def _impl_lower(builder, target, args, kwargs):
        valid_ty = builder.get_mlir_type(types.boolean)
        true_val = arith.constant(valid_ty, 1)
        recv, vr = _view_and_valid(builder, args[0], true_val)
        other, vo = _view_and_valid(builder, args[1], true_val)
        result = impl_fn(recv, other)
        _store_maybe_masked(builder, target, result, arith.andi(vr, vo))

    def _getattr(context, builder, target, value, attr=None):
        builder.store_var(target, DeferredMethodCall(value, _impl_lower))

    return _getattr


def _register() -> None:
    """Register ``mlir_string`` op typing and lowering with ``numba_cuda_mlir``."""
    lower = lowering_registry.lower
    lower_getattr = lowering_registry.lower_getattr

    # len (plain + masked)
    typing_registry.register_global(len)(LenMLIRStringTemplate)
    lower(len, mlir_string)(_lower_len_op)
    lower(len, MaskedType)(_lower_len_op)

    # comparisons
    for cmp_op, predicate in _CMPOP_PREDICATES.items():
        typing_registry.register_global(cmp_op)(_make_cmpop_template(cmp_op))
        impl = _make_lower_cmpop(predicate)
        for lhs_ty, rhs_ty in _STR_COMBOS:
            lower(cmp_op, lhs_ty, rhs_ty)(impl)

    # contains (``in``)
    typing_registry.register_global(operator.contains)(
        ContainsMLIRStringTemplate
    )
    for lhs_ty, rhs_ty in _STR_COMBOS:
        lower(operator.contains, lhs_ty, rhs_ty)(_lower_contains_op)

    # methods (registered on both mlir_string and Masked(mlir_string))
    for name, impl_fn in _IS_METHODS.items():
        getattr_fn = _make_lower_is(impl_fn)
        lower_getattr(mlir_string, name)(getattr_fn)
        lower_getattr(_masked_string, name)(getattr_fn)
    for name, (impl_fn, _rty) in _BINARY_METHODS.items():
        getattr_fn = _make_lower_binary(impl_fn)
        lower_getattr(mlir_string, name)(getattr_fn)
        lower_getattr(_masked_string, name)(getattr_fn)


_register()
