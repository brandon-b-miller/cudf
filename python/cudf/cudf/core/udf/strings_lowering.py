# Copyright (c) 2022, NVIDIA CORPORATION.

from strings_udf._typing import string_view
from strings_udf.lowering import (
    string_view_len_impl, 
    string_view_startswith_impl, 
    string_view_endswith_impl, 
    string_view_find_impl, 
    string_view_rfind_impl,
    string_view_contains_impl,
)

from numba import types
from cudf.core.udf.masked_typing import MaskedType

from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)

from numba.core import cgutils
import operator


@cuda_lower(len, MaskedType(string_view))
def masked_string_view_len_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    result = string_view_len_impl(context, builder, types.int32(string_view), (masked_sv.value,))
    ret.value = result
    ret.valid = masked_sv.valid

    return ret._getvalue()


@cuda_lower(
    "MaskedType.startswith", MaskedType(string_view), MaskedType(string_view)
)
def masked_string_view_startswith_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_startswith_impl(
        context, 
        builder, 
        types.boolean(string_view, string_view), 
        (masked_sv_str.value, masked_sv_substr.value)
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()


@cuda_lower(
    "MaskedType.endswith", MaskedType(string_view), MaskedType(string_view)
)
def masked_string_view_endswith_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_endswith_impl(
        context, 
        builder, 
        types.boolean(string_view, string_view), 
        (masked_sv_str.value, masked_sv_substr.value)
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()

@cuda_lower(
    "MaskedType.find", MaskedType(string_view), MaskedType(string_view)
)
def masked_string_view_find_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_find_impl(
        context, 
        builder, 
        types.boolean(string_view, string_view), 
        (masked_sv_str.value, masked_sv_substr.value)
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()

@cuda_lower(
    "MaskedType.rfind", MaskedType(string_view), MaskedType(string_view)
)
def masked_string_view_rfind_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_rfind_impl(
        context, 
        builder, 
        types.boolean(string_view, string_view), 
        (masked_sv_str.value, masked_sv_substr.value)
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()

@cuda_lower(operator.contains, MaskedType(string_view), MaskedType(string_view))
def masked_string_view_contains_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_contains_impl(
        context, 
        builder, 
        types.boolean(string_view, string_view), 
        (masked_sv_str.value, masked_sv_substr.value)
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()
