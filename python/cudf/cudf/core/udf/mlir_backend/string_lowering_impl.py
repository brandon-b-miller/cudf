# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-MLIR building blocks composed by the ``mlir_string`` lowerings.

Every helper works on raw MLIR SSA values (no cuDF/NRT dependency) and backs the
read-only, non-string-producing string ops: ``len``, comparisons, ``find``,
``rfind``, ``count``, ``startswith``, ``endswith``, ``in`` and the character-class
predicates (``isalpha`` ...). ``mlir_string`` layout is
``{ptr meminfo, ptr data, i64 nbytes}``; the ops run over an internal *view*
value ``{ptr data, i32 nbytes, i32 length}`` built by :func:`_mlir_string_to_view`
so that ``mlir_string`` and string-literal operands share one representation.
"""

from __future__ import annotations

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm, scf
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.lowering_utilities import GEP_DYNAMIC_INDEX

# Character-type flag bits (cudf::strings::string_character_types), used by the
# character-class predicates against the libcudf flags table.
_CT_DECIMAL = 1 << 0
_CT_NUMERIC = 1 << 1
_CT_DIGIT = 1 << 2
_CT_ALPHA = 1 << 3
_CT_SPACE = 1 << 4
_CT_UPPER = 1 << 5
_CT_LOWER = 1 << 6
_CT_ALPHANUM = _CT_DECIMAL | _CT_NUMERIC | _CT_DIGIT | _CT_ALPHA
_CT_CASE_TYPES = _CT_UPPER | _CT_LOWER
_CT_ALL_TYPES = _CT_ALPHANUM | _CT_CASE_TYPES | _CT_SPACE


# --- Small MLIR helpers (heavily reused across the op suite below) ----------
def _i32() -> ir.Type:
    return ir.IntegerType.get_signless(32)


def _i1() -> ir.Type:
    return ir.IntegerType.get_signless(1)


def _ptr() -> ir.Type:
    return llvm.PointerType.get()


def _const_i32(val: int) -> ir.Value:
    return arith.constant(_i32(), val)


def _const_i64(val: int) -> ir.Value:
    return arith.constant(T.i64(), val)


def _zext_i32_to_i64(val: ir.Value) -> ir.Value:
    return llvm.zext(T.i64(), val)


def _byte_at(data: ir.Value, idx: ir.Value) -> ir.Value:
    """Load ``data[idx]`` as ``i8`` via a dynamic GEP on the ``i8*`` buffer."""
    return llvm.load(
        T.i8(),
        llvm.getelementptr(
            _ptr(), data, [idx], [GEP_DYNAMIC_INDEX], T.i8(), None
        ),
    )


# --- mlir_string / view field accessors ------------------------------------
# view layout: {ptr data, i32 nbytes, i32 length}; only data + nbytes are used.
def _view_data(view_val: ir.Value) -> ir.Value:
    return llvm.extractvalue(_ptr(), view_val, [0])


def _view_nbytes(view_val: ir.Value) -> ir.Value:
    return llvm.extractvalue(_i32(), view_val, [1])


def _ms_extract_nbytes(ms_val: ir.Value) -> ir.Value:
    """Extract ``nbytes`` (i64, field 2) from an ``mlir_string`` value."""
    return llvm.extractvalue(T.i64(), ms_val, [2])


def _mlir_string_to_view(ms_val: ir.Value) -> ir.Value:
    """Build an internal view ``{ptr data, i32 nbytes, i32 length}``.

    ``length`` is set equal to ``nbytes`` here; the ops that need a true
    character count recompute it via :func:`_lower_len`.
    """
    data = llvm.extractvalue(_ptr(), ms_val, [1])
    nbytes_i32 = arith.trunci(_i32(), _ms_extract_nbytes(ms_val))
    view_ty = llvm.StructType.get_literal([_ptr(), _i32(), _i32()])
    val = llvm.UndefOp(view_ty)
    val = llvm.insertvalue(
        container=val, value=data, position=ir.DenseI64ArrayAttr.get([0])
    )
    val = llvm.insertvalue(
        container=val, value=nbytes_i32, position=ir.DenseI64ArrayAttr.get([1])
    )
    return llvm.insertvalue(
        container=val, value=nbytes_i32, position=ir.DenseI64ArrayAttr.get([2])
    )


# --- UTF-8 helpers ----------------------------------------------------------
def _is_begin_utf8_char(byte_val: ir.Value) -> ir.Value:
    """i1: whether ``byte_val`` is a UTF-8 start byte (top 2 bits != ``10``)."""
    masked = arith.andi(arith.extui(_i32(), byte_val), _const_i32(0xC0))
    return arith.cmpi(arith.CmpIPredicate.ne, masked, _const_i32(0x80))


def _utf8_byte_width(first_byte: ir.Value) -> ir.Value:
    """Width in ``[1, 4]`` (i64) of the UTF-8 char starting at ``first_byte``."""
    fb = arith.extui(T.i64(), first_byte)
    is4 = arith.cmpi(
        arith.CmpIPredicate.eq,
        arith.andi(fb, _const_i64(0xF0)),
        _const_i64(0xF0),
    )
    is3 = arith.cmpi(
        arith.CmpIPredicate.eq,
        arith.andi(fb, _const_i64(0xE0)),
        _const_i64(0xE0),
    )
    is2 = arith.cmpi(
        arith.CmpIPredicate.eq,
        arith.andi(fb, _const_i64(0xC0)),
        _const_i64(0xC0),
    )
    width = _const_i64(1)
    width = arith.select(is2, _const_i64(2), width)
    width = arith.select(is3, _const_i64(3), width)
    return arith.select(is4, _const_i64(4), width)


def _decode_utf8_to_codepoint(
    data_ptr: ir.Value, byte_offset: ir.Value, nbytes_i64: ir.Value
) -> tuple[ir.Value, ir.Value]:
    """Decode the UTF-8 char at ``data_ptr[byte_offset]``.

    Returns ``(codepoint_i32, width_i64)``. Continuation-byte loads are clamped
    to ``nbytes - 1`` so multi-byte decoding never reads past the buffer.
    """
    b0 = _byte_at(data_ptr, byte_offset)
    width = _utf8_byte_width(b0)
    b0_32 = arith.extui(_i32(), b0)
    last_valid = arith.subi(nbytes_i64, _const_i64(1))

    cp1 = arith.andi(b0_32, _const_i32(0x7F))

    off1 = arith.minui(arith.addi(byte_offset, _const_i64(1)), last_valid)
    b1 = arith.extui(_i32(), _byte_at(data_ptr, off1))
    cp2 = arith.ori(
        arith.shli(arith.andi(b0_32, _const_i32(0x1F)), _const_i32(6)),
        arith.andi(b1, _const_i32(0x3F)),
    )

    off2 = arith.minui(arith.addi(byte_offset, _const_i64(2)), last_valid)
    b2 = arith.extui(_i32(), _byte_at(data_ptr, off2))
    cp3 = arith.ori(
        arith.ori(
            arith.shli(arith.andi(b0_32, _const_i32(0x0F)), _const_i32(12)),
            arith.shli(arith.andi(b1, _const_i32(0x3F)), _const_i32(6)),
        ),
        arith.andi(b2, _const_i32(0x3F)),
    )

    off3 = arith.minui(arith.addi(byte_offset, _const_i64(3)), last_valid)
    b3 = arith.extui(_i32(), _byte_at(data_ptr, off3))
    cp4 = arith.ori(
        arith.ori(
            arith.shli(arith.andi(b0_32, _const_i32(0x07)), _const_i32(18)),
            arith.shli(arith.andi(b1, _const_i32(0x3F)), _const_i32(12)),
        ),
        arith.ori(
            arith.shli(arith.andi(b2, _const_i32(0x3F)), _const_i32(6)),
            arith.andi(b3, _const_i32(0x3F)),
        ),
    )

    is1 = arith.cmpi(arith.CmpIPredicate.eq, width, _const_i64(1))
    is2 = arith.cmpi(arith.CmpIPredicate.eq, width, _const_i64(2))
    is3 = arith.cmpi(arith.CmpIPredicate.eq, width, _const_i64(3))
    cp = cp4
    cp = arith.select(is3, cp3, cp)
    cp = arith.select(is2, cp2, cp)
    cp = arith.select(is1, cp1, cp)
    return cp, width


# --- len --------------------------------------------------------------------
def _lower_len(str_view: ir.Value) -> ir.Value:
    """``len`` -> ``i32`` character count (number of UTF-8 start bytes)."""
    data = _view_data(str_view)
    nb = _zext_i32_to_i64(_view_nbytes(str_view))
    zero_i32 = _const_i32(0)
    loop = scf.ForOp(_const_i64(0), nb, _const_i64(1), [zero_i32])
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc = loop.inner_iter_args[0]
        inc = arith.select(
            _is_begin_utf8_char(_byte_at(data, idx)), _const_i32(1), zero_i32
        )
        scf.yield_([arith.addi(acc, inc)])
    return loop.results[0]


# --- comparison -------------------------------------------------------------
def _lower_compare(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    """Byte-wise lexicographic compare -> ``i32`` (sign gives order).

    Matches ``cudf::string_view::compare``: the first differing byte's
    difference, else a length comparison.
    """
    lhs_data = _view_data(lhs)
    rhs_data = _view_data(rhs)
    lhs_nb = _zext_i32_to_i64(_view_nbytes(lhs))
    rhs_nb = _zext_i32_to_i64(_view_nbytes(rhs))
    zero_i32 = _const_i32(0)

    min_len = arith.select(
        arith.cmpi(arith.CmpIPredicate.slt, lhs_nb, rhs_nb), lhs_nb, rhs_nb
    )
    loop = scf.ForOp(
        _const_i64(0),
        min_len,
        _const_i64(1),
        [zero_i32, arith.constant(_i1(), 0)],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc_diff = loop.inner_iter_args[0]
        acc_found = loop.inner_iter_args[1]
        b1 = arith.extui(_i32(), _byte_at(lhs_data, idx))
        b2 = arith.extui(_i32(), _byte_at(rhs_data, idx))
        diff = arith.subi(b1, b2)
        ne = arith.cmpi(arith.CmpIPredicate.ne, diff, zero_i32)
        first_diff = arith.andi(
            ne, arith.xori(acc_found, arith.constant(_i1(), 1))
        )
        scf.yield_(
            [
                arith.select(first_diff, diff, acc_diff),
                arith.ori(acc_found, ne),
            ]
        )

    byte_diff = loop.results[0]
    found = loop.results[1]
    lhs_longer = arith.cmpi(arith.CmpIPredicate.sgt, lhs_nb, rhs_nb)
    rhs_longer = arith.cmpi(arith.CmpIPredicate.slt, lhs_nb, rhs_nb)
    len_cmp = arith.select(
        lhs_longer,
        _const_i32(1),
        arith.select(rhs_longer, _const_i32(-1), zero_i32),
    )
    return arith.select(found, byte_diff, len_cmp)


# --- affix / search ---------------------------------------------------------
def _bytes_equal(
    a_data: ir.Value, a_off: ir.Value, b_data: ir.Value, n: ir.Value
) -> ir.Value:
    """i1: whether ``a_data[a_off:a_off+n] == b_data[0:n]`` byte-for-byte."""
    loop = scf.ForOp(
        _const_i64(0), n, _const_i64(1), [arith.constant(_i1(), 1)]
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc = loop.inner_iter_args[0]
        av = _byte_at(a_data, arith.addi(a_off, idx))
        bv = _byte_at(b_data, idx)
        eq = arith.cmpi(arith.CmpIPredicate.eq, av, bv)
        scf.yield_([arith.andi(acc, eq)])
    return loop.results[0]


def _lower_startswith(str_view: ir.Value, prefix: ir.Value) -> ir.Value:
    """``startswith`` -> ``i1``."""
    src_nb = _zext_i32_to_i64(_view_nbytes(str_view))
    pfx_nb = _zext_i32_to_i64(_view_nbytes(prefix))
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, pfx_nb, src_nb)
    # Clamp the compared length to 0 when the prefix cannot fit so the byte
    # loop never reads past the source buffer; the result is false anyway.
    n = arith.select(too_long, _const_i64(0), pfx_nb)
    eq = _bytes_equal(
        _view_data(str_view), _const_i64(0), _view_data(prefix), n
    )
    return arith.andi(arith.xori(too_long, arith.constant(_i1(), 1)), eq)


def _lower_endswith(str_view: ir.Value, suffix: ir.Value) -> ir.Value:
    """``endswith`` -> ``i1``."""
    src_nb = _zext_i32_to_i64(_view_nbytes(str_view))
    sfx_nb = _zext_i32_to_i64(_view_nbytes(suffix))
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, sfx_nb, src_nb)
    n = arith.select(too_long, _const_i64(0), sfx_nb)
    offset = arith.select(too_long, _const_i64(0), arith.subi(src_nb, sfx_nb))
    eq = _bytes_equal(_view_data(str_view), offset, _view_data(suffix), n)
    return arith.andi(arith.xori(too_long, arith.constant(_i1(), 1)), eq)


def _lower_find(str_view: ir.Value, target: ir.Value) -> ir.Value:
    """``find`` -> ``i32`` character position of the first match, else ``-1``."""
    src_data = _view_data(str_view)
    src_nb = _zext_i32_to_i64(_view_nbytes(str_view))
    tgt_data = _view_data(target)
    tgt_nb = _zext_i32_to_i64(_view_nbytes(target))
    zero_i32 = _const_i32(0)
    neg_one_i32 = _const_i32(-1)

    tgt_empty = arith.cmpi(arith.CmpIPredicate.eq, tgt_nb, _const_i64(0))
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, tgt_nb, src_nb)
    search_len = arith.addi(arith.subi(src_nb, tgt_nb), _const_i64(1))
    search_len = arith.select(too_long, _const_i64(0), search_len)

    loop = scf.ForOp(
        _const_i64(0),
        search_len,
        _const_i64(1),
        [neg_one_i32, zero_i32, arith.constant(_i1(), 0)],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc_result = loop.inner_iter_args[0]
        acc_char_pos = loop.inner_iter_args[1]
        acc_found = loop.inner_iter_args[2]
        matched = _bytes_equal(src_data, idx, tgt_data, tgt_nb)
        first_match = arith.andi(
            matched, arith.xori(acc_found, arith.constant(_i1(), 1))
        )
        new_result = arith.select(first_match, acc_char_pos, acc_result)
        new_found = arith.ori(acc_found, matched)
        is_start = _is_begin_utf8_char(_byte_at(src_data, idx))
        new_char_pos = arith.addi(
            acc_char_pos, arith.select(is_start, _const_i32(1), zero_i32)
        )
        scf.yield_([new_result, new_char_pos, new_found])

    return arith.select(tgt_empty, zero_i32, loop.results[0])


def _lower_rfind(str_view: ir.Value, target: ir.Value) -> ir.Value:
    """``rfind`` -> ``i32`` character position of the last match, else ``-1``."""
    src_data = _view_data(str_view)
    src_nb = _zext_i32_to_i64(_view_nbytes(str_view))
    tgt_data = _view_data(target)
    tgt_nb = _zext_i32_to_i64(_view_nbytes(target))
    zero_i32 = _const_i32(0)
    neg_one_i32 = _const_i32(-1)

    tgt_empty = arith.cmpi(arith.CmpIPredicate.eq, tgt_nb, _const_i64(0))
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, tgt_nb, src_nb)
    search_len = arith.addi(arith.subi(src_nb, tgt_nb), _const_i64(1))
    search_len = arith.select(too_long, _const_i64(0), search_len)

    loop = scf.ForOp(
        _const_i64(0), search_len, _const_i64(1), [neg_one_i32, zero_i32]
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc_result = loop.inner_iter_args[0]
        acc_char_pos = loop.inner_iter_args[1]
        matched = _bytes_equal(src_data, idx, tgt_data, tgt_nb)
        new_result = arith.select(matched, acc_char_pos, acc_result)
        is_start = _is_begin_utf8_char(_byte_at(src_data, idx))
        new_char_pos = arith.addi(
            acc_char_pos, arith.select(is_start, _const_i32(1), zero_i32)
        )
        scf.yield_([new_result, new_char_pos])

    haystack_len = _lower_len(str_view)
    return arith.select(
        tgt_empty,
        haystack_len,
        arith.select(too_long, neg_one_i32, loop.results[0]),
    )


def _lower_count(str_view: ir.Value, target: ir.Value) -> ir.Value:
    """``count`` -> ``i32`` non-overlapping occurrence count."""
    src_data = _view_data(str_view)
    src_nb = _zext_i32_to_i64(_view_nbytes(str_view))
    tgt_data = _view_data(target)
    tgt_nb = _zext_i32_to_i64(_view_nbytes(target))
    zero_i32 = _const_i32(0)

    tgt_empty = arith.cmpi(arith.CmpIPredicate.eq, tgt_nb, _const_i64(0))
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, tgt_nb, src_nb)

    loop = scf.ForOp(
        _const_i64(0), src_nb, _const_i64(1), [_const_i64(0), zero_i32]
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        cur_pos = loop.inner_iter_args[0]
        cur_count = loop.inner_iter_args[1]
        at_pos = arith.cmpi(arith.CmpIPredicate.eq, idx, cur_pos)
        remaining = arith.subi(src_nb, cur_pos)
        enough = arith.cmpi(arith.CmpIPredicate.sge, remaining, tgt_nb)
        should_check = arith.andi(at_pos, enough)
        matched = arith.andi(
            should_check, _bytes_equal(src_data, cur_pos, tgt_data, tgt_nb)
        )
        new_count = arith.select(
            matched, arith.addi(cur_count, _const_i32(1)), cur_count
        )
        next_pos = arith.select(
            matched,
            arith.addi(cur_pos, tgt_nb),
            arith.addi(cur_pos, _const_i64(1)),
        )
        final_pos = arith.select(at_pos, next_pos, cur_pos)
        final_count = arith.select(at_pos, new_count, cur_count)
        scf.yield_([final_pos, final_count])

    src_len_plus_one = arith.addi(_lower_len(str_view), _const_i32(1))
    return arith.select(
        tgt_empty,
        src_len_plus_one,
        arith.select(too_long, zero_i32, loop.results[1]),
    )


def _lower_contains(str_view: ir.Value, target: ir.Value) -> ir.Value:
    """``target in str_view`` -> ``i1``."""
    return arith.cmpi(
        arith.CmpIPredicate.sge, _lower_find(str_view, target), _const_i32(0)
    )


# --- character-class predicates (libcudf flags table) -----------------------
def _all_characters_of_type(
    str_view: ir.Value,
    flags_table_ptr: ir.Value,
    types_i32: ir.Value,
    verify_types_i32: ir.Value,
) -> ir.Value:
    """i1: mirror ``cudf::strings::udf::all_characters_of_type``.

    ``flags_table_ptr`` is a device pointer to the ``uint8`` character-flags
    table; ``types``/``verify_types`` are the character-class bit masks.
    """
    data = _view_data(str_view)
    nb = _zext_i32_to_i64(_view_nbytes(str_view))
    zero_i32 = _const_i32(0)
    true_i1 = arith.constant(_i1(), 1)
    max_cp = _const_i32(0xFFFF)
    all_types_i32 = _const_i32(_CT_ALL_TYPES)
    is_empty = arith.cmpi(arith.CmpIPredicate.eq, nb, _const_i64(0))

    loop = scf.ForOp(
        _const_i64(0),
        nb,
        _const_i64(1),
        [_const_i64(0), true_i1, zero_i32],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        cur_off = loop.inner_iter_args[0]
        cur_check = loop.inner_iter_args[1]
        cur_count = loop.inner_iter_args[2]

        at_boundary = arith.cmpi(arith.CmpIPredicate.eq, idx, cur_off)
        cp, char_width = _decode_utf8_to_codepoint(data, cur_off, nb)
        in_range = arith.cmpi(arith.CmpIPredicate.ule, cp, max_cp)
        flag = arith.extui(
            _i32(), _byte_at(flags_table_ptr, arith.extui(T.i64(), cp))
        )
        flag = arith.select(in_range, flag, zero_i32)

        verify_and = arith.andi(verify_types_i32, flag)
        should_verify_1 = arith.cmpi(
            arith.CmpIPredicate.ne, verify_and, zero_i32
        )
        flag_is_zero = arith.cmpi(arith.CmpIPredicate.eq, flag, zero_i32)
        verify_is_all = arith.cmpi(
            arith.CmpIPredicate.eq, verify_types_i32, all_types_i32
        )
        should_verify = arith.ori(
            should_verify_1, arith.andi(flag_is_zero, verify_is_all)
        )

        type_match = arith.cmpi(
            arith.CmpIPredicate.ne, arith.andi(types_i32, flag), zero_i32
        )
        new_check = arith.select(
            should_verify, arith.andi(cur_check, type_match), cur_check
        )
        new_count = arith.select(
            should_verify, arith.addi(cur_count, _const_i32(1)), cur_count
        )
        final_check = arith.select(at_boundary, new_check, cur_check)
        final_count = arith.select(at_boundary, new_count, cur_count)
        next_off = arith.select(
            at_boundary, arith.addi(cur_off, char_width), cur_off
        )
        scf.yield_([next_off, final_check, final_count])

    result_check = loop.results[1]
    has_chars = arith.cmpi(arith.CmpIPredicate.sgt, loop.results[2], zero_i32)
    return arith.andi(
        arith.andi(result_check, has_chars),
        arith.xori(is_empty, true_i1),
    )


def _lower_isalpha(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isalpha`` -> ``i1``."""
    return _all_characters_of_type(
        str_view, flags, _const_i32(_CT_ALPHA), _const_i32(_CT_ALL_TYPES)
    )


def _lower_isalnum(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isalnum`` -> ``i1``."""
    return _all_characters_of_type(
        str_view, flags, _const_i32(_CT_ALPHANUM), _const_i32(_CT_ALL_TYPES)
    )


def _lower_isdigit(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isdigit`` -> ``i1``."""
    return _all_characters_of_type(
        str_view, flags, _const_i32(_CT_DIGIT), _const_i32(_CT_ALL_TYPES)
    )


def _lower_isdecimal(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isdecimal`` -> ``i1``."""
    return _all_characters_of_type(
        str_view, flags, _const_i32(_CT_DECIMAL), _const_i32(_CT_ALL_TYPES)
    )


def _lower_isnumeric(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isnumeric`` -> ``i1``."""
    return _all_characters_of_type(
        str_view, flags, _const_i32(_CT_NUMERIC), _const_i32(_CT_ALL_TYPES)
    )


def _lower_isspace(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isspace`` -> ``i1``."""
    return _all_characters_of_type(
        str_view, flags, _const_i32(_CT_SPACE), _const_i32(_CT_ALL_TYPES)
    )


def _lower_isupper(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isupper`` -> ``i1``."""
    return _all_characters_of_type(
        str_view, flags, _const_i32(_CT_UPPER), _const_i32(_CT_CASE_TYPES)
    )


def _lower_islower(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``islower`` -> ``i1``."""
    return _all_characters_of_type(
        str_view, flags, _const_i32(_CT_LOWER), _const_i32(_CT_CASE_TYPES)
    )


def _lower_istitle(str_view: ir.Value, flags_table_ptr: ir.Value) -> ir.Value:
    """``istitle`` -> ``i1``. Mirrors ``cudf::strings::udf::is_title``."""
    data = _view_data(str_view)
    nb = _zext_i32_to_i64(_view_nbytes(str_view))
    zero_i32 = _const_i32(0)
    true_i1 = arith.constant(_i1(), 1)
    false_i1 = arith.constant(_i1(), 0)
    max_cp = _const_i32(0xFFFF)
    upper_mask = _const_i32(_CT_UPPER)
    case_mask = _const_i32(_CT_CASE_TYPES)

    loop = scf.ForOp(
        _const_i64(0),
        nb,
        _const_i64(1),
        [_const_i64(0), false_i1, true_i1, true_i1],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        cur_off = loop.inner_iter_args[0]
        cur_valid = loop.inner_iter_args[1]
        cur_should_cap = loop.inner_iter_args[2]
        cur_ok = loop.inner_iter_args[3]

        at_boundary = arith.cmpi(arith.CmpIPredicate.eq, idx, cur_off)
        cp, char_width = _decode_utf8_to_codepoint(data, cur_off, nb)
        in_range = arith.cmpi(arith.CmpIPredicate.ule, cp, max_cp)
        flag = arith.extui(
            _i32(), _byte_at(flags_table_ptr, arith.extui(T.i64(), cp))
        )
        flag = arith.select(in_range, flag, zero_i32)

        is_upper = arith.cmpi(
            arith.CmpIPredicate.ne, arith.andi(flag, upper_mask), zero_i32
        )
        is_cased = arith.cmpi(
            arith.CmpIPredicate.ne, arith.andi(flag, case_mask), zero_i32
        )
        wrong_case = arith.xori(cur_should_cap, is_upper)
        check_fail = arith.andi(is_cased, wrong_case)
        new_ok = arith.select(check_fail, false_i1, cur_ok)
        new_valid = arith.select(is_cased, true_i1, cur_valid)
        new_should_cap = arith.select(is_cased, false_i1, true_i1)

        final_ok = arith.select(at_boundary, new_ok, cur_ok)
        final_valid = arith.select(at_boundary, new_valid, cur_valid)
        final_should_cap = arith.select(
            at_boundary, new_should_cap, cur_should_cap
        )
        next_off = arith.select(
            at_boundary, arith.addi(cur_off, char_width), cur_off
        )
        scf.yield_([next_off, final_valid, final_should_cap, final_ok])

    return arith.andi(loop.results[1], loop.results[3])
