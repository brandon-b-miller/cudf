"""
Pure-MLIR lowerings for mlir_string operations.

All string operations are implemented without C++ shims.  Allocation
goes through NRT_Allocate (arena-backed when the host configures it).

The public helpers in this module operate on raw MLIR SSA values so
they can be called from cudf's lowering registration code without
introducing a cudf dependency in numba_cuda_mlir.
"""

import operator

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.dialects import arith, func, scf
from numba_cuda_mlir._mlir.dialects import llvm

from numba_cuda_mlir.lowering_utilities import (
    GEP_DYNAMIC_INDEX,
    get_or_insert_function,
)


# ---------------------------------------------------------------------------
# MLIR helpers
# ---------------------------------------------------------------------------

def _i32():
    return ir.IntegerType.get_signless(32)


def _i64():
    return ir.IntegerType.get_signless(64)


def _ptr():
    return llvm.PointerType.get()


def _const_i64(val):
    return arith.constant(T.i64(), val)


def _zext_i32_to_i64(val):
    return llvm.zext(T.i64(), val)


def _gep_byte_offset(base, offset):
    """base + offset bytes (GEP on i8*)."""
    return llvm.getelementptr(
        _ptr(), base, [offset], [GEP_DYNAMIC_INDEX], T.i8(), None,
    )


def _call_nrt(gpu_module, name, arg_types, result_types, args):
    """Call an NRT function that lives in the same module."""
    ft = ir.FunctionType.get(arg_types, result_types)
    callee = get_or_insert_function(name, ft, gpu_module)
    return func.call(
        result=result_types, callee=callee.name.value, operands_=args,
    )


# ---------------------------------------------------------------------------
# View field accessors
#
# Internal view layout: {ptr data, i32 nbytes, i32 length}
# We only need data (field 0) and nbytes (field 1).
# ---------------------------------------------------------------------------

def view_extract_data(view_val):
    """Extract the data pointer from an internal view SSA value."""
    return llvm.extractvalue(_ptr(), view_val, [0])


def view_extract_nbytes(view_val):
    """Extract nbytes (i32) from an internal view SSA value."""
    return llvm.extractvalue(_i32(), view_val, [1])


# ---------------------------------------------------------------------------
# mlir_string construction
#
# mlir_string layout: {ptr meminfo, ptr data, i64 nbytes}
# ---------------------------------------------------------------------------

def _mlir_string_type():
    """The LLVM struct type for mlir_string."""
    return llvm.StructType.get_literal([_ptr(), _ptr(), T.i64()])


def build_mlir_string(meminfo, data, nbytes_i64):
    """Build a mlir_string SSA value from its three fields."""
    ty = _mlir_string_type()
    undef = llvm.UndefOp(ty)
    with_mi = llvm.insertvalue(
        container=undef,
        value=meminfo,
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    with_data = llvm.insertvalue(
        container=with_mi,
        value=data,
        position=ir.DenseI64ArrayAttr.get([1]),
    )
    return llvm.insertvalue(
        container=with_data,
        value=nbytes_i64,
        position=ir.DenseI64ArrayAttr.get([2]),
    )


def ms_extract_meminfo(ms_val):
    return llvm.extractvalue(_ptr(), ms_val, [0])


def ms_extract_data(ms_val):
    return llvm.extractvalue(_ptr(), ms_val, [1])


def ms_extract_nbytes(ms_val):
    return llvm.extractvalue(T.i64(), ms_val, [2])


def mlir_string_to_view(ms_val):
    """Convert a mlir_string SSA value to an internal view {ptr, i32 nbytes, i32 length}.

    For the time being nbytes == length (assumes per-byte character counting
    that matches how cudf strings work for variable-width UTF-8).
    """
    data = ms_extract_data(ms_val)
    nbytes_i64 = ms_extract_nbytes(ms_val)
    nbytes_i32 = arith.trunci(_i32(), nbytes_i64)
    sv_ty = llvm.StructType.get_literal([_ptr(), _i32(), _i32()])
    undef = llvm.UndefOp(sv_ty)
    with_data = llvm.insertvalue(
        container=undef, value=data,
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    with_bytes = llvm.insertvalue(
        container=with_data, value=nbytes_i32,
        position=ir.DenseI64ArrayAttr.get([1]),
    )
    return llvm.insertvalue(
        container=with_bytes, value=nbytes_i32,
        position=ir.DenseI64ArrayAttr.get([2]),
    )


# ---------------------------------------------------------------------------
# Allocating a new mlir_string via NRT
# ---------------------------------------------------------------------------

def allocate_mlir_string(gpu_module, nbytes_i64):
    """Allocate an NRT-managed byte buffer and return (meminfo, data_ptr).

    Uses NRT_MemInfo_new_varsize which sets up _nrt_varsize_dtor so the
    data buffer is freed when the refcount drops to zero (or no-op under
    arena).  Allocates at least 1 byte to avoid device malloc(0).
    """
    alloc_size = arith.maxui(nbytes_i64, _const_i64(1))
    mi = _call_nrt(
        gpu_module, "NRT_MemInfo_new_varsize",
        [T.i64()], [_ptr()], [alloc_size],
    )
    data = _call_nrt(
        gpu_module, "NRT_MemInfo_data_fast",
        [_ptr()], [_ptr()], [mi],
    )
    return mi, data


# ---------------------------------------------------------------------------
# concat:  (view, view) -> mlir_string
# ---------------------------------------------------------------------------

def lower_concat(gpu_module, lhs_view, rhs_view):
    """Concatenate two view values, returning a mlir_string.

    1. Compute total_bytes = lhs.nbytes + rhs.nbytes
    2. Allocate NRT-managed buffer of total_bytes
    3. memcpy lhs bytes, then rhs bytes
    4. Return {meminfo, data, total_bytes}
    """
    lhs_data = view_extract_data(lhs_view)
    rhs_data = view_extract_data(rhs_view)
    lhs_nbytes = _zext_i32_to_i64(view_extract_nbytes(lhs_view))
    rhs_nbytes = _zext_i32_to_i64(view_extract_nbytes(rhs_view))

    total = arith.addi(lhs_nbytes, rhs_nbytes)

    mi, data = allocate_mlir_string(gpu_module, total)

    llvm.intr_memcpy(data, lhs_data, lhs_nbytes, False)

    dst_offset = _gep_byte_offset(data, lhs_nbytes)
    llvm.intr_memcpy(dst_offset, rhs_data, rhs_nbytes, False)

    return build_mlir_string(mi, data, total)


# ---------------------------------------------------------------------------
# Byte-level comparisons -> i1
# ---------------------------------------------------------------------------

def _compare_bytes(lhs, rhs, predicate):
    """Compare two view values lexicographically.

    Returns an i1.  Uses a byte-by-byte loop over the shorter length,
    then falls back to length comparison if the common prefix matches.
    """
    lhs_data = view_extract_data(lhs)
    rhs_data = view_extract_data(rhs)
    lhs_nb = _zext_i32_to_i64(view_extract_nbytes(lhs))
    rhs_nb = _zext_i32_to_i64(view_extract_nbytes(rhs))

    zero = _const_i64(0)
    one = _const_i64(1)

    # min_len = min(lhs_nb, rhs_nb)
    lhs_shorter = arith.cmpi(arith.CmpIPredicate.ult, lhs_nb, rhs_nb)
    min_len = arith.select(lhs_shorter, lhs_nb, rhs_nb)

    # Scan bytes for first difference.  Accumulate index of first diff
    # (or min_len if all match).
    # iter_arg = first_diff_index, starts at min_len (meaning "no diff found")
    loop = scf.ForOp(zero, min_len, one, [min_len])
    iv = loop.body.arguments[0]
    current_diff = loop.body.arguments[1]
    with ir.InsertionPoint(loop.body):
        lhs_byte = llvm.load(T.i8(), _gep_byte_offset(lhs_data, iv))
        rhs_byte = llvm.load(T.i8(), _gep_byte_offset(rhs_data, iv))
        bytes_ne = arith.cmpi(arith.CmpIPredicate.ne, lhs_byte, rhs_byte)
        # If we already found a diff, keep it; otherwise update if this is a diff
        already_found = arith.cmpi(arith.CmpIPredicate.ult, current_diff, min_len)
        new_diff = arith.select(already_found, current_diff, iv)
        updated = arith.select(bytes_ne, new_diff, current_diff)
        scf.YieldOp([updated])

    first_diff = loop.results[0]
    found_diff = arith.cmpi(arith.CmpIPredicate.ult, first_diff, min_len)

    # If a byte difference was found, compare those bytes.
    # Otherwise compare lengths.
    lhs_at_diff = llvm.load(T.i8(), _gep_byte_offset(lhs_data, first_diff))
    rhs_at_diff = llvm.load(T.i8(), _gep_byte_offset(rhs_data, first_diff))

    byte_cmp = arith.cmpi(predicate, lhs_at_diff, rhs_at_diff)
    len_cmp = arith.cmpi(predicate, lhs_nb, rhs_nb)

    return arith.select(found_diff, byte_cmp, len_cmp)


def lower_eq(lhs, rhs):
    """mlir_string == mlir_string -> i1."""
    return _compare_bytes(lhs, rhs, arith.CmpIPredicate.eq)


def lower_ne(lhs, rhs):
    """mlir_string != mlir_string -> i1."""
    eq = lower_eq(lhs, rhs)
    return arith.xori(eq, arith.constant(ir.IntegerType.get_signless(1), 1))


def lower_lt(lhs, rhs):
    return _compare_bytes(lhs, rhs, arith.CmpIPredicate.ult)


def lower_le(lhs, rhs):
    return _compare_bytes(lhs, rhs, arith.CmpIPredicate.ule)


def lower_gt(lhs, rhs):
    return _compare_bytes(lhs, rhs, arith.CmpIPredicate.ugt)


def lower_ge(lhs, rhs):
    return _compare_bytes(lhs, rhs, arith.CmpIPredicate.uge)


# ---------------------------------------------------------------------------
# strip / lstrip / rstrip -> mlir_string
# ---------------------------------------------------------------------------

def _is_strip_char(byte_val, strip_data, strip_nb):
    """Return i1 true if byte_val should be stripped.

    Mirrors libcudf detail::strip: if the strip string is empty, strip chars
    with code <= 0x20 (whitespace).  Otherwise scan the strip string for a
    matching byte.
    """
    zero = _const_i64(0)
    one = _const_i64(1)
    strip_empty = arith.cmpi(arith.CmpIPredicate.eq, strip_nb, zero)
    is_ws = arith.cmpi(arith.CmpIPredicate.ule, byte_val, arith.constant(T.i8(), 0x20))

    found_loop = scf.ForOp(zero, strip_nb, one, [arith.constant(ir.IntegerType.get_signless(1), 0)])
    si = found_loop.body.arguments[0]
    found = found_loop.body.arguments[1]
    with ir.InsertionPoint(found_loop.body):
        sb = llvm.load(T.i8(), _gep_byte_offset(strip_data, si))
        eq = arith.cmpi(arith.CmpIPredicate.eq, byte_val, sb)
        scf.YieldOp([arith.ori(found, eq)])
    in_set = found_loop.results[0]

    return arith.select(strip_empty, is_ws, in_set)


def _strip_core(gpu_module, str_view, strip_chars, do_left, do_right):
    """Shared implementation for strip/lstrip/rstrip.

    strip_chars is the view of characters to strip (may be empty
    for whitespace mode).
    """
    data = view_extract_data(str_view)
    nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    strip_data = view_extract_data(strip_chars)
    strip_nb = _zext_i32_to_i64(view_extract_nbytes(strip_chars))

    zero = _const_i64(0)
    one = _const_i64(1)

    if do_left:
        loop_l = scf.ForOp(zero, nb, one, [zero])
        i = loop_l.body.arguments[0]
        left = loop_l.body.arguments[1]
        with ir.InsertionPoint(loop_l.body):
            byte_val = llvm.load(T.i8(), _gep_byte_offset(data, i))
            should_strip = _is_strip_char(byte_val, strip_data, strip_nb)
            still_stripping = arith.cmpi(arith.CmpIPredicate.eq, left, i)
            advance = arith.andi(should_strip, still_stripping)
            new_left = arith.select(advance, arith.addi(left, one), left)
            scf.YieldOp([new_left])
        left_idx = loop_l.results[0]
    else:
        left_idx = zero

    if do_right:
        loop_r = scf.ForOp(zero, nb, one, [nb])
        j = loop_r.body.arguments[0]
        right = loop_r.body.arguments[1]
        with ir.InsertionPoint(loop_r.body):
            ridx = arith.subi(arith.subi(nb, j), one)
            byte_r = llvm.load(T.i8(), _gep_byte_offset(data, ridx))
            should_strip_r = _is_strip_char(byte_r, strip_data, strip_nb)
            expected_right = arith.subi(nb, j)
            still_strip_r = arith.cmpi(arith.CmpIPredicate.eq, right, expected_right)
            advance_r = arith.andi(should_strip_r, still_strip_r)
            new_right = arith.select(advance_r, ridx, right)
            scf.YieldOp([new_right])
        right_idx = loop_r.results[0]
    else:
        right_idx = nb

    valid = arith.cmpi(arith.CmpIPredicate.ult, left_idx, right_idx)
    result_len = arith.select(valid, arith.subi(right_idx, left_idx), zero)

    mi, out_data = allocate_mlir_string(gpu_module, result_len)
    src = _gep_byte_offset(data, left_idx)
    llvm.intr_memcpy(out_data, src, result_len, False)

    return build_mlir_string(mi, out_data, result_len)


def lower_strip(gpu_module, str_view, strip_chars=None):
    """mlir_string.strip(chars) -> mlir_string."""
    if strip_chars is None:
        strip_chars = _make_empty_view()
    return _strip_core(gpu_module, str_view, strip_chars, True, True)


def lower_lstrip(gpu_module, str_view, strip_chars=None):
    """mlir_string.lstrip(chars) -> mlir_string."""
    if strip_chars is None:
        strip_chars = _make_empty_view()
    return _strip_core(gpu_module, str_view, strip_chars, True, False)


def lower_rstrip(gpu_module, str_view, strip_chars=None):
    """mlir_string.rstrip(chars) -> mlir_string."""
    if strip_chars is None:
        strip_chars = _make_empty_view()
    return _strip_core(gpu_module, str_view, strip_chars, False, True)


def _make_empty_view():
    """Build a view with nbytes=0 (for whitespace-mode strip)."""
    sv_ty = llvm.StructType.get_literal([_ptr(), _i32(), _i32()])
    undef = llvm.UndefOp(sv_ty)
    zero_i32 = arith.constant(_i32(), 0)
    null_ptr = llvm.ZeroOp(_ptr())
    s = llvm.insertvalue(container=undef, value=null_ptr, position=ir.DenseI64ArrayAttr.get([0]))
    s = llvm.insertvalue(container=s, value=zero_i32, position=ir.DenseI64ArrayAttr.get([1]))
    return llvm.insertvalue(container=s, value=zero_i32, position=ir.DenseI64ArrayAttr.get([2]))


# ---------------------------------------------------------------------------
# replace: (view, old, new) -> mlir_string
#
# Simple O(n*m) implementation: count occurrences, compute result size,
# allocate, then scan-and-copy.
# ---------------------------------------------------------------------------

def lower_replace(gpu_module, src, old, new):
    """mlir_string.replace(old, new) -> mlir_string.

    Handles the empty-target special case (insert replacement between every
    character) following the same semantics as libcudf's C++ replace.
    """
    s_data = view_extract_data(src)
    o_data = view_extract_data(old)
    n_data = view_extract_data(new)
    s_nb = _zext_i32_to_i64(view_extract_nbytes(src))
    o_nb = _zext_i32_to_i64(view_extract_nbytes(old))
    n_nb = _zext_i32_to_i64(view_extract_nbytes(new))

    zero = _const_i64(0)
    one = _const_i64(1)
    false_val = arith.constant(ir.IntegerType.get_signless(1), 0)

    target_empty = arith.cmpi(arith.CmpIPredicate.eq, o_nb, zero)

    # --- Count non-overlapping occurrences ---
    # When o_nb=0 the inner loop body never executes so every position
    # "matches", giving occ_count = s_nb + 1.  The formula below works
    # for both empty and non-empty targets.
    long_enough = arith.cmpi(arith.CmpIPredicate.uge, s_nb, o_nb)
    diff = arith.subi(s_nb, o_nb)
    search_range = arith.select(long_enough, arith.addi(diff, one), zero)

    count_loop = scf.ForOp(zero, search_range, one, [zero, zero])
    ci = count_loop.body.arguments[0]
    c_count = count_loop.body.arguments[1]
    c_skip = count_loop.body.arguments[2]
    with ir.InsertionPoint(count_loop.body):
        c_should_skip = arith.cmpi(arith.CmpIPredicate.ult, ci, c_skip)
        c_inner = scf.ForOp(zero, o_nb, one, [zero])
        cj = c_inner.body.arguments[0]
        cmm = c_inner.body.arguments[1]
        with ir.InsertionPoint(c_inner.body):
            c_hidx = arith.addi(ci, cj)
            c_hb = llvm.load(T.i8(), _gep_byte_offset(s_data, c_hidx))
            c_ob = llvm.load(T.i8(), _gep_byte_offset(o_data, cj))
            c_ne = arith.cmpi(arith.CmpIPredicate.ne, c_hb, c_ob)
            c_ne64 = arith.extui(T.i64(), c_ne)
            scf.YieldOp([arith.addi(cmm, c_ne64)])
        c_matches = arith.cmpi(arith.CmpIPredicate.eq, c_inner.results[0], zero)
        c_valid = arith.andi(
            c_matches, arith.cmpi(arith.CmpIPredicate.eq, c_should_skip, false_val),
        )
        c_new_count = arith.select(c_valid, arith.addi(c_count, one), c_count)
        c_new_skip = arith.select(c_valid, arith.addi(ci, o_nb), c_skip)
        scf.YieldOp([c_new_count, c_new_skip])

    occ_count = count_loop.results[0]

    # result_len = s_nb - occ_count * o_nb + occ_count * n_nb
    result_len = arith.addi(
        arith.subi(s_nb, arith.muli(occ_count, o_nb)),
        arith.muli(occ_count, n_nb),
    )

    mi, out_data = allocate_mlir_string(gpu_module, result_len)

    scratch_one = llvm.alloca(
        _ptr(), arith.constant(T.i64(), 1), elem_type=T.i8(),
    )

    # --- Scan-and-copy pass ---
    # Each iteration can emit up to two chunks:
    #   chunk_a = replacement bytes  (when a match fires)
    #   chunk_b = one source byte    (when no non-empty match consumed it)
    # Lengths are set to 0 for inactive paths so the memcpy is a no-op.

    build_loop = scf.ForOp(zero, s_nb, one, [zero, zero])
    si = build_loop.body.arguments[0]
    out_off = build_loop.body.arguments[1]
    b_skip = build_loop.body.arguments[2]
    with ir.InsertionPoint(build_loop.body):
        b_should_skip = arith.cmpi(arith.CmpIPredicate.ult, si, b_skip)
        active = arith.cmpi(arith.CmpIPredicate.eq, b_should_skip, false_val)

        remaining = arith.subi(s_nb, si)
        can_match = arith.cmpi(arith.CmpIPredicate.uge, remaining, o_nb)
        b_inner = scf.ForOp(zero, o_nb, one, [zero])
        bj = b_inner.body.arguments[0]
        bmm = b_inner.body.arguments[1]
        with ir.InsertionPoint(b_inner.body):
            b_hidx = arith.addi(si, bj)
            b_hb = llvm.load(T.i8(), _gep_byte_offset(s_data, b_hidx))
            b_ob = llvm.load(T.i8(), _gep_byte_offset(o_data, bj))
            b_ne = arith.cmpi(arith.CmpIPredicate.ne, b_hb, b_ob)
            b_ne64 = arith.extui(T.i64(), b_ne)
            scf.YieldOp([arith.addi(bmm, b_ne64)])

        b_match = arith.andi(
            can_match,
            arith.cmpi(arith.CmpIPredicate.eq, b_inner.results[0], zero),
        )
        b_match_valid = arith.andi(b_match, active)

        nonempty_match = arith.andi(
            b_match_valid,
            arith.cmpi(arith.CmpIPredicate.eq, target_empty, false_val),
        )

        # write_repl: emit replacement bytes
        #   empty target + active  OR  non-empty-target match
        write_repl = arith.ori(
            arith.andi(target_empty, active),
            nonempty_match,
        )
        # write_byte: emit source byte
        #   empty target + active  OR  non-empty, no match, active
        write_byte = arith.ori(
            arith.andi(target_empty, active),
            arith.andi(
                arith.cmpi(arith.CmpIPredicate.eq, nonempty_match, false_val),
                active,
            ),
        )

        len_a = arith.select(write_repl, n_nb, zero)
        llvm.intr_memcpy(
            _gep_byte_offset(out_data, out_off), n_data, len_a, False,
        )

        off_after_a = arith.addi(out_off, len_a)

        one_byte = llvm.load(T.i8(), _gep_byte_offset(s_data, si))
        llvm.store(one_byte, scratch_one)
        len_b = arith.select(write_byte, one, zero)
        llvm.intr_memcpy(
            _gep_byte_offset(out_data, off_after_a), scratch_one, len_b, False,
        )

        new_out_off = arith.addi(off_after_a, len_b)
        new_skip = arith.select(nonempty_match, arith.addi(si, o_nb), b_skip)
        scf.YieldOp([new_out_off, new_skip])

    # Trailing replacement for empty target: repl is inserted after the last
    # character as well.
    final_off = build_loop.results[0]
    trailing_len = arith.select(target_empty, n_nb, zero)
    llvm.intr_memcpy(
        _gep_byte_offset(out_data, final_off), n_data, trailing_len, False,
    )

    return build_mlir_string(mi, out_data, result_len)




# ---------------------------------------------------------------------------
# rfind: mlir_string.rfind(needle) -> i32.  Returns -1 if not found.
# ---------------------------------------------------------------------------

def lower_rfind(haystack, needle):
    """mlir_string.rfind(needle) -> i32 character position. -1 if not found.

    Scans forward tracking character positions; keeps updating result on each
    match so that the last match position is returned.
    """
    h_data = view_extract_data(haystack)
    n_data = view_extract_data(needle)
    h_nb = _zext_i32_to_i64(view_extract_nbytes(haystack))
    n_nb = _zext_i32_to_i64(view_extract_nbytes(needle))

    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)
    zero_i32 = _const_i32(0)
    neg_one_i32 = _const_i32(-1)

    tgt_empty = arith.cmpi(arith.CmpIPredicate.eq, n_nb, zero_i64)
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, n_nb, h_nb)
    diff = arith.subi(h_nb, n_nb)
    search_len = arith.addi(diff, one_i64)
    search_len = arith.select(too_long, zero_i64, search_len)

    # Loop: [result_pos i32, char_pos i32]
    loop = scf.ForOp(
        zero_i64, search_len, one_i64,
        [neg_one_i32, zero_i32],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc_result = loop.inner_iter_args[0]
        acc_char_pos = loop.inner_iter_args[1]

        cmp_loop = scf.ForOp(
            zero_i64, n_nb, one_i64,
            [arith.constant(T.bool(), 1)],
        )
        with ir.InsertionPoint(cmp_loop.body):
            cidx = cmp_loop.induction_variable
            cmatch = cmp_loop.inner_iter_args[0]
            sb = llvm.load(T.i8(), _gep_byte_offset(h_data, arith.addi(idx, cidx)))
            tb = llvm.load(T.i8(), _gep_byte_offset(n_data, cidx))
            eq = arith.cmpi(arith.CmpIPredicate.eq, sb, tb)
            scf.yield_([arith.andi(cmatch, eq)])

        matched = cmp_loop.results[0]
        new_result = arith.select(matched, acc_char_pos, acc_result)

        byte_val = llvm.load(T.i8(), _gep_byte_offset(h_data, idx))
        is_start = _is_begin_utf8_char(byte_val)
        new_char_pos = arith.addi(acc_char_pos, arith.select(is_start, _const_i32(1), zero_i32))

        scf.yield_([new_result, new_char_pos])

    # Empty target: return len(haystack) (Python convention for rfind(''))
    haystack_len = lower_len(haystack)
    result = arith.select(tgt_empty, haystack_len,
                          arith.select(too_long, neg_one_i32, loop.results[0]))
    return result


# ---------------------------------------------------------------------------
# UTF-8 helpers for upper/lower case conversion
#
# The algorithm mirrors libcudf's convert_case():
#   1. Iterate over UTF-8 bytes, decoding each character to a code point
#   2. Look up the code point in the flags table
#   3. If the flag matches case_flag (or is special), convert:
#      - Non-special: look up in cases_table, encode back to UTF-8
#      - Special: look up in special_case_mapping_table, emit multiple chars
#   4. Otherwise copy the original bytes
#
# The tables are device pointers passed in as i64 constants by the caller.
# Table types:
#   flags_table:   uint8_t[65536]   (character_flags_table_type)
#   cases_table:   uint16_t[65536]  (character_cases_table_type)
#   special_case_mapping_table: struct[499] each 16 bytes:
#       {uint16_t num_upper, uint16_t upper[3], uint16_t num_lower, uint16_t lower[3]}
#
# Two-pass approach: pass 1 counts output bytes, pass 2 writes.
# ---------------------------------------------------------------------------

def _const_i32(val):
    return arith.constant(_i32(), val)


def _const_i8(val):
    return arith.constant(T.i8(), val)


def _utf8_byte_width(first_byte):
    """Given the first byte of a UTF-8 sequence (i8), return the width (i64) in [1..4]."""
    fb = arith.extui(T.i64(), first_byte)
    is4 = arith.cmpi(arith.CmpIPredicate.eq,
                     arith.andi(fb, _const_i64(0xF0)), _const_i64(0xF0))
    is3 = arith.cmpi(arith.CmpIPredicate.eq,
                     arith.andi(fb, _const_i64(0xE0)), _const_i64(0xE0))
    is2 = arith.cmpi(arith.CmpIPredicate.eq,
                     arith.andi(fb, _const_i64(0xC0)), _const_i64(0xC0))
    w = _const_i64(1)
    w = arith.select(is2, _const_i64(2), w)
    w = arith.select(is3, _const_i64(3), w)
    w = arith.select(is4, _const_i64(4), w)
    return w


def _decode_utf8_to_codepoint(data_ptr, byte_offset, nbytes_i64):
    """Read a UTF-8 character at data_ptr+byte_offset, return (codepoint_i32, width_i64).

    nbytes_i64 is the total buffer length; speculative loads for multi-byte
    sequences are clamped to stay within bounds.
    """
    b0_ptr = _gep_byte_offset(data_ptr, byte_offset)
    b0 = llvm.load(T.i8(), b0_ptr)
    width = _utf8_byte_width(b0)

    b0_32 = arith.extui(_i32(), b0)
    last_valid = arith.subi(nbytes_i64, _const_i64(1))

    cp1 = arith.andi(b0_32, _const_i32(0x7F))

    off1 = arith.minui(arith.addi(byte_offset, _const_i64(1)), last_valid)
    b1 = arith.extui(_i32(), llvm.load(T.i8(), _gep_byte_offset(data_ptr, off1)))
    cp2 = arith.ori(
        arith.shli(arith.andi(b0_32, _const_i32(0x1F)), _const_i32(6)),
        arith.andi(b1, _const_i32(0x3F)))

    off2 = arith.minui(arith.addi(byte_offset, _const_i64(2)), last_valid)
    b2 = arith.extui(_i32(), llvm.load(T.i8(), _gep_byte_offset(data_ptr, off2)))
    cp3 = arith.ori(
        arith.ori(
            arith.shli(arith.andi(b0_32, _const_i32(0x0F)), _const_i32(12)),
            arith.shli(arith.andi(b1, _const_i32(0x3F)), _const_i32(6))),
        arith.andi(b2, _const_i32(0x3F)))

    off3 = arith.minui(arith.addi(byte_offset, _const_i64(3)), last_valid)
    b3 = arith.extui(_i32(), llvm.load(T.i8(), _gep_byte_offset(data_ptr, off3)))
    cp4 = arith.ori(
        arith.ori(
            arith.shli(arith.andi(b0_32, _const_i32(0x07)), _const_i32(18)),
            arith.shli(arith.andi(b1, _const_i32(0x3F)), _const_i32(12))),
        arith.ori(
            arith.shli(arith.andi(b2, _const_i32(0x3F)), _const_i32(6)),
            arith.andi(b3, _const_i32(0x3F))))

    is1 = arith.cmpi(arith.CmpIPredicate.eq, width, _const_i64(1))
    is2 = arith.cmpi(arith.CmpIPredicate.eq, width, _const_i64(2))
    is3 = arith.cmpi(arith.CmpIPredicate.eq, width, _const_i64(3))

    cp = cp4
    cp = arith.select(is3, cp3, cp)
    cp = arith.select(is2, cp2, cp)
    cp = arith.select(is1, cp1, cp)

    return cp, width


def _codepoint_to_utf8_bytes(cp_i32):
    """Convert a code point (i32) to its UTF-8 byte count (i64)."""
    is1 = arith.cmpi(arith.CmpIPredicate.ult, cp_i32, _const_i32(0x80))
    is2 = arith.cmpi(arith.CmpIPredicate.ult, cp_i32, _const_i32(0x800))
    is3 = arith.cmpi(arith.CmpIPredicate.ult, cp_i32, _const_i32(0x10000))
    n = _const_i64(4)
    n = arith.select(is3, _const_i64(3), n)
    n = arith.select(is2, _const_i64(2), n)
    n = arith.select(is1, _const_i64(1), n)
    return n


def _write_codepoint_utf8(data_ptr, byte_offset, cp_i32):
    """Encode a code point as UTF-8 at data_ptr+byte_offset. Return bytes written (i64)."""
    nbytes = _codepoint_to_utf8_bytes(cp_i32)
    is1 = arith.cmpi(arith.CmpIPredicate.eq, nbytes, _const_i64(1))
    is2 = arith.cmpi(arith.CmpIPredicate.eq, nbytes, _const_i64(2))
    is3 = arith.cmpi(arith.CmpIPredicate.eq, nbytes, _const_i64(3))

    b1_0 = arith.trunci(T.i8(), cp_i32)

    b2_0 = arith.trunci(T.i8(), arith.ori(arith.shrui(cp_i32, _const_i32(6)), _const_i32(0xC0)))
    b2_1 = arith.trunci(T.i8(), arith.ori(arith.andi(cp_i32, _const_i32(0x3F)), _const_i32(0x80)))

    b3_0 = arith.trunci(T.i8(), arith.ori(arith.shrui(cp_i32, _const_i32(12)), _const_i32(0xE0)))
    b3_1 = arith.trunci(T.i8(), arith.ori(arith.andi(arith.shrui(cp_i32, _const_i32(6)), _const_i32(0x3F)), _const_i32(0x80)))
    b3_2 = arith.trunci(T.i8(), arith.ori(arith.andi(cp_i32, _const_i32(0x3F)), _const_i32(0x80)))

    b4_0 = arith.trunci(T.i8(), arith.ori(arith.shrui(cp_i32, _const_i32(18)), _const_i32(0xF0)))
    b4_1 = arith.trunci(T.i8(), arith.ori(arith.andi(arith.shrui(cp_i32, _const_i32(12)), _const_i32(0x3F)), _const_i32(0x80)))
    b4_2 = arith.trunci(T.i8(), arith.ori(arith.andi(arith.shrui(cp_i32, _const_i32(6)), _const_i32(0x3F)), _const_i32(0x80)))
    b4_3 = arith.trunci(T.i8(), arith.ori(arith.andi(cp_i32, _const_i32(0x3F)), _const_i32(0x80)))

    byte0 = b4_0
    byte0 = arith.select(is3, b3_0, byte0)
    byte0 = arith.select(is2, b2_0, byte0)
    byte0 = arith.select(is1, b1_0, byte0)

    byte1 = b4_1
    byte1 = arith.select(is3, b3_1, byte1)
    byte1 = arith.select(is2, b2_1, byte1)

    byte2 = b4_2
    byte2 = arith.select(is3, b3_2, byte2)

    byte3 = b4_3

    # Write all 4 positions unconditionally; only advance by nbytes
    dst = _gep_byte_offset(data_ptr, byte_offset)
    llvm.store(byte0, dst)
    llvm.store(byte1, _gep_byte_offset(data_ptr, arith.addi(byte_offset, _const_i64(1))))
    llvm.store(byte2, _gep_byte_offset(data_ptr, arith.addi(byte_offset, _const_i64(2))))
    llvm.store(byte3, _gep_byte_offset(data_ptr, arith.addi(byte_offset, _const_i64(3))))

    return nbytes


# Flag bit positions matching libcudf char_tables.hpp
_FLAG_UPPER = 1 << 5    # 0x20
_FLAG_LOWER = 1 << 6    # 0x40
_FLAG_SPECIAL = 1 << 7  # 0x80
_SPECIAL_CASE_PRIME = 499


def _case_convert_char_loop(data_ptr, nbytes_i64, flags_table_ptr, cases_table_ptr,
                            special_table_ptr, case_flag_i8, is_upper, out_ptr):
    """Core character iteration for case conversion.

    When out_ptr is None, only counts output bytes (pass 1).
    When out_ptr is provided, writes converted bytes (pass 2).
    Returns total output bytes (i64).

    Pass 2 writes each path's output into its own scratch buffer, then does
    a single memcpy from the selected source into the real output.  This
    avoids the arith.select-over-side-effects bug where all write paths
    would execute unconditionally.
    """
    zero = _const_i64(0)
    one = _const_i64(1)
    three = _const_i64(3)
    i16_ty = ir.IntegerType.get_signless(16)
    false_i1 = arith.constant(ir.IntegerType.get_signless(1), 0)

    if out_ptr is not None:
        scratch_normal = llvm.alloca(
            _ptr(), arith.constant(T.i64(), 4), elem_type=T.i8(),
        )
        scratch_special = llvm.alloca(
            _ptr(), arith.constant(T.i64(), 12), elem_type=T.i8(),
        )

    loop = scf.ForOp(zero, nbytes_i64, one, [zero, zero])
    pos = loop.body.arguments[0]
    out_off = loop.body.arguments[1]
    skip_until = loop.body.arguments[2]
    with ir.InsertionPoint(loop.body):
        should_skip = arith.cmpi(arith.CmpIPredicate.ult, pos, skip_until)

        cp, width = _decode_utf8_to_codepoint(data_ptr, pos, nbytes_i64)

        cp_i64 = arith.extui(T.i64(), cp)
        in_range = arith.cmpi(arith.CmpIPredicate.ule, cp, _const_i32(0xFFFF))
        flag_ptr = _gep_byte_offset(flags_table_ptr, cp_i64)
        flag_raw = llvm.load(T.i8(), flag_ptr)
        flag = arith.select(in_range, flag_raw, _const_i8(0))

        matches_case = arith.cmpi(arith.CmpIPredicate.ne,
                                  arith.andi(flag, case_flag_i8), _const_i8(0))
        is_special = arith.cmpi(arith.CmpIPredicate.ne,
                                arith.andi(flag, _const_i8(_FLAG_SPECIAL)), _const_i8(0))
        is_upper_or_lower = arith.cmpi(arith.CmpIPredicate.ne,
                                       arith.andi(flag, _const_i8(_FLAG_UPPER | _FLAG_LOWER)),
                                       _const_i8(0))
        special_only = arith.andi(is_special, arith.cmpi(arith.CmpIPredicate.eq,
                                                         is_upper_or_lower, false_i1))
        needs_convert = arith.ori(matches_case, special_only)

        # --- Non-special: cases_table[cp] ---
        cp_times2 = arith.shli(cp_i64, _const_i64(1))
        case_entry_ptr = _gep_byte_offset(cases_table_ptr, cp_times2)
        case_cp_i16 = llvm.load(i16_ty, case_entry_ptr)
        case_cp_i32 = arith.extui(_i32(), case_cp_i16)
        normal_out_bytes = _codepoint_to_utf8_bytes(case_cp_i32)
        normal_new_off = arith.addi(out_off, normal_out_bytes)

        # --- Special case: special_case_mapping_table[cp % 499] ---
        hash_idx = arith.remui(cp, _const_i32(_SPECIAL_CASE_PRIME))
        hash_i64 = arith.extui(T.i64(), hash_idx)
        entry_offset = arith.muli(hash_i64, _const_i64(16))
        entry_ptr = _gep_byte_offset(special_table_ptr, entry_offset)

        count_offset = arith.select(
            arith.constant(ir.IntegerType.get_signless(1), is_upper),
            _const_i64(0), _const_i64(8))
        num_chars_i16 = llvm.load(i16_ty, _gep_byte_offset(entry_ptr, count_offset))
        num_chars_raw = arith.extui(T.i64(), num_chars_i16)
        num_chars_i64 = arith.select(
            arith.cmpi(arith.CmpIPredicate.ugt, num_chars_raw, three),
            three, num_chars_raw)

        chars_base_offset = arith.select(
            arith.constant(ir.IntegerType.get_signless(1), is_upper),
            _const_i64(2), _const_i64(10))
        chars_base = _gep_byte_offset(entry_ptr, chars_base_offset)

        # Count special-case output bytes (read-only, always executed)
        sc_count = scf.ForOp(zero, num_chars_i64, one, [zero])
        sc_ci = sc_count.body.arguments[0]
        sc_cacc = sc_count.body.arguments[1]
        with ir.InsertionPoint(sc_count.body):
            sc_ccp16 = llvm.load(
                i16_ty,
                _gep_byte_offset(chars_base, arith.muli(sc_ci, _const_i64(2))),
            )
            sc_ccp32 = arith.extui(_i32(), sc_ccp16)
            sc_cbytes = _codepoint_to_utf8_bytes(sc_ccp32)
            scf.YieldOp([arith.addi(sc_cacc, sc_cbytes)])
        special_out_bytes = sc_count.results[0]
        special_new_off = arith.addi(out_off, special_out_bytes)

        copy_new_off = arith.addi(out_off, width)

        converted_off = arith.select(is_special, special_new_off, normal_new_off)
        new_out_off = arith.select(needs_convert, converted_off, copy_new_off)
        final_out_off = arith.select(should_skip, out_off, new_out_off)
        new_skip = arith.select(should_skip, skip_until, arith.addi(pos, width))

        if out_ptr is not None:
            _write_codepoint_utf8(scratch_normal, zero, case_cp_i32)

            sc_wr = scf.ForOp(zero, num_chars_i64, one, [zero])
            sc_wi = sc_wr.body.arguments[0]
            sc_wacc = sc_wr.body.arguments[1]
            with ir.InsertionPoint(sc_wr.body):
                sc_wcp16 = llvm.load(
                    i16_ty,
                    _gep_byte_offset(
                        chars_base, arith.muli(sc_wi, _const_i64(2)),
                    ),
                )
                sc_wcp32 = arith.extui(_i32(), sc_wcp16)
                sc_wb = _write_codepoint_utf8(scratch_special, sc_wacc, sc_wcp32)
                scf.YieldOp([arith.addi(sc_wacc, sc_wb)])

            orig_src = _gep_byte_offset(data_ptr, pos)
            conv_src = arith.select(is_special, scratch_special, scratch_normal)
            chosen_src = arith.select(needs_convert, conv_src, orig_src)
            chosen_len = arith.select(
                needs_convert,
                arith.select(is_special, special_out_bytes, normal_out_bytes),
                width,
            )
            safe_len = arith.select(should_skip, zero, chosen_len)
            llvm.intr_memcpy(
                _gep_byte_offset(out_ptr, out_off), chosen_src, safe_len, False,
            )

        scf.YieldOp([final_out_off, new_skip])

    return loop.results[0]


def lower_upper(gpu_module, str_view, flags_table_i64, cases_table_i64, special_table_i64):
    """mlir_string.upper() -> mlir_string using Unicode lookup tables.

    Table arguments are i64 SSA values holding device addresses.
    """
    data = view_extract_data(str_view)
    nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    case_flag = _const_i8(_FLAG_LOWER)

    flags_ptr = llvm.inttoptr(_ptr(), flags_table_i64)
    cases_ptr = llvm.inttoptr(_ptr(), cases_table_i64)
    special_ptr = llvm.inttoptr(_ptr(), special_table_i64)

    out_nbytes = _case_convert_char_loop(data, nb, flags_ptr, cases_ptr, special_ptr,
                                         case_flag, True, None)
    mi, out_data = allocate_mlir_string(gpu_module, out_nbytes)
    _case_convert_char_loop(data, nb, flags_ptr, cases_ptr, special_ptr,
                            case_flag, True, out_data)

    return build_mlir_string(mi, out_data, out_nbytes)


def lower_lower(gpu_module, str_view, flags_table_i64, cases_table_i64, special_table_i64):
    """mlir_string.lower() -> mlir_string using Unicode lookup tables.

    Table arguments are i64 SSA values holding device addresses.
    """
    data = view_extract_data(str_view)
    nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    case_flag = _const_i8(_FLAG_UPPER)

    flags_ptr = llvm.inttoptr(_ptr(), flags_table_i64)
    cases_ptr = llvm.inttoptr(_ptr(), cases_table_i64)
    special_ptr = llvm.inttoptr(_ptr(), special_table_i64)

    out_nbytes = _case_convert_char_loop(data, nb, flags_ptr, cases_ptr, special_ptr,
                                         case_flag, False, None)
    mi, out_data = allocate_mlir_string(gpu_module, out_nbytes)
    _case_convert_char_loop(data, nb, flags_ptr, cases_ptr, special_ptr,
                            case_flag, False, out_data)

    return build_mlir_string(mi, out_data, out_nbytes)


# ---------------------------------------------------------------------------
# Pure-MLIR read-only string operations
# ---------------------------------------------------------------------------

# Character type flag constants (from cudf::strings::string_character_types)
_CT_DECIMAL    = 1       # 1 << 0
_CT_NUMERIC    = 2       # 1 << 1
_CT_DIGIT      = 4       # 1 << 2
_CT_ALPHA      = 8       # 1 << 3
_CT_SPACE      = 16      # 1 << 4
_CT_UPPER      = 32      # 1 << 5
_CT_LOWER      = 64      # 1 << 6
_CT_ALPHANUM   = _CT_DECIMAL | _CT_NUMERIC | _CT_DIGIT | _CT_ALPHA  # 15
_CT_CASE_TYPES = _CT_UPPER | _CT_LOWER                              # 96
_CT_ALL_TYPES  = _CT_ALPHANUM | _CT_CASE_TYPES | _CT_SPACE          # 127


def _is_begin_utf8_char(byte_val):
    """Return i1: True if byte_val is the first byte of a UTF-8 char (not a continuation byte)."""
    masked = arith.andi(arith.extui(_i32(), byte_val), _const_i32(0xC0))
    return arith.cmpi(arith.CmpIPredicate.ne, masked, _const_i32(0x80))


def lower_len(str_view):
    """Pure MLIR: len(mlir_string) -> i32 character count.

    Counts the number of UTF-8 start bytes (bytes where top 2 bits != 10xxxxxx).
    """
    data = view_extract_data(str_view)
    nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    zero_i32 = _const_i32(0)
    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)

    loop = scf.ForOp(zero_i64, nb, one_i64, [zero_i32])
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc = loop.inner_iter_args[0]
        b = llvm.load(T.i8(), _gep_byte_offset(data, idx))
        is_start = _is_begin_utf8_char(b)
        inc = arith.select(is_start, _const_i32(1), zero_i32)
        new_acc = arith.addi(acc, inc)
        scf.yield_([new_acc])

    return loop.results[0]


def _all_characters_of_type(str_view, flags_table_ptr, types_i32, verify_types_i32):
    """Pure MLIR: all_characters_of_type -> i1.

    Mirrors cudf::strings::udf::all_characters_of_type from char_types.cuh.
    flags_table_ptr is a device pointer to the uint8 flags table.
    """
    data = view_extract_data(str_view)
    nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)
    zero_i32 = _const_i32(0)
    true_i1 = arith.constant(T.bool(), 1)
    false_i1 = arith.constant(T.bool(), 0)
    max_cp = _const_i32(0xFFFF)
    all_types_i32 = _const_i32(_CT_ALL_TYPES)

    is_empty = arith.cmpi(arith.CmpIPredicate.eq, nb, zero_i64)

    # Loop state: [byte_offset i64, check i1, check_count i32]
    loop = scf.ForOp(
        zero_i64, nb, one_i64,
        [zero_i64, true_i1, zero_i32],
    )
    with ir.InsertionPoint(loop.body):
        _loop_idx = loop.induction_variable
        cur_off = loop.inner_iter_args[0]
        cur_check = loop.inner_iter_args[1]
        cur_count = loop.inner_iter_args[2]

        at_boundary = arith.cmpi(arith.CmpIPredicate.eq, _loop_idx, cur_off)

        cp, char_width = _decode_utf8_to_codepoint(data, cur_off, nb)

        in_range = arith.cmpi(arith.CmpIPredicate.ule, cp, max_cp)
        cp_i64 = arith.extui(_i64(), cp)
        flag_ptr = _gep_byte_offset(flags_table_ptr, cp_i64)
        flag_raw = llvm.load(T.i8(), flag_ptr)
        flag = arith.extui(_i32(), flag_raw)
        flag = arith.select(in_range, flag, zero_i32)

        # (verify_types & flag) != 0
        verify_and = arith.andi(verify_types_i32, flag)
        should_verify_1 = arith.cmpi(arith.CmpIPredicate.ne, verify_and, zero_i32)
        # flag == 0 && verify_types == ALL_TYPES
        flag_is_zero = arith.cmpi(arith.CmpIPredicate.eq, flag, zero_i32)
        verify_is_all = arith.cmpi(arith.CmpIPredicate.eq, verify_types_i32, all_types_i32)
        should_verify_2 = arith.andi(flag_is_zero, verify_is_all)
        should_verify = arith.ori(should_verify_1, should_verify_2)

        # check = check & ((types & flag) > 0)  -- AND with running result
        types_and = arith.andi(types_i32, flag)
        type_match = arith.cmpi(arith.CmpIPredicate.ne, types_and, zero_i32)
        anded_check = arith.andi(cur_check, type_match)

        new_check = arith.select(should_verify, anded_check, cur_check)
        new_count = arith.select(
            should_verify,
            arith.addi(cur_count, _const_i32(1)),
            cur_count,
        )
        # Only update when at the start of a UTF-8 char
        final_check = arith.select(at_boundary, new_check, cur_check)
        final_count = arith.select(at_boundary, new_count, cur_count)
        next_off = arith.select(at_boundary, arith.addi(cur_off, char_width), cur_off)

        scf.yield_([next_off, final_check, final_count])

    result_check = loop.results[1]
    result_count = loop.results[2]
    has_chars = arith.cmpi(arith.CmpIPredicate.sgt, result_count, zero_i32)
    result = arith.andi(arith.andi(result_check, has_chars), arith.xori(is_empty, true_i1))
    return result


def lower_isalpha(str_view, flags_table_ptr):
    return _all_characters_of_type(str_view, flags_table_ptr, _const_i32(_CT_ALPHA), _const_i32(_CT_ALL_TYPES))

def lower_isalnum(str_view, flags_table_ptr):
    return _all_characters_of_type(str_view, flags_table_ptr, _const_i32(_CT_ALPHANUM), _const_i32(_CT_ALL_TYPES))

def lower_isdigit(str_view, flags_table_ptr):
    return _all_characters_of_type(str_view, flags_table_ptr, _const_i32(_CT_DIGIT), _const_i32(_CT_ALL_TYPES))

def lower_isdecimal(str_view, flags_table_ptr):
    return _all_characters_of_type(str_view, flags_table_ptr, _const_i32(_CT_DECIMAL), _const_i32(_CT_ALL_TYPES))

def lower_isupper(str_view, flags_table_ptr):
    return _all_characters_of_type(str_view, flags_table_ptr, _const_i32(_CT_UPPER), _const_i32(_CT_CASE_TYPES))

def lower_islower(str_view, flags_table_ptr):
    return _all_characters_of_type(str_view, flags_table_ptr, _const_i32(_CT_LOWER), _const_i32(_CT_CASE_TYPES))

def lower_isspace(str_view, flags_table_ptr):
    return _all_characters_of_type(str_view, flags_table_ptr, _const_i32(_CT_SPACE), _const_i32(_CT_ALL_TYPES))

def lower_isnumeric(str_view, flags_table_ptr):
    return _all_characters_of_type(str_view, flags_table_ptr, _const_i32(_CT_NUMERIC), _const_i32(_CT_ALL_TYPES))

def lower_istitle(str_view, flags_table_ptr):
    """Pure MLIR: istitle() -> i1. Mirrors cudf::strings::udf::is_title."""
    data = view_extract_data(str_view)
    nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)
    zero_i32 = _const_i32(0)
    true_i1 = arith.constant(T.bool(), 1)
    false_i1 = arith.constant(T.bool(), 0)
    max_cp = _const_i32(0xFFFF)
    upper_mask = _const_i32(_CT_UPPER)
    lower_mask = _const_i32(_CT_LOWER)
    case_mask = _const_i32(_CT_CASE_TYPES)

    # Loop state: [byte_offset, valid (found cased char), should_be_cap, ok]
    loop = scf.ForOp(
        zero_i64, nb, one_i64,
        [zero_i64, false_i1, true_i1, true_i1],
    )
    with ir.InsertionPoint(loop.body):
        _loop_idx = loop.induction_variable
        cur_off = loop.inner_iter_args[0]
        cur_valid = loop.inner_iter_args[1]
        cur_should_cap = loop.inner_iter_args[2]
        cur_ok = loop.inner_iter_args[3]

        at_boundary = arith.cmpi(arith.CmpIPredicate.eq, _loop_idx, cur_off)

        cp, char_width = _decode_utf8_to_codepoint(data, cur_off, nb)

        in_range = arith.cmpi(arith.CmpIPredicate.ule, cp, max_cp)
        cp_i64 = arith.extui(_i64(), cp)
        flag_ptr = _gep_byte_offset(flags_table_ptr, cp_i64)
        flag_raw = llvm.load(T.i8(), flag_ptr)
        flag = arith.extui(_i32(), flag_raw)
        flag = arith.select(in_range, flag, zero_i32)

        is_upper = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(flag, upper_mask), zero_i32)
        is_lower = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(flag, lower_mask), zero_i32)
        is_cased = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(flag, case_mask), zero_i32)

        # If cased: should_be_cap XOR is_upper must be false (they must match)
        wrong_case = arith.xori(cur_should_cap, is_upper)
        check_fail = arith.andi(is_cased, wrong_case)
        new_ok = arith.select(check_fail, false_i1, cur_ok)
        new_valid = arith.select(is_cased, true_i1, cur_valid)
        # After a cased char, next should not be capitalized; after non-cased, next should be
        new_should_cap = arith.select(is_cased, false_i1, true_i1)

        final_ok = arith.select(at_boundary, new_ok, cur_ok)
        final_valid = arith.select(at_boundary, new_valid, cur_valid)
        final_should_cap = arith.select(at_boundary, new_should_cap, cur_should_cap)
        next_off = arith.select(at_boundary, arith.addi(cur_off, char_width), cur_off)

        scf.yield_([next_off, final_valid, final_should_cap, final_ok])

    return arith.andi(loop.results[1], loop.results[3])


def lower_compare(lhs, rhs):
    """Pure MLIR: mlir_string.compare(other) -> i32.

    Byte-by-byte comparison matching cudf::string_view::compare.
    """
    lhs_data = view_extract_data(lhs)
    rhs_data = view_extract_data(rhs)
    lhs_nb = _zext_i32_to_i64(view_extract_nbytes(lhs))
    rhs_nb = _zext_i32_to_i64(view_extract_nbytes(rhs))

    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)
    zero_i32 = _const_i32(0)

    min_len = arith.select(
        arith.cmpi(arith.CmpIPredicate.slt, lhs_nb, rhs_nb), lhs_nb, rhs_nb
    )

    # Loop over min(len1, len2) bytes, carry a diff (i32) and a found flag (i1)
    loop = scf.ForOp(
        zero_i64, min_len, one_i64,
        [zero_i32, arith.constant(T.bool(), 0)],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc_diff = loop.inner_iter_args[0]
        acc_found = loop.inner_iter_args[1]

        b1 = arith.extui(_i32(), llvm.load(T.i8(), _gep_byte_offset(lhs_data, idx)))
        b2 = arith.extui(_i32(), llvm.load(T.i8(), _gep_byte_offset(rhs_data, idx)))
        diff = arith.subi(b1, b2)
        ne = arith.cmpi(arith.CmpIPredicate.ne, diff, zero_i32)
        first_diff = arith.andi(ne, arith.xori(acc_found, arith.constant(T.bool(), 1)))
        new_diff = arith.select(first_diff, diff, acc_diff)
        new_found = arith.ori(acc_found, ne)
        scf.yield_([new_diff, new_found])

    byte_diff = loop.results[0]
    found = loop.results[1]

    # If no byte difference found, compare lengths
    lhs_longer = arith.cmpi(arith.CmpIPredicate.sgt, lhs_nb, rhs_nb)
    rhs_longer = arith.cmpi(arith.CmpIPredicate.slt, lhs_nb, rhs_nb)
    len_cmp = arith.select(lhs_longer, _const_i32(1),
                           arith.select(rhs_longer, _const_i32(-1), zero_i32))
    return arith.select(found, byte_diff, len_cmp)


def lower_find(str_view, target):
    """Pure MLIR: mlir_string.find(target) -> i32 character position or -1.

    Simple byte-level search with character position tracking.
    """
    src_data = view_extract_data(str_view)
    src_nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    tgt_data = view_extract_data(target)
    tgt_nb = _zext_i32_to_i64(view_extract_nbytes(target))

    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)
    zero_i32 = _const_i32(0)
    neg_one_i32 = _const_i32(-1)

    tgt_empty = arith.cmpi(arith.CmpIPredicate.eq, tgt_nb, zero_i64)

    # search_len = src_nb - tgt_nb + 1 (clamped to 0 if target longer)
    diff = arith.subi(src_nb, tgt_nb)
    search_len = arith.addi(diff, one_i64)
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, tgt_nb, src_nb)
    search_len = arith.select(too_long, zero_i64, search_len)

    # Loop: [result_pos i32, char_pos i32, found i1]
    loop = scf.ForOp(
        zero_i64, search_len, one_i64,
        [neg_one_i32, zero_i32, arith.constant(T.bool(), 0)],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc_result = loop.inner_iter_args[0]
        acc_char_pos = loop.inner_iter_args[1]
        acc_found = loop.inner_iter_args[2]

        # Compare tgt_nb bytes at src_data+idx with tgt_data
        cmp_loop = scf.ForOp(
            zero_i64, tgt_nb, one_i64,
            [arith.constant(T.bool(), 1)],
        )
        with ir.InsertionPoint(cmp_loop.body):
            cidx = cmp_loop.induction_variable
            cmatch = cmp_loop.inner_iter_args[0]
            sb = llvm.load(T.i8(), _gep_byte_offset(src_data, arith.addi(idx, cidx)))
            tb = llvm.load(T.i8(), _gep_byte_offset(tgt_data, cidx))
            eq = arith.cmpi(arith.CmpIPredicate.eq, sb, tb)
            scf.yield_([arith.andi(cmatch, eq)])

        matched = cmp_loop.results[0]
        # Matched at this position, and target is empty also matches at pos 0
        first_match = arith.andi(matched, arith.xori(acc_found, arith.constant(T.bool(), 1)))
        new_result = arith.select(first_match, acc_char_pos, acc_result)
        new_found = arith.ori(acc_found, matched)

        # Track character position
        byte_val = llvm.load(T.i8(), _gep_byte_offset(src_data, idx))
        is_start = _is_begin_utf8_char(byte_val)
        new_char_pos = arith.addi(acc_char_pos, arith.select(is_start, _const_i32(1), zero_i32))

        scf.yield_([new_result, new_char_pos, new_found])

    result = loop.results[0]
    # Empty target: always found at position 0
    return arith.select(tgt_empty, zero_i32, result)


def lower_count(str_view, target):
    """Pure MLIR: mlir_string.count(target) -> i32 non-overlapping occurrence count."""
    src_data = view_extract_data(str_view)
    src_nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    tgt_data = view_extract_data(target)
    tgt_nb = _zext_i32_to_i64(view_extract_nbytes(target))

    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)
    zero_i32 = _const_i32(0)

    tgt_empty = arith.cmpi(arith.CmpIPredicate.eq, tgt_nb, zero_i64)
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, tgt_nb, src_nb)

    # Loop: [byte_pos i64, count i32]
    loop = scf.ForOp(
        zero_i64, src_nb, one_i64,
        [zero_i64, zero_i32],
    )
    with ir.InsertionPoint(loop.body):
        _loop_idx = loop.induction_variable
        cur_pos = loop.inner_iter_args[0]
        cur_count = loop.inner_iter_args[1]

        at_pos = arith.cmpi(arith.CmpIPredicate.eq, _loop_idx, cur_pos)
        remaining = arith.subi(src_nb, cur_pos)
        enough = arith.cmpi(arith.CmpIPredicate.sge, remaining, tgt_nb)
        should_check = arith.andi(at_pos, enough)

        # Compare tgt_nb bytes
        cmp_loop = scf.ForOp(
            zero_i64, tgt_nb, one_i64,
            [arith.constant(T.bool(), 1)],
        )
        with ir.InsertionPoint(cmp_loop.body):
            cidx = cmp_loop.induction_variable
            cmatch = cmp_loop.inner_iter_args[0]
            sb = llvm.load(T.i8(), _gep_byte_offset(src_data, arith.addi(cur_pos, cidx)))
            tb = llvm.load(T.i8(), _gep_byte_offset(tgt_data, cidx))
            eq = arith.cmpi(arith.CmpIPredicate.eq, sb, tb)
            scf.yield_([arith.andi(cmatch, eq)])

        matched = arith.andi(should_check, cmp_loop.results[0])
        new_count = arith.select(matched, arith.addi(cur_count, _const_i32(1)), cur_count)
        # Skip past match (advance by tgt_nb), else advance by 1
        next_pos = arith.select(matched, arith.addi(cur_pos, tgt_nb), arith.addi(cur_pos, one_i64))
        # If not at boundary, don't advance
        final_pos = arith.select(at_pos, next_pos, cur_pos)
        final_count = arith.select(at_pos, new_count, cur_count)

        scf.yield_([final_pos, final_count])

    # Empty target: count = len(src) + 1 (Python convention)
    src_len_plus_one = arith.addi(lower_len(str_view), _const_i32(1))
    result = arith.select(tgt_empty, src_len_plus_one,
                          arith.select(too_long, zero_i32, loop.results[1]))
    return result


def lower_startswith(str_view, prefix):
    """Pure MLIR: mlir_string.startswith(prefix) -> i1."""
    src_data = view_extract_data(str_view)
    src_nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    pfx_data = view_extract_data(prefix)
    pfx_nb = _zext_i32_to_i64(view_extract_nbytes(prefix))

    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)

    too_long = arith.cmpi(arith.CmpIPredicate.sgt, pfx_nb, src_nb)

    loop = scf.ForOp(zero_i64, pfx_nb, one_i64, [arith.constant(T.bool(), 1)])
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc = loop.inner_iter_args[0]
        sb = llvm.load(T.i8(), _gep_byte_offset(src_data, idx))
        pb = llvm.load(T.i8(), _gep_byte_offset(pfx_data, idx))
        eq = arith.cmpi(arith.CmpIPredicate.eq, sb, pb)
        scf.yield_([arith.andi(acc, eq)])

    return arith.andi(arith.xori(too_long, arith.constant(T.bool(), 1)), loop.results[0])


def lower_endswith(str_view, suffix):
    """Pure MLIR: mlir_string.endswith(suffix) -> i1."""
    src_data = view_extract_data(str_view)
    src_nb = _zext_i32_to_i64(view_extract_nbytes(str_view))
    sfx_data = view_extract_data(suffix)
    sfx_nb = _zext_i32_to_i64(view_extract_nbytes(suffix))

    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)

    too_long = arith.cmpi(arith.CmpIPredicate.sgt, sfx_nb, src_nb)
    offset = arith.subi(src_nb, sfx_nb)

    loop = scf.ForOp(zero_i64, sfx_nb, one_i64, [arith.constant(T.bool(), 1)])
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc = loop.inner_iter_args[0]
        sb = llvm.load(T.i8(), _gep_byte_offset(src_data, arith.addi(offset, idx)))
        xb = llvm.load(T.i8(), _gep_byte_offset(sfx_data, idx))
        eq = arith.cmpi(arith.CmpIPredicate.eq, sb, xb)
        scf.yield_([arith.andi(acc, eq)])

    return arith.andi(arith.xori(too_long, arith.constant(T.bool(), 1)), loop.results[0])


def lower_contains(str_view, target):
    """Pure MLIR: target in mlir_string -> i1."""
    result = lower_find(str_view, target)
    return arith.cmpi(arith.CmpIPredicate.sge, result, _const_i32(0))


# ---------------------------------------------------------------------------
# Lowering registry: intrinsic attribute access and constructor
# ---------------------------------------------------------------------------

from numba_cuda_mlir.lowering_registry import LoweringRegistry
from cudf.core.udf.mlir_backend.string_types import MLIRStringType

registry = LoweringRegistry()

_mlir_string_inst = MLIRStringType()


@registry.lower_getattr(_mlir_string_inst, "nbytes")
def _lower_ms_nbytes(context, builder, target, value, attr=None):
    ms_val = builder.load_var(value)
    builder.store_var(target, ms_extract_nbytes(ms_val))


@registry.lower_getattr(_mlir_string_inst, "data_ptr")
def _lower_ms_data_ptr(context, builder, target, value, attr=None):
    ms_val = builder.load_var(value)
    ptr = ms_extract_data(ms_val)
    builder.store_var(target, llvm.ptrtoint(T.i64(), ptr))


def _register_mlir_string_from_ptr():
    from cudf.core.udf.mlir_backend.string_typing_impl import mlir_string_from_ptr as _stub
    from numba_cuda_mlir.numba_cuda import types as nb_types

    @registry.lower(_stub, nb_types.int64, nb_types.int64)
    def _lower_from_ptr(builder, target, args, kwargs):
        data_i64 = builder.load_var(args[0])
        nbytes_i64 = builder.load_var(args[1])
        data_ptr = llvm.inttoptr(_ptr(), data_i64)
        null_ptr = llvm.ZeroOp(_ptr())
        result = build_mlir_string(null_ptr, data_ptr, nbytes_i64)
        builder.store_var(target, result)


_register_mlir_string_from_ptr()


# ---------------------------------------------------------------------------
# getitem core logic (shared by numba_cuda_mlir and Masked lowerings)
# ---------------------------------------------------------------------------

def string_getitem_int_core(gpu_module, data, nbytes_i64, idx_i64):
    """Core logic for string[int]. Returns a mlir_string (meminfo, ptr, len).

    data:       !llvm.ptr to the source bytes
    nbytes_i64: i64 byte count of the source string
    idx_i64:    i64 index (may be negative)
    """
    zero = _const_i64(0)
    is_neg = arith.cmpi(arith.CmpIPredicate.slt, idx_i64, zero)
    adjusted = arith.addi(idx_i64, nbytes_i64)
    idx_i64 = arith.select(is_neg, adjusted, idx_i64)

    one = _const_i64(1)
    mi, out_data = allocate_mlir_string(gpu_module, one)
    src = _gep_byte_offset(data, idx_i64)
    llvm.intr_memcpy(out_data, src, one, False)
    return build_mlir_string(mi, out_data, one)


def string_getitem_slice_core(gpu_module, data, nbytes_i64, slice_val):
    """Core logic for string[a:b]. Returns a mlir_string (meminfo, ptr, len).

    data:       !llvm.ptr to the source bytes
    nbytes_i64: i64 byte count of the source string
    slice_val:  Slice object with .start (index), .stop (index|None)
    """
    zero = _const_i64(0)

    start_idx = arith.index_cast(T.i64(), slice_val.start)
    if slice_val.stop is None:
        stop_idx = nbytes_i64
    else:
        stop_idx = arith.index_cast(T.i64(), slice_val.stop)

    start_neg = arith.cmpi(arith.CmpIPredicate.slt, start_idx, zero)
    start_idx = arith.select(start_neg, arith.addi(start_idx, nbytes_i64), start_idx)
    stop_neg = arith.cmpi(arith.CmpIPredicate.slt, stop_idx, zero)
    stop_idx = arith.select(stop_neg, arith.addi(stop_idx, nbytes_i64), stop_idx)

    start_clamped = arith.maxsi(zero, arith.minsi(start_idx, nbytes_i64))
    stop_clamped = arith.maxsi(start_clamped, arith.minsi(stop_idx, nbytes_i64))
    result_len = arith.subi(stop_clamped, start_clamped)

    mi, out_data = allocate_mlir_string(gpu_module, result_len)
    src = _gep_byte_offset(data, start_clamped)
    llvm.intr_memcpy(out_data, src, result_len, False)
    return build_mlir_string(mi, out_data, result_len)


# ---------------------------------------------------------------------------
# getitem: mlir_string[int] -> single-char mlir_string
# ---------------------------------------------------------------------------

def _register_string_int_getitem():
    from numba_cuda_mlir.numba_cuda import types as nb_types
    import operator as _op

    for int_ty in (nb_types.int32, nb_types.int64, nb_types.intp):
        def _make(int_ty):
            @registry.lower(_op.getitem, _mlir_string_inst, int_ty)
            def _lower_getitem_int(builder, target, args, kwargs):
                ms_val = builder.load_var(args[0])
                idx_val = builder.load_var(args[1])
                data = ms_extract_data(ms_val)
                nbytes = ms_extract_nbytes(ms_val)
                idx_i64 = builder.mlir_convert(idx_val, T.i64())
                result = string_getitem_int_core(
                    builder.mlir_gpu_module, data, nbytes, idx_i64
                )
                builder.store_var(target, result)
        _make(int_ty)


_register_string_int_getitem()


# ---------------------------------------------------------------------------
# getitem: mlir_string[a:b] -> substring mlir_string
# ---------------------------------------------------------------------------

def _register_string_slice_getitem():
    from numba_cuda_mlir.numba_cuda import types as nb_types
    import operator as _op

    @registry.lower(_op.getitem, _mlir_string_inst, nb_types.slice2_type)
    def _lower_getitem_slice(builder, target, args, kwargs):
        ms_val = builder.load_var(args[0])
        slice_val = builder.load_var(args[1])
        data = ms_extract_data(ms_val)
        nbytes = ms_extract_nbytes(ms_val)
        result = string_getitem_slice_core(
            builder.mlir_gpu_module, data, nbytes, slice_val
        )
        builder.store_var(target, result)


_register_string_slice_getitem()
