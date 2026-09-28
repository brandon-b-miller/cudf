# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-MLIR building blocks composed by the ``mlir_string`` lowerings."""

from __future__ import annotations

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm, scf
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._threading import _LockedCounter
from numba_cuda_mlir.lowering_utilities import GEP_DYNAMIC_INDEX
from numba_cuda_mlir.mlir.dialect_exts import llvm as _llvm_ext

# Monotonic counter for uniquely naming emitted literal globals.
_literal_counter = _LockedCounter(1)


def _ms_data_nbytes(ms_val: ir.Value) -> tuple[ir.Value, ir.Value]:
    """``(data_ptr, nbytes)`` for an ``mlir_string`` SSA value.

    ``mlir_string`` layout is ``{ptr meminfo, ptr data, i64 nbytes}``; this
    reads field 1 (the raw UTF-8 buffer) and field 2 (its byte length).
    """
    data = llvm.extractvalue(llvm.PointerType.get(), ms_val, [1])
    nbytes = llvm.extractvalue(T.i64(), ms_val, [2])
    return data, nbytes


def _byte_at(data: ir.Value, idx: ir.Value) -> ir.Value:
    """Load ``data[idx]`` as ``i8`` via a dynamic GEP on the ``i8*`` buffer."""
    byte_ptr = llvm.getelementptr(
        llvm.PointerType.get(), data, [idx], [GEP_DYNAMIC_INDEX], T.i8(), None
    )
    return llvm.load(T.i8(), byte_ptr)


def _materialize_utf8_literal(
    gpu_module: ir.Module, pystr: str
) -> tuple[ir.Value, ir.Value]:
    """Emit an internal global holding ``pystr``'s UTF-8 bytes.

    ``mlir_string`` stores raw UTF-8, so a string-literal operand (e.g.
    ``s == "foo"``) is encoded to UTF-8 and placed in a constant device global.

    Parameters
    ----------
    gpu_module : ir.Module
        The MLIR GPU module to emit the global into.
    pystr : str
        The compile-time literal value.

    Returns
    -------
    tuple of ir.Value
        ``(data_ptr, nbytes)`` where ``nbytes`` is an ``i64`` constant.
    """
    data = pystr.encode("utf-8")
    n = len(data)
    name = f"__cudf_udf_str_{next(_literal_counter)}"
    arr_type = ir.Type.parse(f"!llvm.array<{max(n, 1)} x i8>")
    block = gpu_module.bodyRegion.blocks[0]
    with ir.InsertionPoint.at_block_begin(block):
        linkage = ir.Attribute.parse("#llvm.linkage<internal>")
        payload = data if n else b"\x00"
        _llvm_ext.GlobalOp(
            arr_type,
            name,
            linkage,
            addr_space=0,
            constant=True,
            value=ir.StringAttr.get(payload.decode("latin-1")),
        )
    return _llvm_ext.addressof(name), arith.constant(T.i64(), n)


def _is_begin_utf8_char(byte_val: ir.Value) -> ir.Value:
    """i1: whether ``byte_val`` is a UTF-8 start byte (top 2 bits != ``10``)."""
    i32 = ir.IntegerType.get_signless(32)
    masked = arith.andi(arith.extui(i32, byte_val), arith.constant(i32, 0xC0))
    return arith.cmpi(
        arith.CmpIPredicate.ne, masked, arith.constant(i32, 0x80)
    )


def _lower_len(ms_val: ir.Value) -> ir.Value:
    """``len(mlir_string) -> i64`` character count (pure MLIR).

    Counts UTF-8 start bytes over the ``nbytes``-long data buffer, so the result
    is the number of characters (code points), matching cuDF ``str.len``. The
    count is accumulated directly in ``i64`` to match the declared result type.
    """
    data, nb = _ms_data_nbytes(ms_val)
    zero_i64 = arith.constant(T.i64(), 0)
    one_i64 = arith.constant(T.i64(), 1)

    loop = scf.ForOp(zero_i64, nb, one_i64, [zero_i64])
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc = loop.inner_iter_args[0]
        inc = arith.select(
            _is_begin_utf8_char(_byte_at(data, idx)), one_i64, zero_i64
        )
        scf.yield_([arith.addi(acc, inc)])

    return loop.results[0]


def _lower_bytes_compare(
    a_data: ir.Value,
    a_nbytes: ir.Value,
    b_data: ir.Value,
    b_nbytes: ir.Value,
) -> ir.Value:
    """Lexicographic unsigned byte comparison -> ``i32`` in ``{-1, 0, 1}``.

    Mirrors ``memcmp`` over the shared prefix followed by a length tiebreak,
    matching libcudf ``string_view`` ordering (a proper prefix sorts first).
    """
    i32 = ir.IntegerType.get_signless(32)
    i1 = ir.IntegerType.get_signless(1)
    neg1 = arith.constant(i32, -1)
    pos1 = arith.constant(i32, 1)
    zero32 = arith.constant(i32, 0)
    zero_i64 = arith.constant(T.i64(), 0)
    one_i64 = arith.constant(T.i64(), 1)
    true_i1 = arith.constant(i1, 1)

    a_shorter = arith.cmpi(arith.CmpIPredicate.ult, a_nbytes, b_nbytes)
    min_len = arith.select(a_shorter, a_nbytes, b_nbytes)

    # Carry (found_mismatch, prefix_cmp) across the shared prefix.
    loop = scf.ForOp(
        zero_i64, min_len, one_i64, [arith.constant(i1, 0), zero32]
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        found = loop.inner_iter_args[0]
        cmp = loop.inner_iter_args[1]
        av = arith.extui(i32, _byte_at(a_data, idx))
        bv = arith.extui(i32, _byte_at(b_data, idx))
        differ = arith.cmpi(arith.CmpIPredicate.ne, av, bv)
        set_now = arith.andi(differ, arith.xori(found, true_i1))
        this_cmp = arith.select(
            arith.cmpi(arith.CmpIPredicate.ult, av, bv), neg1, pos1
        )
        scf.yield_(
            [
                arith.ori(found, differ),
                arith.select(set_now, this_cmp, cmp),
            ]
        )

    found = loop.results[0]
    prefix_cmp = loop.results[1]
    len_lt = arith.cmpi(arith.CmpIPredicate.ult, a_nbytes, b_nbytes)
    len_gt = arith.cmpi(arith.CmpIPredicate.ugt, a_nbytes, b_nbytes)
    len_cmp = arith.select(len_lt, neg1, arith.select(len_gt, pos1, zero32))
    return arith.select(found, prefix_cmp, len_cmp)
