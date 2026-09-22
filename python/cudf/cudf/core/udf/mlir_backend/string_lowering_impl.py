# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-MLIR building blocks for ``mlir_string`` operations.

These helpers operate on raw MLIR SSA values (no cuDF dependency) and are the
low-level primitives the string lowerings in ``strings_lowering`` compose. They
are intentionally private module-level functions.

``mlir_string`` layout: ``{ptr meminfo, ptr data, i64 nbytes}``.
"""

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm, scf
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.lowering_utilities import GEP_DYNAMIC_INDEX


def _i32() -> ir.Type:
    return ir.IntegerType.get_signless(32)


def _ptr() -> ir.Type:
    return llvm.PointerType.get()


def _const_i32(val: int) -> ir.Value:
    return arith.constant(_i32(), val)


def _const_i64(val: int) -> ir.Value:
    return arith.constant(T.i64(), val)


def _zext_i32_to_i64(val: ir.Value) -> ir.Value:
    return llvm.zext(T.i64(), val)


def _gep_byte_offset(base: ir.Value, offset: ir.Value) -> ir.Value:
    """``base + offset`` bytes (GEP on an ``i8*``)."""
    return llvm.getelementptr(
        _ptr(), base, [offset], [GEP_DYNAMIC_INDEX], T.i8(), None
    )


# --- mlir_string field accessors: {ptr meminfo, ptr data, i64 nbytes} ---
def _ms_extract_data(ms_val: ir.Value) -> ir.Value:
    return llvm.extractvalue(_ptr(), ms_val, [1])


def _ms_extract_nbytes(ms_val: ir.Value) -> ir.Value:
    return llvm.extractvalue(T.i64(), ms_val, [2])


def _is_begin_utf8_char(byte_val: ir.Value) -> ir.Value:
    """i1: whether ``byte_val`` is a UTF-8 start byte (top 2 bits != ``10``)."""
    masked = arith.andi(arith.extui(_i32(), byte_val), _const_i32(0xC0))
    return arith.cmpi(arith.CmpIPredicate.ne, masked, _const_i32(0x80))


def _lower_len(ms_val: ir.Value) -> ir.Value:
    """``len(mlir_string) -> i32`` character count (pure MLIR).

    Counts UTF-8 start bytes over the ``nbytes``-long data buffer, so the result
    is the number of characters (code points), matching cuDF ``str.len``.
    """
    data = _ms_extract_data(ms_val)
    nb = _ms_extract_nbytes(ms_val)
    zero_i32 = _const_i32(0)
    zero_i64 = _const_i64(0)
    one_i64 = _const_i64(1)

    loop = scf.ForOp(zero_i64, nb, one_i64, [zero_i32])
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc = loop.inner_iter_args[0]
        b = llvm.load(T.i8(), _gep_byte_offset(data, idx))
        inc = arith.select(_is_begin_utf8_char(b), _const_i32(1), zero_i32)
        scf.yield_([arith.addi(acc, inc)])

    return loop.results[0]
