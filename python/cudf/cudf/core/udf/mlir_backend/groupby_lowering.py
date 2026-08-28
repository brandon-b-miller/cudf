# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
Lowerings for the groupby _group_slot helper.

All GroupType / Block* shim lowerings have been removed — cooperative
reductions are now handled by numba_cuda_mlir's MLIR cooperative module.
"""

from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.numba_cuda import types

from cudf.core.udf.mlir_backend.groupby_typing import _group_slot

lower = lowering_registry.lower


def _lower_group_slot_impl(builder, target, group_type):
    """Allocate record_size bytes and store the pointer in target."""
    record_size = group_type.size
    i8 = builder.get_mlir_type(types.uint8)
    ptr = builder.alloca(i8, count=record_size)
    builder.store_var(target, ptr)


def register_group_slot_lowering(dataframe_group_type):
    """Register lowering for _group_slot() for this kernel's group type."""
    lower(_group_slot, *())(
        lambda builder, target, args, kwargs: _lower_group_slot_impl(
            builder, target, dataframe_group_type
        )
    )
