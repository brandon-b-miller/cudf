# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cudf.core.udf.mlir_backend import strings_lowering as _impl

cast_string_view_to_managed_udf_string = getattr(
    _impl, "cast_string_view_to_managed_udf_string", None
)

__all__ = ["cast_string_view_to_managed_udf_string"]
