# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# strings_udf.pyx and cudf_strings_udf have been removed --
# Unicode tables are now embedded in the cuDF MLIR backend
# (cudf.core.udf.mlir_backend.unicode), and all string/groupby device
# operations are pure MLIR via numba-cuda-mlir.
