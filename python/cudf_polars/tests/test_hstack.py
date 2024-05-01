# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_hstack():
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()

    query = ldf.with_columns(pl.col("a") + pl.col("b"))
    assert_gpu_result_equal(query)
