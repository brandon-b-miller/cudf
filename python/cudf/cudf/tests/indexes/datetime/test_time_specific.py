# Copyright (c) 2022-2023, NVIDIA CORPORATION.
import pandas as pd

import cudf
from cudf.testing._utils import assert_eq


def test_tz_localize():
    pidx = pd.date_range("2001-01-01", "2001-01-02", freq="1s")
    pidx = pidx.astype("<M8[ns]")
    idx = cudf.from_pandas(pidx)
    assert pidx.dtype == idx.dtype
    assert_eq(
        pidx.tz_localize("America/New_York"),
        idx.tz_localize("America/New_York"),
    )
