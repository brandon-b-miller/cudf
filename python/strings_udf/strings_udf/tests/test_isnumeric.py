# Copyright (c) 2022, NVIDIA CORPORATION.

from .utils import run_udf_test
import pytest

@pytest.mark.parametrize("data", [["1", "12", "123abc", "2.1", "", "0003"]])
def test_string_udf_isnumeric(data):
    # tests the `isnumeric` function in string udfs

    def func(st):
        return st.isnumeric()

    run_udf_test(data, func, 'bool')
