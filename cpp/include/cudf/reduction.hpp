/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/cudf.h>
#include <cudf/scalar/scalar.hpp>
#include <cudf/aggregation.hpp>

namespace cudf {
namespace experimental {

// @brief Enum to describe scan operation type
enum class scan_type : bool {
   INCLUSIVE, 
   EXCLUSIVE
};

// @brief Enum to describe include nulls or exclude nulls in an aggregation
enum class include_nulls : bool {
   YES, 
   NO
};

/** --------------------------------------------------------------------------*
 * @brief  Computes the reduction of the values in all rows of a column.
 * This function does not detect overflows in reductions.
 * Using a higher precision `data_type` may prevent overflow.
 * Only `min` and `max` ops are supported for reduction of non-arithmetic
 * types (timestamp, string...).
 * The null values are skipped for the operation.
 * If the column is empty, the member `is_valid()` of the output scalar
 * will contain `false`.
 *
 * @throws `cudf::logic_error` if reduction is called for non-arithmetic output
 * type and operator other than `min` and `max`.
 * @throws `cudf::logic_error` if input column data type is not convertible to
 * output data type.
 * If the input column has arithmetic type, output_dtype can be any arithmetic
 * type. For `mean`, `var` and `std` ops, a floating point output type must be 
 * specified. If the input column has non-arithmetic type
 *   eg.(timestamp, string...), the same type must be specified.
 *
 * @param[in] col Input column view
 * @param[in] agg unique_ptr of the aggregation operator applied by the reduction
 * @param[in] output_dtype  The computation and output precision.
 * @params[in] mr The resource to use for all allocations
 * @returns  cudf::scalar the result value
 * If the reduction fails, the member is_valid of the output scalar
 * will contain `false`.
 * ----------------------------------------------------------------------------**/
std::unique_ptr<scalar> reduce(
    const column_view& col, 
    std::unique_ptr<aggregation> const &agg,
    data_type output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** --------------------------------------------------------------------------*
 * @brief  Computes the scan of a column.
 * The null values are skipped for the operation, and if an input element
 * at `i` is null, then the output element at `i` will also be null.
 *
 * @throws `cudf::logic_error` if column datatype is not numeric type.
 *
 * @param[in] input The input column view for the scan
 * @param[in] agg unique_ptr to aggregation operator applied by the scan
 * @param[in] inclusive The flag for applying an inclusive scan if
 *            scan_type::INCLUSIVE, an exclusive scan if scan_type::EXCLUSIVE.
 * @param[in] include_nulls_flag Exclude null values when computing the result if
 * include_nulls::NO. Include nulls if include_nulls::YES.
 * Any operation with a null results in a null.
 * @params[in] mr The resource to use for all allocations
 * @returns unique pointer to new output column
 * ----------------------------------------------------------------------------**/
std::unique_ptr<column>
scan(const column_view &input, std::unique_ptr<aggregation> const &agg,
     scan_type inclusive, include_nulls include_nulls_flag = include_nulls::NO,
     rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf
