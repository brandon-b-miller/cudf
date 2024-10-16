/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <memory>

namespace cudf {
namespace experimental {

/**---------------------------------------------------------------------------*
 * @brief Fills a range of elements in-place in a column with a scalar value.
 * 
 * Fills N elements of @p destination starting at @p begin with @p value, where
 * N = (@p end - @p begin).
 *
 * Overwrites the range of elements in @p destination indicated by the indices
 * [@p begin, @p end) with @p value. Use the out-of-place fill function
 * returning std::unique_ptr<column> for use cases requiring memory
 * reallocation.
 *
 * @throws `cudf::logic_error` if memory reallocation is required (e.g. for
 * variable width types).
 * @throws `cudf::logic_error` for invalid range (if @p begin < 0,
 * @p begin > @p end, @p begin >= @p destination.size(), or
 * @p end > @p destination.size()).
 * @throws `cudf::logic_error` if @p destination and @p value have different
 * types.
 * @throws `cudf::logic_error` if @p value is invalid but @p destination is not
 * nullable.
 *
 * @param destination The preallocated column to fill into
 * @param begin The starting index of the fill range (inclusive)
 * @param end The index of the last element in the fill range (exclusive)
 * @param value The scalar value to fill
 * @return void
 *---------------------------------------------------------------------------**/
void fill(mutable_column_view& destination, size_type begin, size_type end,
          scalar const& value);

/**---------------------------------------------------------------------------*
 * @brief Fills a range of elements in a column out-of-place with a scalar
 * value.
 * 
 * Creates a new column as-if an in-place fill was performed into @p input;
 * i.e. it is as if a copy of @p input was created first and then the elements
 * indicated by the indices [@p begin, @p end) were overwritten by @p value.
 *
 * @throws `cudf::logic_error` for invalid range (if @p begin < 0,
 * @p begin > @p end, @p begin >= @p destination.size(), or
 * @p end > @p destination.size()).
 * @throws `cudf::logic_error` if @p destination and @p value have different
 * types.
 *
 * @param input The input column used to create a new column. The new column
 * is created by replacing the values of @p input in the specified range with
 * @p value.
 * @param begin The starting index of the fill range (inclusive)
 * @param end The index of the last element in the fill range (exclusive)
 * @param value The scalar value to fill
 * @param mr Memory resource to allocate the result output column
 * @return std::unique_ptr<column> The result output column
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> fill(
    column_view const& input, size_type begin, size_type end,
    scalar const& value,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Repeat rows of a Table.
 * 
 * Creates a new table by repeating the rows of @p input_table. The number of 
 * repetitions of each element is defined by the value at the corresponding 
 * index of @p count
 * Example:
 * ```
 * in = [4,5,6]
 * count = [1,2,3]
 * return = [4,5,5,6,6,6]
 * ```
 * @p count should not have null values; should not contain negative values;
 * and the sum of count elements should not overflow the size_type's limit.
 * It is undefined behavior if @p count has negative values or the sum overflows
 * and @p check_count is set to false.
 *
 * @throws `cudf::logic_error` if the data type of @p count is not size_type.
 * @throws `cudf::logic_error` if @p input_table and @p count have different
 * number of rows.
 * @throws `cudf::logic_error` if @p count has null values.
 * @throws `cudf::logic_error` if @p check_count is set to true and @p count
 * has negative values or the sum of @p count elements overflows.
 *
 * @param input_table Input table
 * @param count Non-nullable column of a integral type
 * @param check_count Whether to check count (negative values and overflow)
 * @param mr Memory resource to allocate the result output table
 * @return std::unique_ptr<table> The result table containing the repetitions
 *---------------------------------------------------------------------------**/
std::unique_ptr<table> repeat(
    table_view const& input_table, column_view const& count,
    bool check_count = false,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Repeat rows of a Table.
 * 
 * Creates a new table by repeating @p count times the rows of @p input_table.
 * Example:
 * ```
 * in = [4,5,6]
 * count = 2
 * return = [4,4,5,5,6,6]
 * ```
 * @throws `cudf::logic_error` if the data type of @p count is not size_type.
 * @throws `cudf::logic_error` if @p count is invalid or @p count is negative.
 * @throws `cudf::logic_error` if @p input_table.num_rows() * @p count overflows
 * size_type.
 * 
 * @param input_table Input table
 * @param count Non-null scalar of a integral type
 * @param mr Memory resource to allocate the result output table
 * @return std::unique_ptr<table> The result table containing the repetitions
 *---------------------------------------------------------------------------**/
std::unique_ptr<table> repeat(
    table_view const& input_table, scalar const& count,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf
