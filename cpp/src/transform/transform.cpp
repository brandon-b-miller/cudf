/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <jit_preprocessed_files/transform/jit/kernel.cu.jit.hpp>
#include <jit_preprocessed_files/transform/jit/masked_udf_kernel.cu.jit.hpp>


#include <jit/cache.hpp>
#include <jit/parser.hpp>
#include <jit/type.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace transformation {
namespace jit {

void unary_operation(mutable_column_view output,
                     column_view input,
                     const std::string& udf,
                     data_type output_type,
                     bool is_ptx,
                     rmm::cuda_stream_view stream)
{
  std::string kernel_name =
    jitify2::reflection::Template("cudf::transformation::jit::kernel")  //
      .instantiate(cudf::jit::get_type_name(output.type()),  // list of template arguments
                   cudf::jit::get_type_name(input.type()));

  std::string cuda_source =
    is_ptx ? cudf::jit::parse_single_function_ptx(udf,  //
                                                  "GENERIC_UNARY_OP",
                                                  cudf::jit::get_type_name(output_type),
                                                  {0})
           : cudf::jit::parse_single_function_cuda(udf,  //
                                                   "GENERIC_UNARY_OP");

  cudf::jit::get_program_cache(*transform_jit_kernel_cu_jit)
    .get_kernel(
      kernel_name, {}, {{"transform/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, 0, stream.value())                                   //
    ->launch(output.size(),                                                                 //
             cudf::jit::get_data_ptr(output),
             cudf::jit::get_data_ptr(input));
}

std::vector<std::string> make_template_types(column_view outcol_view, table_view data_view)
{
  std::string mskptr_type = cudf::jit::get_type_name(cudf::data_type(cudf::type_to_id<cudf::bitmask_type>())) + "*";
  std::string offset_type = cudf::jit::get_type_name(cudf::data_type(cudf::type_to_id<cudf::offset_type>()));

  std::vector<std::string> template_types(1);
  template_types[0] = cudf::jit::get_type_name(outcol_view.type());

  for (auto const& col: data_view) {
    auto these_types = {
       cudf::jit::get_type_name(col.type()) + "*",
       mskptr_type,
       offset_type
    };

    template_types.insert(template_types.end(), these_types);
    
  }
  return template_types;
}


void generalized_operation(table_view data_view,
                           std::string const& binary_udf, 
                           data_type output_type, 
                           column_view const& outcol_view,
                           column_view const& outmsk_view,
                           rmm::mr::device_memory_resource* mr)
{

  std::vector<std::string> template_types = make_template_types(outcol_view, data_view);
  for (size_t i = 0; i < template_types.size(); i++) {
      std::cout << template_types[i] << std::endl;
  };
  std::cout << template_types.size() << std::endl;

  std::string generic_kernel_name = 
  jitify2::reflection::Template("cudf::transformation::jit::generic_udf_kernel")
    .instantiate(template_types);

  std::string generic_cuda_source = cudf::jit::parse_single_function_ptx(
                     binary_udf, "GENERIC_OP", cudf::jit::get_type_name(output_type), {0});

  // {size, out_ptr, out_mask_ptr, col0_ptr, col0_mask_ptr, col0_offset, col1_ptr...}
  std::vector<void*> kernel_args((data_view.num_columns() * 3) + 3);

  cudf::size_type size = outcol_view.size();
  const void* outcol_ptr = cudf::jit::get_data_ptr(outcol_view);
  const void* outmsk_ptr = cudf::jit::get_data_ptr(outmsk_view);
  kernel_args.insert(kernel_args.begin(), {&size, &outcol_ptr, &outmsk_ptr});

  std::vector<const void*> data_ptrs(data_view.num_columns());
  std::vector<cudf::bitmask_type const*> mask_ptrs(data_view.num_columns());
  std::vector<cudf::offset_type> offsets(data_view.num_columns());

  column_view col;
  for (int col_idx = 0; col_idx < data_view.num_columns(); col_idx++) {
    col = data_view.column(col_idx);

    data_ptrs[col_idx] = cudf::jit::get_data_ptr(col);
    mask_ptrs[col_idx] = col.null_mask();
    offsets[col_idx] = col.offset();

    kernel_args.insert(kernel_args.begin() + 3 * (col_idx + 1), {&data_ptrs[col_idx], &mask_ptrs[col_idx], &offsets[col_idx]});

  }

  rmm::cuda_stream_view generic_stream;
  cudf::jit::get_program_cache(*transform_jit_masked_udf_kernel_cu_jit)
    .get_kernel(
      generic_kernel_name, {}, {{"transform/jit/operation-udf.hpp", generic_cuda_source}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, 0, generic_stream.value())                                   //
    ->launch(kernel_args.data());

}

}  // namespace jit
}  // namespace transformation

namespace detail {
std::unique_ptr<column> transform(column_view const& input,
                                  std::string const& unary_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_fixed_width(input.type()), "Unexpected non-fixed-width type.");

  std::unique_ptr<column> output = make_fixed_width_column(
    output_type, input.size(), copy_bitmask(input), cudf::UNKNOWN_NULL_COUNT, stream, mr);

  if (input.is_empty()) { return output; }

  mutable_column_view output_view = *output;

  // transform
  transformation::jit::unary_operation(output_view, input, unary_udf, output_type, is_ptx, stream);

  return output;
}

std::unique_ptr<column> generalized_masked_op(table_view data_view, 
                                               std::string const& udf, 
                                               data_type output_type, 
                                               rmm::mr::device_memory_resource* mr)
{
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;
  std::unique_ptr<column> output = make_fixed_width_column(
    output_type, data_view.num_rows()
  );
  std::unique_ptr<column> output_mask = make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, data_view.num_rows()
  );

  mutable_column_view output_view = *output;
  mutable_column_view output_mask_view = *output_mask;

  transformation::jit::generalized_operation(data_view, udf, output_type, output_view, output_mask_view, mr);

  auto final_output_mask = cudf::bools_to_mask(output_mask_view);
  output.get()->set_null_mask(std::move(*(final_output_mask.first)));
  return output;
}



}  // namespace detail

std::unique_ptr<column> transform(column_view const& input,
                                  std::string const& unary_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::transform(input, unary_udf, output_type, is_ptx, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> generalized_masked_op(table_view data_view,
                                              std::string const& binary_udf, 
                                              data_type output_type, 
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::generalized_masked_op(data_view, binary_udf, output_type, mr);
}

}  // namespace cudf
