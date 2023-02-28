/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/strings/udf/case.cuh>
#include <cudf/strings/udf/char_types.cuh>
#include <cudf/strings/udf/replace.cuh>
#include <cudf/strings/udf/search.cuh>
#include <cudf/strings/udf/starts_with.cuh>
#include <cudf/strings/udf/strip.cuh>
#include <cudf/strings/udf/udf_string.cuh>

#include <cuda/atomic>

#include <cooperative_groups.h>

#include <limits>
#include <type_traits>

using namespace cudf::strings::udf;

extern "C" __device__ int len(int* nb_retval, void const* str)
{
  auto sv    = reinterpret_cast<cudf::string_view const*>(str);
  *nb_retval = sv->length();
  return 0;
}

extern "C" __device__ int startswith(bool* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = starts_with(*str_view, *substr_view);
  return 0;
}

extern "C" __device__ int endswith(bool* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = ends_with(*str_view, *substr_view);
  return 0;
}

extern "C" __device__ int contains(bool* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = (str_view->find(*substr_view) != cudf::string_view::npos);
  return 0;
}

extern "C" __device__ int find(int* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = str_view->find(*substr_view);
  return 0;
}

extern "C" __device__ int rfind(int* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = str_view->rfind(*substr_view);
  return 0;
}

extern "C" __device__ int eq(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view == *rhs_view);
  return 0;
}

extern "C" __device__ int ne(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view != *rhs_view);
  return 0;
}

extern "C" __device__ int ge(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view >= *rhs_view);
  return 0;
}

extern "C" __device__ int le(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view <= *rhs_view);
  return 0;
}

extern "C" __device__ int gt(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view > *rhs_view);
  return 0;
}

extern "C" __device__ int lt(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view < *rhs_view);
  return 0;
}

extern "C" __device__ int pyislower(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_lower(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisupper(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_upper(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisspace(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_space(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisdecimal(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_decimal(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisnumeric(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_numeric(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisdigit(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_digit(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisalnum(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_alpha_numeric(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisalpha(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_alpha(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyistitle(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_title(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pycount(int* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = count(*str_view, *substr_view);
  return 0;
}

extern "C" __device__ int udf_string_from_string_view(int* nb_retbal,
                                                      void const* str,
                                                      void* udf_str)
{
  auto str_view_ptr = reinterpret_cast<cudf::string_view const*>(str);
  auto udf_str_ptr  = new (udf_str) udf_string;
  *udf_str_ptr      = udf_string(*str_view_ptr);

  return 0;
}

extern "C" __device__ int strip(int* nb_retval,
                                void* udf_str,
                                void* const* to_strip,
                                void* const* strip_str)
{
  auto to_strip_ptr  = reinterpret_cast<cudf::string_view const*>(to_strip);
  auto strip_str_ptr = reinterpret_cast<cudf::string_view const*>(strip_str);
  auto udf_str_ptr   = new (udf_str) udf_string;

  *udf_str_ptr = strip(*to_strip_ptr, *strip_str_ptr);

  return 0;
}

extern "C" __device__ int lstrip(int* nb_retval,
                                 void* udf_str,
                                 void* const* to_strip,
                                 void* const* strip_str)
{
  auto to_strip_ptr  = reinterpret_cast<cudf::string_view const*>(to_strip);
  auto strip_str_ptr = reinterpret_cast<cudf::string_view const*>(strip_str);
  auto udf_str_ptr   = new (udf_str) udf_string;

  *udf_str_ptr = strip(*to_strip_ptr, *strip_str_ptr, cudf::strings::side_type::LEFT);

  return 0;
}

extern "C" __device__ int rstrip(int* nb_retval,
                                 void* udf_str,
                                 void* const* to_strip,
                                 void* const* strip_str)
{
  auto to_strip_ptr  = reinterpret_cast<cudf::string_view const*>(to_strip);
  auto strip_str_ptr = reinterpret_cast<cudf::string_view const*>(strip_str);
  auto udf_str_ptr   = new (udf_str) udf_string;

  *udf_str_ptr = strip(*to_strip_ptr, *strip_str_ptr, cudf::strings::side_type::RIGHT);

  return 0;
}
extern "C" __device__ int upper(int* nb_retval,
                                void* udf_str,
                                void const* st,
                                std::uintptr_t flags_table,
                                std::uintptr_t cases_table,
                                std::uintptr_t special_table)
{
  auto udf_str_ptr = new (udf_str) udf_string;
  auto st_ptr      = reinterpret_cast<cudf::string_view const*>(st);

  auto flags_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(flags_table);
  auto cases_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_cases_table_type*>(cases_table);
  auto special_table_ptr =
    reinterpret_cast<cudf::strings::detail::special_case_mapping*>(special_table);

  cudf::strings::udf::chars_tables tables{flags_table_ptr, cases_table_ptr, special_table_ptr};

  *udf_str_ptr = to_upper(tables, *st_ptr);

  return 0;
}

extern "C" __device__ int lower(int* nb_retval,
                                void* udf_str,
                                void const* st,
                                std::uintptr_t flags_table,
                                std::uintptr_t cases_table,
                                std::uintptr_t special_table)
{
  auto udf_str_ptr = new (udf_str) udf_string;
  auto st_ptr      = reinterpret_cast<cudf::string_view const*>(st);

  auto flags_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(flags_table);
  auto cases_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_cases_table_type*>(cases_table);
  auto special_table_ptr =
    reinterpret_cast<cudf::strings::detail::special_case_mapping*>(special_table);

  cudf::strings::udf::chars_tables tables{flags_table_ptr, cases_table_ptr, special_table_ptr};
  *udf_str_ptr = to_lower(tables, *st_ptr);
  return 0;
}

extern "C" __device__ int concat(int* nb_retval, void* udf_str, void* const* lhs, void* const* rhs)
{
  auto lhs_ptr = reinterpret_cast<cudf::string_view const*>(lhs);
  auto rhs_ptr = reinterpret_cast<cudf::string_view const*>(rhs);

  auto udf_str_ptr = new (udf_str) udf_string;

  udf_string result;
  result.append(*lhs_ptr).append(*rhs_ptr);
  *udf_str_ptr = result;
  return 0;
}

extern "C" __device__ int replace(
  int* nb_retval, void* udf_str, void* const src, void* const to_replace, void* const replacement)
{
  auto src_ptr         = reinterpret_cast<cudf::string_view const*>(src);
  auto to_replace_ptr  = reinterpret_cast<cudf::string_view const*>(to_replace);
  auto replacement_ptr = reinterpret_cast<cudf::string_view const*>(replacement);

  auto udf_str_ptr = new (udf_str) udf_string;
  *udf_str_ptr     = replace(*src_ptr, *to_replace_ptr, *replacement_ptr);

  return 0;
}

// Groupby Shim Functions
template <typename T>
__device__ bool are_all_nans(cooperative_groups::thread_block const& block,
                             T const* data,
                             int64_t size)
{
  // TODO: to be refactored with CG vote functions once
  // block size is known at build time
  __shared__ int64_t count;

  if (block.thread_rank() == 0) { count = 0; }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    if (not std::isnan(data[idx])) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_block> ref{count};
      ref.fetch_add(1, cuda::std::memory_order_relaxed);
      break;
    }
  }

  block.sync();
  return count == 0;
}

template <typename T>
__device__ void device_sum(cooperative_groups::thread_block const& block,
                           T const* data,
                           int64_t size,
                           T* sum)
{
  T local_sum = 0;

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_sum += data[idx];
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{*sum};
  ref.fetch_add(local_sum, cuda::std::memory_order_relaxed);

  block.sync();
}

template <typename T>
__device__ T BlockSum(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  if constexpr (std::is_floating_point_v<T>) {
    if (are_all_nans(block, data, size)) { return 0; }
  }

  __shared__ T block_sum;
  if (block.thread_rank() == 0) { block_sum = 0; }
  block.sync();

  device_sum<T>(block, data, size, &block_sum);
  return block_sum;
}

template <typename T>
__device__ double BlockMean(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_sum;
  if (block.thread_rank() == 0) { block_sum = 0; }
  block.sync();

  device_sum<T>(block, data, size, &block_sum);
  return static_cast<double>(block_sum) / static_cast<double>(size);
}

template <typename T>
__device__ double BlockVar(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ double block_var;
  __shared__ T block_sum;
  if (block.thread_rank() == 0) {
    block_var = 0;
    block_sum = 0;
  }
  block.sync();

  T local_sum      = 0;
  double local_var = 0;

  device_sum<T>(block, data, size, &block_sum);

  auto const mean = static_cast<double>(block_sum) / static_cast<double>(size);

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const delta = static_cast<double>(data[idx]) - mean;
    local_var += delta * delta;
  }

  cuda::atomic_ref<double, cuda::thread_scope_block> ref{block_var};
  ref.fetch_add(local_var, cuda::std::memory_order_relaxed);
  block.sync();

  if (block.thread_rank() == 0) { block_var = block_var / static_cast<double>(size - 1); }
  block.sync();
  return block_var;
}

template <typename T>
__device__ double BlockStd(T const* data, int64_t size)
{
  auto const var = BlockVar(data, size);
  return sqrt(var);
}

template <typename T>
__device__ T BlockMax(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  if constexpr (std::is_floating_point_v<T>) {
    if (are_all_nans(block, data, size)) { return std::numeric_limits<T>::quiet_NaN(); }
  }

  auto local_max = cudf::DeviceMax::identity<T>();
  __shared__ T block_max;
  if (block.thread_rank() == 0) { block_max = local_max; }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_max = max(local_max, data[idx]);
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_max};
  ref.fetch_max(local_max, cuda::std::memory_order_relaxed);

  block.sync();

  return block_max;
}

template <typename T>
__device__ T BlockMin(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  if constexpr (std::is_floating_point_v<T>) {
    if (are_all_nans(block, data, size)) { return std::numeric_limits<T>::quiet_NaN(); }
  }

  auto local_min = cudf::DeviceMin::identity<T>();

  __shared__ T block_min;
  if (block.thread_rank() == 0) { block_min = local_min; }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_min = min(local_min, data[idx]);
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_min};
  ref.fetch_min(local_min, cuda::std::memory_order_relaxed);

  block.sync();

  return block_min;
}

template <typename T>
__device__ int64_t BlockIdxMax(T const* data, int64_t* index, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_max;
  __shared__ int64_t block_idx_max;
  __shared__ bool found_max;

  auto local_max     = cudf::DeviceMax::identity<T>();
  auto local_idx_max = cudf::DeviceMin::identity<int64_t>();

  if (block.thread_rank() == 0) {
    block_max     = local_max;
    block_idx_max = local_idx_max;
    found_max     = false;
  }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const current_data = data[idx];
    if (current_data > local_max) {
      local_max     = current_data;
      local_idx_max = index[idx];
      found_max     = true;
    }
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_max};
  ref.fetch_max(local_max, cuda::std::memory_order_relaxed);
  block.sync();

  if (found_max) {
    if (local_max == block_max) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_block> ref_idx{block_idx_max};
      ref_idx.fetch_min(local_idx_max, cuda::std::memory_order_relaxed);
    }
  } else {
    if (block.thread_rank() == 0) { block_idx_max = index[0]; }
  }
  block.sync();

  return block_idx_max;
}

template <typename T>
__device__ int64_t BlockIdxMin(T const* data, int64_t* index, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_min;
  __shared__ int64_t block_idx_min;
  __shared__ bool found_min;

  auto local_min     = cudf::DeviceMin::identity<T>();
  auto local_idx_min = cudf::DeviceMin::identity<int64_t>();

  if (block.thread_rank() == 0) {
    block_min     = local_min;
    block_idx_min = local_idx_min;
    found_min     = false;
  }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const current_data = data[idx];
    if (current_data < local_min) {
      local_min     = current_data;
      local_idx_min = index[idx];
      found_min     = true;
    }
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_min};
  ref.fetch_min(local_min, cuda::std::memory_order_relaxed);
  block.sync();

  if (found_min) {
    if (local_min == block_min) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_block> ref_idx{block_idx_min};
      ref_idx.fetch_min(local_idx_min, cuda::std::memory_order_relaxed);
    }
  } else {
    if (block.thread_rank() == 0) { block_idx_min = index[0]; }
  }
  block.sync();

  return block_idx_min;
}

extern "C" {
#define make_definition(name, cname, type, return_type)                                          \
  __device__ int name##_##cname(return_type* numba_return_value, type* const data, int64_t size) \
  {                                                                                              \
    return_type const res = name<type>(data, size);                                              \
    *numba_return_value   = res;                                                                 \
    __syncthreads();                                                                             \
    return 0;                                                                                    \
  }

make_definition(BlockSum, int64, int64_t, int64_t);
make_definition(BlockSum, float64, double, double);
make_definition(BlockMean, int64, int64_t, double);
make_definition(BlockMean, float64, double, double);
make_definition(BlockStd, int64, int64_t, double);
make_definition(BlockStd, float64, double, double);
make_definition(BlockVar, int64, int64_t, double);
make_definition(BlockVar, float64, double, double);
make_definition(BlockMin, int64, int64_t, int64_t);
make_definition(BlockMin, float64, double, double);
make_definition(BlockMax, int64, int64_t, int64_t);
make_definition(BlockMax, float64, double, double);
#undef make_definition
}

extern "C" {
#define make_definition_idx(name, cname, type)                                   \
  __device__ int name##_##cname(                                                 \
    int64_t* numba_return_value, type* const data, int64_t* index, int64_t size) \
  {                                                                              \
    auto const res      = name<type>(data, index, size);                         \
    *numba_return_value = res;                                                   \
    __syncthreads();                                                             \
    return 0;                                                                    \
  }

make_definition_idx(BlockIdxMin, int64, int64_t);
make_definition_idx(BlockIdxMin, float64, double);
make_definition_idx(BlockIdxMax, int64, int64_t);
make_definition_idx(BlockIdxMax, float64, double);
#undef make_definition_idx
}

// Reference Counting
// nrt.h port, required for nrt.cpp
typedef void (*NRT_dtor_function)(void* ptr, size_t size, void* info);
typedef void (*NRT_dealloc_func)(void* ptr, void* dealloc_info);

typedef void* (*NRT_malloc_func)(size_t size);
typedef void* (*NRT_realloc_func)(void* ptr, size_t new_size);
typedef void (*NRT_free_func)(void* ptr);

// nrt.cpp port
extern "C" {
struct CUDAMemInfo {
  cuda::atomic<size_t> refct;
  NRT_dtor_function dtor;
  void* data;
  size_t size;
};
}

typedef struct CUDAMemInfo CUDANRT_MemInfo;

// vendored directly from nrt.cpp
struct CUDANRT_MemSys {
  /* Shutdown flag */
  int shutting;
  /* Stats */
  struct {
    bool enabled;
    cuda::atomic<size_t> alloc;
    cuda::atomic<size_t> free;
    cuda::atomic<size_t> mi_alloc;
    cuda::atomic<size_t> mi_free;
  } stats;
  /* System allocation functions */
  struct {
    NRT_malloc_func malloc;
    NRT_realloc_func realloc;
    NRT_free_func free;
  } allocator;
};

/* The Memory System object */
__device__ static CUDANRT_MemSys TheMSys;

static void nrt_fatal_error(const char* msg)
{
  fprintf(stderr, "Fatal Numba error: %s\n", msg);
  fflush(stderr); /* it helps in Windows debug build */
}

extern "C" __device__ void CUDANRT_MemSys_init(void)
{
  TheMSys.shutting = 0;
  // Stats are off by default, call NRT_MemSys_enable_stats to enable
  TheMSys.stats.enabled  = false;
  TheMSys.stats.alloc    = 0;
  TheMSys.stats.free     = 0;
  TheMSys.stats.mi_alloc = 0;
  TheMSys.stats.mi_free  = 0;
  /* Bind to CUDA allocator */
  TheMSys.allocator.malloc = malloc;
  //  TheMSys.allocator.realloc = realloc;
  TheMSys.allocator.free = free;
}

extern "C" void CUDANRT_MemSys_shutdown(void) { TheMSys.shutting = 1; }

extern "C" void CUDANRT_MemSys_enable_stats(void) { TheMSys.stats.enabled = true; }

extern "C" void CUDANRT_MemSys_disable_stats(void) { TheMSys.stats.enabled = false; }

extern "C" size_t CUDANRT_MemSys_stats_enabled(void) { return (size_t)TheMSys.stats.enabled; }

extern "C" void NRT_MemSys_set_allocator(NRT_malloc_func malloc_func,
                                         NRT_realloc_func realloc_func,
                                         NRT_free_func free_func)
{
  bool stats_cond = false;
  if ((malloc_func != TheMSys.allocator.malloc ||
       // realloc_func != TheMSys.allocator.realloc ||
       free_func != TheMSys.allocator.free) &&
      stats_cond) {
    nrt_fatal_error("cannot change allocator while blocks are allocated");
  }
  TheMSys.allocator.malloc = malloc_func;
  // TheMSys.allocator.realloc = realloc_func;
  TheMSys.allocator.free = free_func;
}

/* This value is used as a marker for "stats are disabled", it's ASCII "AAAA" */
static size_t _DISABLED_STATS_VALUE = 0x41414141;

extern "C" size_t NRT_MemSys_get_stats_alloc()
{
  if (TheMSys.stats.enabled) {
    return TheMSys.stats.alloc.load();
  } else {
    return _DISABLED_STATS_VALUE;
  }
}

extern "C" size_t NRT_MemSys_get_stats_free()
{
  if (TheMSys.stats.enabled) {
    return TheMSys.stats.free.load();
  } else {
    return _DISABLED_STATS_VALUE;
  }
}

extern "C" size_t NRT_MemSys_get_stats_mi_alloc()
{
  if (TheMSys.stats.enabled) {
    return TheMSys.stats.mi_alloc.load();
  } else {
    return _DISABLED_STATS_VALUE;
  }
}

extern "C" size_t NRT_MemSys_get_stats_mi_free()
{
  if (TheMSys.stats.enabled) {
    return TheMSys.stats.mi_free.load();
  } else {
    return _DISABLED_STATS_VALUE;
  }
}

/*
 * The MemInfo structure.
 */

extern "C" __device__ void CUDANRT_MemInfo_init(CUDAMemInfo* mi,
                                                void* data,
                                                size_t size,
                                                NRT_dtor_function dtor)
{
  mi->refct = 1; /* starts with 1 refct */
  mi->dtor  = dtor;
  mi->data  = data;
  mi->size  = size;
  /* Update stats */
  if (TheMSys.stats.enabled) { TheMSys.stats.mi_alloc++; }
}

extern "C" __device__ void* CUDANRT_Allocate_External(size_t size)
{
  void* ptr = NULL;
  ptr       = TheMSys.allocator.malloc(size);
  if (TheMSys.stats.enabled) { TheMSys.stats.alloc++; }
  return ptr;
}

extern "C" __device__ void* CUDANRT_Allocate(size_t size)
{
  return CUDANRT_Allocate_External(size);
}

__device__ CUDANRT_MemInfo* CUDANRT_MemInfo_new(void* data,
                                                size_t size,
                                                NRT_dtor_function dtor,
                                                void* dtor_info)
{
  CUDANRT_MemInfo* mi = (CUDANRT_MemInfo*)CUDANRT_Allocate(sizeof(CUDANRT_MemInfo));
  if (mi != NULL) { CUDANRT_MemInfo_init(mi, data, size, dtor); }
  return mi;
}

__device__ size_t CUDANRT_MemInfo_refcount(CUDANRT_MemInfo* mi)
{
  /* Should never returns 0 for a valid MemInfo */
  if (mi && mi->data)
    return mi->refct;
  else {
    return (size_t)-1;
  }
}

__device__ static void* nrt_allocate_meminfo_and_data(size_t size, CUDANRT_MemInfo** mi_out)
{
  CUDANRT_MemInfo* mi = NULL;
  char* base          = (char*)CUDANRT_Allocate_External(sizeof(CUDANRT_MemInfo) + size);
  if (base == NULL) {
    *mi_out = NULL; /* set meminfo to NULL as allocation failed */
    return NULL;    /* return early as allocation failed */
  }
  mi      = (CUDANRT_MemInfo*)base;
  *mi_out = mi;
  return (void*)((char*)base + sizeof(CUDANRT_MemInfo));
}

__device__ CUDANRT_MemInfo* CUDANRT_MemInfo_alloc(size_t size)
{
  CUDANRT_MemInfo* mi = NULL;
  void* data          = nrt_allocate_meminfo_and_data(size, &mi);
  if (data == NULL) { return NULL; /* return early as allocation failed */ }
  CUDANRT_MemInfo_init(mi, data, size, NULL);
  return mi;
}

__device__ static void nrt_internal_custom_dtor(void* ptr, size_t size, void* info)
{
  NRT_dtor_function dtor = (NRT_dtor_function)info;
  if (dtor) { dtor(ptr, size, NULL); }
}

__device__ CUDANRT_MemInfo* CUDANRT_MemInfo_alloc_dtor(size_t size, NRT_dtor_function dtor)
{
  CUDANRT_MemInfo* mi = NULL;
  void* data          = (void*)nrt_allocate_meminfo_and_data(size, &mi);
  if (data == NULL) { return NULL; /* return early as allocation failed */ }
  CUDANRT_MemInfo_init(mi, data, size, nrt_internal_custom_dtor);
  return mi;
}

__device__ static void* nrt_allocate_meminfo_and_data_align(size_t size,
                                                            unsigned align,
                                                            CUDANRT_MemInfo** mi)
{
  size_t offset = 0, intptr = 0, remainder = 0;
  char* base = (char*)nrt_allocate_meminfo_and_data(size + 2 * align, mi);
  if (base == NULL) { return NULL; /* return early as allocation failed */ }
  intptr = (size_t)base;
  /*
   * See if the allocation is aligned already...
   * Check if align is a power of 2, if so the modulo can be avoided.
   */
  if ((align & (align - 1)) == 0) {
    remainder = intptr & (align - 1);
  } else {
    remainder = intptr % align;
  }
  if (remainder == 0) { /* Yes */
    offset = 0;
  } else { /* No, move forward `offset` bytes */
    offset = align - remainder;
  }
  return (void*)((char*)base + offset);
}

extern "C" __device__ CUDANRT_MemInfo* NRT_MemInfo_alloc_aligned(size_t size, unsigned align)
{
  CUDANRT_MemInfo* mi = NULL;
  void* data          = nrt_allocate_meminfo_and_data_align(size, align, &mi);
  if (data == NULL) { return NULL; /* return early as allocation failed */ }
  CUDANRT_MemInfo_init(mi, data, size, NULL);
  return mi;
}

extern "C" __device__ void NRT_Free(void* ptr) { TheMSys.allocator.free(ptr); }

extern "C" __device__ void CUDANRT_dealloc(CUDANRT_MemInfo* mi) { NRT_Free(mi); }

typedef void NRT_managed_dtor(void* data);

__device__ static void nrt_manage_memory_dtor(void* data, size_t size, void* info)
{
  NRT_managed_dtor* dtor = (NRT_managed_dtor*)info;
  dtor(data);
}

__device__ static CUDANRT_MemInfo* nrt_manage_memory(void* data, NRT_managed_dtor dtor)
{
  return (CUDANRT_MemInfo*)(CUDANRT_MemInfo_new(data, 0, nrt_manage_memory_dtor, (void*)dtor));
}

typedef struct {
  /* Methods to create MemInfos.
  MemInfos are like smart pointers for objects that are managed by the Numba.
  */

  /* Allocate memory
  *nbytes* is the number of bytes to be allocated
  Returning a new reference.
  */
  CUDANRT_MemInfo* (*allocate)(size_t nbytes);

  /* Convert externally allocated memory into a MemInfo.
   *data* is the memory pointer
   *dtor* is the deallocator of the memory
   */
  CUDANRT_MemInfo* (*manage_memory)(void* data, NRT_managed_dtor dtor);

  /* Acquire a reference */
  void (*acquire)(CUDANRT_MemInfo* mi);

  /* Release a reference */
  void (*release)(CUDANRT_MemInfo* mi);

  /* Get MemInfo data pointer */
  void* (*get_data)(CUDANRT_MemInfo* mi);

} NRT_api_functions;

extern "C" __device__ void NRT_MemInfo_destroy(CUDANRT_MemInfo* mi) { CUDANRT_dealloc(mi); }

extern "C" __device__ void NRT_MemInfo_call_dtor(CUDANRT_MemInfo* mi)
{
  if (mi->dtor && !TheMSys.shutting) /* We have a destructor and the system is not shutting down */
    mi->dtor(mi->data, mi->size, NULL);
  /* Clear and release MemInfo */
  NRT_MemInfo_destroy(mi);
}

extern "C" __device__ void NRT_MemInfo_acquire(CUDANRT_MemInfo* mi)
{
  assert(mi->refct > 0 && "RefCt cannot be zero");
  mi->refct++;
}

extern "C" __device__ void NRT_MemInfo_release(CUDANRT_MemInfo* mi)
{
  assert(mi->refct > 0 && "RefCt cannot be 0");
  /* RefCt drop to zero */
  if ((--(mi->refct)) == 0) { NRT_MemInfo_call_dtor(mi); }
}

extern "C" __device__ void* NRT_MemInfo_data(CUDANRT_MemInfo* mi) { return mi->data; }

static const NRT_api_functions nrt_functions_table = {CUDANRT_MemInfo_alloc,
                                                      nrt_manage_memory,
                                                      NRT_MemInfo_acquire,
                                                      NRT_MemInfo_release,
                                                      NRT_MemInfo_data};

extern "C" const NRT_api_functions* CUDANRT_get_api(void) { return &nrt_functions_table; }
