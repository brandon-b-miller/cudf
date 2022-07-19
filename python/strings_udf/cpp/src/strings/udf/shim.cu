/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/strings/udf/dstring.cuh>
#include <cudf/strings/udf/starts_with.cuh>
#include <cudf/strings/udf/char_types.cuh>
#include <cudf/strings/udf/search.cuh>

using namespace cudf::strings::udf;

/*
len
*/

extern "C"
__device__ int len(int* nb_retval, void* str) {
    cudf::string_view* sv = reinterpret_cast<cudf::string_view*>(str);
    *nb_retval = sv->length();
    return 0;
}


/*
startswith
*/

extern "C"
__device__ int startswith(bool* nb_retval, void* str, void* substr) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* substr_view = reinterpret_cast<cudf::string_view*>(substr);
    
    *nb_retval = starts_with(*str_view, *substr_view);
    return 0;
}

extern "C"
__device__ int endswith(bool* nb_retval, void* str, void* substr) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* substr_view = reinterpret_cast<cudf::string_view*>(substr);
    
    *nb_retval = ends_with(*str_view, *substr_view);
    return 0;
}

/*
contains
*/

extern "C"
__device__ int contains(bool* nb_retval, void* str, void* substr) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* substr_view = reinterpret_cast<cudf::string_view*>(substr);

    *nb_retval = (str_view->find(*substr_view) != cudf::string_view::npos); 
    return 0;
}

/*
find
*/
extern "C"
__device__ int find(int* nb_retval, void* str, void* substr) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* substr_view = reinterpret_cast<cudf::string_view*>(substr);
    
    *nb_retval = str_view->find(*substr_view);
    return 0;
}

/*
rfind
*/
extern "C"
__device__ int rfind(int* nb_retval, void* str, void* substr) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* substr_view = reinterpret_cast<cudf::string_view*>(substr);
    
    *nb_retval = str_view->rfind(*substr_view);
    return 0;
}

/*
==
*/

extern "C"
__device__ int eq(bool* nb_retval, void* str, void* rhs) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* rhs_view = reinterpret_cast<cudf::string_view*>(rhs);

    *nb_retval = (*str_view == *rhs_view);
    return 0;
}

/*
!=
*/

extern "C"
__device__ int ne(bool* nb_retval, void* str, void* rhs) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* rhs_view = reinterpret_cast<cudf::string_view*>(rhs);

    *nb_retval = (*str_view != *rhs_view);
    return 0;
}

/*
>=
*/

extern "C"
__device__ int ge(bool* nb_retval, void* str, void* rhs) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* rhs_view = reinterpret_cast<cudf::string_view*>(rhs);

    *nb_retval = (*str_view >= *rhs_view);
    return 0;
}

/*
<=
*/

extern "C"
__device__ int le(bool* nb_retval, void* str, void* rhs) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* rhs_view = reinterpret_cast<cudf::string_view*>(rhs);

    *nb_retval = (*str_view <= *rhs_view);
    return 0;
}

/*
>
*/

extern "C"
__device__ int gt(bool* nb_retval, void* str, void* rhs) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* rhs_view = reinterpret_cast<cudf::string_view*>(rhs);

    *nb_retval = (*str_view > *rhs_view);
    return 0;
}

/*
<
*/

extern "C"
__device__ int lt(bool* nb_retval, void* str, void* rhs) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* rhs_view = reinterpret_cast<cudf::string_view*>(rhs);

    *nb_retval = (*str_view < *rhs_view);
    return 0;
}

/*
islower
*/

extern "C"
__device__ int pyislower(bool* nb_retval, void* str, std::int64_t chars_table) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);

    *nb_retval = is_lower(reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
    return 0;
}

/*
isupper
*/

extern "C"
__device__ int pyisupper(bool* nb_retval, void* str, std::int64_t chars_table) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);

    *nb_retval = is_upper(reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
    return 0;
}

/*
isspace
*/

extern "C"
__device__ int pyisspace(bool* nb_retval, void* str, std::int64_t chars_table) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);

    *nb_retval = is_space(reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
    return 0;
}

/*
isdecimal
*/

extern "C"
__device__ int pyisdecimal(bool* nb_retval, void* str, std::int64_t chars_table) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);

    *nb_retval = is_decimal(reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
    return 0;
}

/*
isnumeric
*/

extern "C"
__device__ int pyisnumeric(bool* nb_retval, void* str, std::int64_t chars_table) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);

    *nb_retval = is_numeric(reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
    return 0;
}

/*
isdigit
*/

extern "C"
__device__ int pyisdigit(bool* nb_retval, void* str, std::int64_t chars_table) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);

    *nb_retval = is_digit(reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
    return 0;
}

/*
isalnum
*/

extern "C"
__device__ int pyisalnum(bool* nb_retval, void* str, std::int64_t chars_table) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);

    *nb_retval = is_alpha_numeric(reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
    return 0;
}

/*
count
*/

extern "C"
__device__ int pycount(int* nb_retval, void* str, void* substr) {
    cudf::string_view* str_view = reinterpret_cast<cudf::string_view*>(str);
    cudf::string_view* substr_view = reinterpret_cast<cudf::string_view*>(substr);

    *nb_retval = count(*str_view, *substr_view);
    return 0;
}
