/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "udf_string.hpp"

#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>

#include <algorithm>
#include <limits>
#include <string>

namespace cudf {
namespace strings {
namespace udf {
namespace detail {

__device__ inline static cudf::size_type bytes_in_null_terminated_string(char const* str)
{
  if (!str) return 0;
  cudf::size_type bytes = 0;
  while (*str++) ++bytes;
  return bytes;
}

}  // namespace detail

__device__ inline char* udf_string::allocate(cudf::size_type bytes)
{
  char* data  = static_cast<char*>(malloc(bytes + 1));
  data[bytes] = 0;  // add null-terminator so we can printf strings in device code
  return data;
}

__device__ inline void udf_string::deallocate(char* data)
{
  if (data) free(data);
}

__device__ void udf_string::reallocate(cudf::size_type bytes)
{
  m_capacity    = bytes;
  auto new_data = allocate(m_capacity);
  memcpy(new_data, m_data, std::min(m_bytes, bytes));
  deallocate(m_data);
  m_data = new_data;
}

__device__ inline udf_string::udf_string(char const* data, cudf::size_type bytes)
  : m_bytes(bytes), m_capacity(bytes)
{
  m_data = allocate(m_capacity);
  memcpy(m_data, data, bytes);
}

__device__ udf_string::udf_string(cudf::size_type count, cudf::char_utf8 chr)
{
  if (count <= 0) { return; }
  m_bytes = m_capacity = cudf::strings::detail::bytes_in_char_utf8(chr) * count;
  m_data               = allocate(m_capacity);
  auto out_ptr         = m_data;
  for (auto idx = 0; idx < count; ++idx) {
    out_ptr += cudf::strings::detail::from_char_utf8(chr, out_ptr);
  }
}

__device__ inline udf_string::udf_string(char const* data)
{
  m_bytes = m_capacity = detail::bytes_in_null_terminated_string(data);
  m_data               = allocate(m_capacity);
  memcpy(m_data, data, m_bytes);
}

__device__ inline udf_string::udf_string(udf_string const& src)
  : m_bytes(src.m_bytes), m_capacity(src.m_bytes)
{
  m_data = allocate(m_capacity);
  memcpy(m_data, src.m_data, m_bytes);
}

__device__ inline udf_string::udf_string(udf_string&& src)
  : m_data(src.m_data), m_bytes(src.m_bytes), m_capacity(src.m_capacity)
{
  src.m_data     = nullptr;
  src.m_bytes    = 0;
  src.m_capacity = 0;
}

__device__ inline udf_string::udf_string(cudf::string_view const str)
  : m_bytes(str.size_bytes()), m_capacity(str.size_bytes())
{
  m_data = allocate(m_capacity);
  memcpy(m_data, str.data(), m_bytes);
}

__device__ inline udf_string::~udf_string() { deallocate(m_data); }

__device__ inline udf_string& udf_string::operator=(udf_string const& str) { return assign(str); }

__device__ inline udf_string& udf_string::operator=(udf_string&& str)
{
  return assign(std::move(str));
}

__device__ inline udf_string& udf_string::operator=(cudf::string_view const str)
{
  return assign(str);
}

__device__ inline udf_string& udf_string::operator=(char const* str) { return assign(str); }

__device__ udf_string& udf_string::assign(udf_string&& str)
{
  if (this == &str) { return *this; }
  m_data         = str.m_data;
  m_bytes        = str.m_bytes;
  m_capacity     = str.m_capacity;
  str.m_data     = nullptr;
  str.m_bytes    = 0;
  str.m_capacity = 0;
  return *this;
}

__device__ udf_string& udf_string::assign(cudf::string_view const str)
{
  return assign(str.data(), str.size_bytes());
}

__device__ udf_string& udf_string::assign(char const* str)
{
  return assign(str, detail::bytes_in_null_terminated_string(str));
}

__device__ udf_string& udf_string::assign(char const* str, cudf::size_type bytes)
{
  if (bytes >= m_capacity) {
    deallocate(m_data);
    m_capacity = bytes;
    m_data     = allocate(m_capacity);
  }
  m_bytes = bytes;
  memcpy(m_data, str, bytes);
  m_data[m_bytes] = 0;
  return *this;
}

__device__ inline cudf::size_type udf_string::size_bytes() const { return m_bytes; }

__device__ inline cudf::size_type udf_string::length() const
{
  return cudf::strings::detail::characters_in_string(m_data, m_bytes);
}

__device__ cudf::size_type udf_string::max_size() const
{
  return std::numeric_limits<cudf::size_type>::max() - 1;
}

__device__ inline char* udf_string::data() { return m_data; }

__device__ inline char const* udf_string::data() const { return m_data; }

__device__ inline bool udf_string::is_empty() const { return m_bytes == 0; }

__device__ inline bool udf_string::is_null() const { return m_data == nullptr; }

__device__ inline cudf::string_view::const_iterator udf_string::begin() const
{
  return cudf::string_view::const_iterator(cudf::string_view(m_data, m_bytes), 0);
}

__device__ inline cudf::string_view::const_iterator udf_string::end() const
{
  return cudf::string_view::const_iterator(cudf::string_view(m_data, m_bytes), length());
}

__device__ inline cudf::char_utf8 udf_string::at(cudf::size_type pos) const
{
  auto const offset = byte_offset(pos);
  auto chr          = cudf::char_utf8{0};
  if (offset < m_bytes) { cudf::strings::detail::to_char_utf8(data() + offset, chr); }
  return chr;
}

__device__ inline cudf::char_utf8 udf_string::operator[](cudf::size_type pos) const
{
  return at(pos);
}

__device__ inline cudf::size_type udf_string::byte_offset(cudf::size_type pos) const
{
  cudf::size_type offset = 0;

  auto sptr = m_data;
  auto eptr = sptr + m_bytes;
  while ((pos > 0) && (sptr < eptr)) {
    auto const byte       = static_cast<uint8_t>(*sptr++);
    auto const char_bytes = cudf::strings::detail::bytes_in_utf8_byte(byte);
    if (char_bytes) { --pos; }
    offset += char_bytes;
  }
  return offset;
}

__device__ inline int udf_string::compare(cudf::string_view const in) const
{
  return compare(in.data(), in.size_bytes());
}

__device__ inline int udf_string::compare(char const* data, cudf::size_type bytes) const
{
  auto const view = static_cast<cudf::string_view>(*this);
  return view.compare(data, bytes);
}

__device__ inline bool udf_string::operator==(cudf::string_view const rhs) const
{
  return m_bytes == rhs.size_bytes() && compare(rhs) == 0;
}

__device__ inline bool udf_string::operator!=(cudf::string_view const rhs) const
{
  return compare(rhs) != 0;
}

__device__ inline bool udf_string::operator<(cudf::string_view const rhs) const
{
  return compare(rhs) < 0;
}

__device__ inline bool udf_string::operator>(cudf::string_view const rhs) const
{
  return compare(rhs) > 0;
}

__device__ inline bool udf_string::operator<=(cudf::string_view const rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc < 0);
}

__device__ inline bool udf_string::operator>=(cudf::string_view const rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc > 0);
}

__device__ inline void udf_string::clear()
{
  deallocate(m_data);
  m_data     = nullptr;
  m_bytes    = 0;
  m_capacity = 0;
}

__device__ inline void udf_string::resize(cudf::size_type count)
{
  if (count > max_size()) { return; }
  if (count > m_capacity) { reallocate(count); }

  // add padding if necessary (null chars)
  if (count > m_bytes) { memset(m_data + m_bytes, 0, count - m_bytes); }

  m_bytes         = count;
  m_data[m_bytes] = 0;
}

__device__ void udf_string::reserve(cudf::size_type count)
{
  if (count < max_size() && count > m_capacity) { reallocate(count); }
}

__device__ cudf::size_type udf_string::capacity() const { return m_capacity; }

__device__ void udf_string::shrink_to_fit()
{
  if (m_bytes < m_capacity) { reallocate(m_bytes); }
}

__device__ inline udf_string& udf_string::append(char const* str, cudf::size_type in_bytes)
{
  if (in_bytes <= 0) { return *this; }
  auto const nbytes = m_bytes + in_bytes;
  if (nbytes > m_capacity) { reallocate(2 * nbytes); }
  memcpy(m_data + m_bytes, str, in_bytes);
  m_bytes         = nbytes;
  m_data[m_bytes] = 0;
  return *this;
}

__device__ inline udf_string& udf_string::append(char const* str)
{
  return append(str, detail::bytes_in_null_terminated_string(str));
}

__device__ inline udf_string& udf_string::append(cudf::char_utf8 chr, cudf::size_type count)
{
  if (count <= 0) { return *this; }
  auto const char_bytes = cudf::strings::detail::bytes_in_char_utf8(chr) * count;
  auto const nbytes     = m_bytes + char_bytes;
  if (nbytes > m_capacity) { reallocate(2 * nbytes); }
  auto out_ptr = m_data + m_bytes;
  for (auto idx = 0; idx < count; ++idx) {
    out_ptr += cudf::strings::detail::from_char_utf8(chr, out_ptr);
  }
  m_bytes         = nbytes;
  m_data[m_bytes] = 0;
  return *this;
}

__device__ inline udf_string& udf_string::append(cudf::string_view const in)
{
  return append(in.data(), in.size_bytes());
}

__device__ inline udf_string& udf_string::operator+=(cudf::string_view const in)
{
  return append(in);
}

__device__ inline udf_string& udf_string::operator+=(cudf::char_utf8 chr) { return append(chr); }

__device__ inline udf_string& udf_string::operator+=(char const* str) { return append(str); }

__device__ inline udf_string& udf_string::insert(cudf::size_type pos,
                                                 char const* str,
                                                 cudf::size_type in_bytes)
{
  return replace(pos, 0, str, in_bytes);
}

__device__ inline udf_string& udf_string::insert(cudf::size_type pos, char const* str)
{
  return insert(pos, str, detail::bytes_in_null_terminated_string(str));
}

__device__ inline udf_string& udf_string::insert(cudf::size_type pos, cudf::string_view const in)
{
  return insert(pos, in.data(), in.size_bytes());
}

__device__ inline udf_string& udf_string::insert(cudf::size_type pos,
                                                 cudf::size_type count,
                                                 cudf::char_utf8 chr)
{
  return replace(pos, 0, count, chr);
}

__device__ inline udf_string udf_string::substr(cudf::size_type pos, cudf::size_type count) const
{
  if (pos < 0) { return udf_string{"", 0}; }
  auto const spos = byte_offset(pos);
  if (spos >= m_bytes) { return udf_string{"", 0}; }
  auto const epos = count < 0 ? m_bytes : std::min(byte_offset(pos + count), m_bytes);
  return udf_string{data() + spos, epos - spos};
}

// utility for replace()
__device__ void udf_string::shift_bytes(cudf::size_type spos,
                                        cudf::size_type epos,
                                        cudf::size_type nbytes)
{
  if (nbytes < m_bytes) {
    // shift bytes to the left [...wxyz] -> [wxyzxyz]
    auto src = epos;
    auto tgt = spos;
    while (tgt < nbytes) { m_data[tgt++] = m_data[src++]; }
  } else if (nbytes > m_bytes) {
    // shift bytes to the right [abcd...] -> [abcabcd]
    auto src = m_bytes;
    auto tgt = nbytes;
    while (src > epos) { m_data[--tgt] = m_data[--src]; }
  }
}

__device__ inline udf_string& udf_string::replace(cudf::size_type pos,
                                                  cudf::size_type count,
                                                  char const* str,
                                                  cudf::size_type in_bytes)
{
  if (pos < 0 || in_bytes < 0) { return *this; }
  auto const spos = byte_offset(pos);
  if (spos > m_bytes) { return *this; }
  auto const epos = count < 0 ? m_bytes : std::min(byte_offset(pos + count), m_bytes);

  // compute new size
  auto const nbytes = m_bytes + in_bytes - (epos - spos);
  if (nbytes > m_capacity) { reallocate(2 * nbytes); }

  // move bytes -- make room for replacement
  shift_bytes(spos + in_bytes, epos, nbytes);

  // insert the replacement
  memcpy(m_data + spos, str, in_bytes);

  m_bytes         = nbytes;
  m_data[m_bytes] = 0;
  return *this;
}

__device__ inline udf_string& udf_string::replace(cudf::size_type pos,
                                                  cudf::size_type count,
                                                  char const* str)
{
  return replace(pos, count, str, detail::bytes_in_null_terminated_string(str));
}

__device__ inline udf_string& udf_string::replace(cudf::size_type pos,
                                                  cudf::size_type count,
                                                  cudf::string_view const in)
{
  return replace(pos, count, in.data(), in.size_bytes());
}

__device__ inline udf_string& udf_string::replace(cudf::size_type pos,
                                                  cudf::size_type count,
                                                  cudf::size_type chr_count,
                                                  cudf::char_utf8 chr)
{
  if (pos < 0 || chr_count < 0) { return *this; }
  auto const spos = byte_offset(pos);
  if (spos > m_bytes) { return *this; }
  auto const epos = count < 0 ? m_bytes : std::min(byte_offset(pos + count), m_bytes);

  // compute input size
  auto const char_bytes = cudf::strings::detail::bytes_in_char_utf8(chr) * chr_count;
  // compute new output size
  auto const nbytes = m_bytes + char_bytes - (epos - spos);
  if (nbytes > m_capacity) { reallocate(2 * nbytes); }

  // move bytes -- make room for the new character(s)
  shift_bytes(spos + char_bytes, epos, nbytes);

  // copy chr chr_count times
  auto out_ptr = m_data + spos;
  for (auto idx = 0; idx < chr_count; ++idx) {
    out_ptr += cudf::strings::detail::from_char_utf8(chr, out_ptr);
  }

  m_bytes         = nbytes;
  m_data[m_bytes] = 0;
  return *this;
}

__device__ udf_string& udf_string::erase(cudf::size_type pos, cudf::size_type count)
{
  return replace(pos, count, nullptr, 0);
}

__device__ inline cudf::size_type udf_string::char_offset(cudf::size_type bytepos) const
{
  return cudf::strings::detail::characters_in_string(data(), bytepos);
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
