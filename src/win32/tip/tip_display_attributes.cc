// Copyright 2010-2021, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "win32/tip/tip_display_attributes.h"

#include <windows.h>

#include <string_view>

#include "absl/base/nullability.h"
#include "base/win32/com.h"

namespace mozc {
namespace win32 {
namespace tsf {

namespace {

constexpr std::wstring_view kInputDescription =
    L"TextService Display Attribute Input";
// atok-custom: ATOKに下線表示が無いため、下線を非表示にする。
constexpr TF_DISPLAYATTRIBUTE kInputAttribute = {
    {TF_CT_NONE, {}},  // text color
    {TF_CT_NONE, {}},  // background color
    TF_LS_NONE,        // underline style
    FALSE,             // underline boldness
    {TF_CT_NONE, {}},  // underline color
    TF_ATTR_INPUT      // attribute info
};

constexpr std::wstring_view kConvertedDescription =
    L"TextService Display Attribute Converted";
// atok-custom: ATOK風に「変換中」をシアンハイライトで明示する。
// 色はATOK for Windows 一太郎2020 Limitedの実機スクリーンショットから
// RGB値を実測して再現（背景=シアン、文字=黒）。
// atok-custom: ATOKに下線表示が無いため、下線を非表示にする。
constexpr TF_DISPLAYATTRIBUTE kConvertedAttribute = {
    {TF_CT_COLORREF, RGB(0, 0, 0)},      // text color: 黒
    {TF_CT_COLORREF, RGB(0, 255, 255)},  // background color: シアン
    TF_LS_NONE,               // underline style
    FALSE,                    // underline boldness
    {TF_CT_NONE, {}},         // underline color
    TF_ATTR_TARGET_CONVERTED  // attribute info
};

constexpr std::wstring_view kFocusedInputDescription =
    L"TextService Display Attribute Focused Input";
// atok-custom: 区切り調整（Must #6）でひらがなに復帰したフォーカス中文節を、
// 明示選択された変換候補（シアン）と区別するための色。色は仮値で、
// 実機で見比べて微調整する（2026-09-11 時点で未実測）。
constexpr TF_DISPLAYATTRIBUTE kFocusedInputAttribute = {
    {TF_CT_COLORREF, RGB(255, 255, 255)},  // text color: 白
    {TF_CT_COLORREF, RGB(0, 0, 128)},      // background color: 紺色
    TF_LS_NONE,               // underline style
    FALSE,                    // underline boldness
    {TF_CT_NONE, {}},         // underline color
    TF_ATTR_TARGET_CONVERTED  // attribute info
};

#ifdef GOOGLE_JAPANESE_INPUT_BUILD

// {DDF5CDBA-C3FF-4BAF-B817-CC9210FAD27E}
constexpr GUID kDisplayAttributeInput = {
    0xddf5cdba,
    0xc3ff,
    0x4baf,
    {0xb8, 0x17, 0xcc, 0x92, 0x10, 0xfa, 0xd2, 0x7e}};

// {F829C8C0-0EBB-4D29-BD2F-E413A944B7E4}
constexpr GUID kDisplayAttributeConverted = {
    0xf829c8c0,
    0x0ebb,
    0x4d29,
    {0xbd, 0x2f, 0xe4, 0x13, 0xa9, 0x44, 0xb7, 0xe4}};

// atok-custom: {5B3E9A1C-7D42-4F1B-9C3A-1E7F2B4D6A80}
constexpr GUID kDisplayAttributeFocusedInput = {
    0x5b3e9a1c,
    0x7d42,
    0x4f1b,
    {0x9c, 0x3a, 0x1e, 0x7f, 0x2b, 0x4d, 0x6a, 0x80}};

#else  // GOOGLE_JAPANESE_INPUT_BUILD

// {84CA1E7E-3020-4D1C-8968-DDA372D1E067}
constexpr GUID kDisplayAttributeInput = {
    0x84ca1e7e,
    0x3020,
    0x4d1c,
    {0x89, 0x68, 0xdd, 0xa3, 0x72, 0xd1, 0xe0, 0x67}};

// {8A4028E5-2DCD-4365-A5DC-71F67E797437}
constexpr GUID kDisplayAttributeConverted = {
    0x8a4028e5,
    0x2dcd,
    0x4365,
    {0xa5, 0xdc, 0x71, 0xf6, 0x7e, 0x79, 0x74, 0x37}};

// atok-custom: {C2F7A9E4-6B1D-4A8F-9E3C-2D5F8A1B7C40}
constexpr GUID kDisplayAttributeFocusedInput = {
    0xc2f7a9e4,
    0x6b1d,
    0x4a8f,
    {0x9e, 0x3c, 0x2d, 0x5f, 0x8a, 0x1b, 0x7c, 0x40}};

#endif  // !GOOGLE_JAPANESE_INPUT_BUILD

}  // namespace

TipDisplayAttribute::TipDisplayAttribute(const GUID& guid,
                                         const TF_DISPLAYATTRIBUTE& attribute,
                                         const std::wstring_view description)
    : guid_(guid),
      description_(description),
      attribute_(attribute),
      original_attribute_(attribute) {}

STDMETHODIMP TipDisplayAttribute::GetGUID(GUID* absl_nullable guid) {
  return SaveToOutParam(guid_, guid);
}

STDMETHODIMP
TipDisplayAttribute::GetDescription(BSTR* absl_nullable description) {
  return SaveToOutParam(MakeUniqueBSTR(description_), description);
}

STDMETHODIMP
TipDisplayAttribute::GetAttributeInfo(
    TF_DISPLAYATTRIBUTE* absl_nullable attribute) {
  return SaveToOutParam(attribute_, attribute);
}

STDMETHODIMP
TipDisplayAttribute::SetAttributeInfo(
    const TF_DISPLAYATTRIBUTE* absl_nullable attribute) {
  if (attribute == nullptr) {
    return E_INVALIDARG;
  }
  attribute_ = *attribute;
  return S_OK;
}

STDMETHODIMP TipDisplayAttribute::Reset() {
  attribute_ = original_attribute_;
  return S_OK;
}

TipDisplayAttributeInput::TipDisplayAttributeInput()
    : TipDisplayAttribute(kDisplayAttributeInput, kInputAttribute,
                          kInputDescription) {}

const GUID& TipDisplayAttributeInput::guid() { return kDisplayAttributeInput; }

TipDisplayAttributeConverted::TipDisplayAttributeConverted()
    : TipDisplayAttribute(kDisplayAttributeConverted, kConvertedAttribute,
                          kConvertedDescription) {}

const GUID& TipDisplayAttributeConverted::guid() {
  return kDisplayAttributeConverted;
}

TipDisplayAttributeFocusedInput::TipDisplayAttributeFocusedInput()
    : TipDisplayAttribute(kDisplayAttributeFocusedInput, kFocusedInputAttribute,
                          kFocusedInputDescription) {}

const GUID& TipDisplayAttributeFocusedInput::guid() {
  return kDisplayAttributeFocusedInput;
}

}  // namespace tsf
}  // namespace win32
}  // namespace mozc
