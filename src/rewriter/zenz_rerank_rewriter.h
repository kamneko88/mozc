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
//
// mozc_custom original file. Not part of upstream google/mozc.
//
// ZenzRerankRewriter re-scores the top-N candidates of each conversion
// segment using a small GGUF language model (zenz-v3.1-small, served via
// a vendored azooKey fork of llama.cpp) and nudges Mozc's own candidate
// cost by a penalty-only correction derived from the model's per-character
// negative log-likelihood (NLL). This is mozc_custom's "Should"-priority
// item: improving multi-segment conversion accuracy without touching
// Mozc's own converter/dictionary (see project Dev_note §6-19/§6-21).
//
// Design summary (agreed 2026-09-18, Dev_note §6-21):
//  - The shared prompt prefix (left context + reading + output tag) is
//    decoded once per segment; its last-position logits are cached and
//    reused for every candidate (they only depend on the shared prefix).
//  - Each candidate's own tokens are decoded on top of that shared prefix,
//    then the KV cache is rolled back (llama_kv_cache_seq_rm) so the next
//    candidate starts from the same shared prefix state.
//  - correction = weight * clamp(nll_per_char_i - nll_per_char_min, 0, kNllSpan)
//    is ADDED to the candidate's existing Mozc cost. Since Mozc cost is
//    "lower is better" (same direction as NLL), the best-NLL candidate in
//    the scored group gets correction == 0 (never rewarded), and worse
//    candidates only get penalized (never promoted past what Mozc itself
//    already believed). This mirrors azooKey-Desktop's own NLL reranker
//    (PR #356/#385: `max(nll_min - nll_i, -2.0) * weight` added to a
//    higher-is-better score, ranked descending) with the sign flipped to
//    match Mozc's lower-is-better cost.
//  - TODO(要検証・要チューニング): kWeight below is a placeholder. It must
//    be tuned against real on-device conversions once this is wired in
//    (batched verification, per project workflow).
//  - TODO(要検証): the model file path is currently the developer-machine
//    path under mozc_custom/_local/models. Packaging/installer strategy
//    for shipping the model has not been decided yet; see Dev_note §7.

#ifndef MOZC_REWRITER_ZENZ_RERANK_REWRITER_H_
#define MOZC_REWRITER_ZENZ_RERANK_REWRITER_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "converter/segments.h"
#include "request/conversion_request.h"
#include "rewriter/rewriter_interface.h"

// Forward-declare the llama.cpp C API types so this header does not need to
// pull in third_party/llama_cpp/include/llama.h (and its ggml transitive
// includes) for every translation unit that merely constructs a Rewriter.
struct llama_model;
struct llama_context;
struct llama_vocab;

namespace mozc {

// A rewriter that re-ranks the top candidates of each conversion segment
// using a local GGUF language model (zenz). See file comment for design.
//
// If the model fails to load (missing file, incompatible GGUF, etc.), the
// rewriter silently disables itself: Rewrite() becomes a no-op returning
// false, so the rest of the conversion pipeline is unaffected. This is a
// deliberate MVP choice so a missing/misconfigured model never breaks
// ordinary conversion.
class ZenzRerankRewriter : public RewriterInterface {
 public:
  // `model_path` is a filesystem path to a GGUF model file (e.g.
  // zenz-v3.1-small-Q5_K_M.gguf). The model is loaded eagerly in the
  // constructor; see IsAvailable() to check whether loading succeeded.
  explicit ZenzRerankRewriter(absl::string_view model_path);
  ZenzRerankRewriter(const ZenzRerankRewriter&) = delete;
  ZenzRerankRewriter& operator=(const ZenzRerankRewriter&) = delete;
  ~ZenzRerankRewriter() override;

  int capability(const ConversionRequest& request) const override;

  bool Rewrite(const ConversionRequest& request,
               Segments* segments) const override;

  // Returns true iff the model/context were loaded successfully and this
  // rewriter is able to score candidates. Exposed mainly for testing.
  bool IsAvailable() const { return model_ != nullptr && ctx_ != nullptr; }

 private:
  // Re-scores the candidates of a single conversion segment in place.
  // `left_context` is optional preceding confirmed text (katakana/kanji
  // mixed surface, as-is; empty means "no context"). Returns true if the
  // candidate order was changed.
  bool RerankSegment(absl::string_view left_context,
                     Segment* segment) const;

  // Computes the per-character NLL of `candidate_value` following the
  // already-decoded shared prefix (whose KV cache state is assumed current
  // at position `prefix_len`, with `cached_prefix_logits` being the logits
  // produced at the prefix's last position). On return, the KV cache is
  // rolled back to `prefix_len` again (i.e. candidate-specific tokens are
  // removed) so the next call starts from the same shared prefix state.
  // Returns std::nullopt if the candidate could not be tokenized/scored.
  std::optional<double> ScoreCandidateNllPerChar(
      absl::string_view candidate_value, int prefix_len,
      const std::vector<float>& cached_prefix_logits) const;

  std::string model_path_;
  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  int n_vocab_ = 0;
};

}  // namespace mozc

#endif  // MOZC_REWRITER_ZENZ_RERANK_REWRITER_H_
