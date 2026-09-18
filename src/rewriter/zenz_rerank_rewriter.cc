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
// See zenz_rerank_rewriter.h for the design summary.

#include "rewriter/zenz_rerank_rewriter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "base/strings/japanese.h"
#include "base/util.h"
#include "converter/candidate.h"
#include "converter/segments.h"
#include "request/conversion_request.h"
#include "llama.h"  // @llama_cpp//:llama_cpp (see bazel/BUILD.llama_cpp.bazel)

namespace mozc {
namespace {

// zenz-v3 prompt special tokens (Unicode Private Use Area). Written as
// literal UTF-8 byte sequences (rather than \u escapes) so the exact bytes
// are independent of the compiler's source/execution charset handling.
//   U+EE00 (input/reading marker)  -> EE B8 80
//   U+EE01 (output marker)         -> EE B8 81
//   U+EE02 (left-context marker)   -> EE B8 82
// Source: AzooKeyKanaKanjiConverter/Docs/zenzai.md (azooKey/AzooKeyKanaKanjiConverter).
// TODO(要検証): 実際のzenzモデルのGGUFボキャブラリでこれらが1トークンとして
// 認識されるか（llama_tokenize(parse_special=true)）は未確認。
constexpr absl::string_view kInputTag("\xEE\xB8\x80", 3);    // U+EE00
constexpr absl::string_view kOutputTag("\xEE\xB8\x81", 3);   // U+EE01
constexpr absl::string_view kContextTag("\xEE\xB8\x82", 3);  // U+EE02

// Score at most this many top candidates per segment (Dev_note §6-21).
constexpr size_t kTopN = 5;

// Penalty-only correction span, mirrored from azooKey-Desktop's own NLL
// reranker (PR #356/#385: `max(nll_min - nll_i, -2.0) * weight`). Mozc's
// candidate cost is "lower is better", the same direction as NLL itself, so
// the correction below is added with the opposite sign from azooKey's
// higher-is-better score: correction >= 0, and it is 0 for whichever scored
// candidate the language model likes best (never a reward, only a penalty
// for the others).
constexpr double kNllSpan = 2.0;

// TODO(要検証・要チューニング): 暫定値。Mozcのcostは数百〜のオーダーで動くため、
// このweightは実機での体感確認をもとに調整すること（本セッションではビルドが
// 通ることの確認までがスコープ。実機確認は他機能とまとめて後日実施）。
constexpr double kWeight = 300.0;

// Computes -log(softmax(logits)[token]), i.e. the negative log-likelihood
// of `token` under the distribution described by `logits` (length n_vocab).
double NegLogLikelihood(const float* logits, int n_vocab, llama_token token) {
  float max_logit = logits[0];
  for (int i = 1; i < n_vocab; ++i) {
    max_logit = std::max(max_logit, logits[i]);
  }
  double sum_exp = 0.0;
  for (int i = 0; i < n_vocab; ++i) {
    sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
  }
  const double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);
  if (token < 0 || token >= n_vocab) {
    // Should not happen (token ids come from llama_tokenize against the
    // same vocab), but guard defensively rather than reading out of bounds.
    return log_sum_exp;
  }
  return log_sum_exp - static_cast<double>(logits[token]);
}

// Builds the left-context text: up to the last 2 confirmed history segments
// plus (if scoring a non-first conversion segment in this call) the
// immediately preceding conversion segment's current top candidate.
// Dev_note §6-21: "文脈は直前の確定セグメント1〜2個程度を使う想定".
std::string BuildLeftContext(const Segments& segments, size_t conv_index) {
  std::string context;
  const size_t hist_size = segments.history_segments_size();
  const size_t hist_start = hist_size > 2 ? hist_size - 2 : 0;
  for (size_t h = hist_start; h < hist_size; ++h) {
    const Segment& hs = segments.history_segment(h);
    if (hs.candidates_size() > 0) {
      context.append(hs.candidate(0).value);
    }
  }
  if (conv_index > 0) {
    const Segment& prev = segments.conversion_segment(conv_index - 1);
    if (prev.candidates_size() > 0) {
      context.append(prev.candidate(0).value);
    }
  }
  return context;
}

}  // namespace

ZenzRerankRewriter::ZenzRerankRewriter(absl::string_view model_path)
    : model_path_(std::string(model_path)) {
  // Safe to call even if another llama.cpp-backed component has already
  // initialized the backend; ZenzRerankRewriter is currently the only user.
  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0;  // CPU only for now (GPU/CUDA path is 未検証).

  model_ = llama_model_load_from_file(model_path_.c_str(), mparams);
  if (model_ == nullptr) {
    LOG(WARNING) << "ZenzRerankRewriter: failed to load model from "
                 << model_path_ << "; rewriter disabled.";
    return;
  }

  vocab_ = llama_model_get_vocab(model_);
  n_vocab_ = llama_vocab_n_tokens(vocab_);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;  // Prompts here are short: brief context + 1 segment.
  cparams.n_batch = 512;
  // atok-custom: 8/16 logical processors (was 4; bumped 2026-09-18 after
  // on-device feedback that conversion felt slightly sluggish). Kept below
  // the full core count so it doesn't compete heavily with other apps
  // (e.g. video/image editors) running at the same time.
  cparams.n_threads = 8;        // TODO(要検証・要チューニング)
  cparams.n_threads_batch = 8;  // TODO(要検証・要チューニング)

  ctx_ = llama_init_from_model(model_, cparams);
  if (ctx_ == nullptr) {
    LOG(WARNING) << "ZenzRerankRewriter: failed to create context; "
                    "rewriter disabled.";
    llama_model_free(model_);
    model_ = nullptr;
    vocab_ = nullptr;
    return;
  }
}

ZenzRerankRewriter::~ZenzRerankRewriter() {
  if (ctx_ != nullptr) {
    llama_free(ctx_);
  }
  if (model_ != nullptr) {
    llama_model_free(model_);
  }
}

int ZenzRerankRewriter::capability(const ConversionRequest& request) const {
  if (!IsAvailable()) {
    return RewriterInterface::NOT_AVAILABLE;
  }
  return RewriterInterface::CONVERSION;
}

std::optional<double> ZenzRerankRewriter::ScoreCandidateNllPerChar(
    absl::string_view candidate_value, int prefix_len,
    const std::vector<float>& cached_prefix_logits) const {
  // Defensive reset: make sure the KV cache is exactly at "right after the
  // shared prefix" before decoding this candidate's own tokens, regardless
  // of how the previous call to this function returned.
  llama_kv_cache_seq_rm(ctx_, /*seq_id=*/0, /*p0=*/prefix_len, /*p1=*/-1);

  const size_t char_len = Util::CharsLen(candidate_value);
  if (char_len == 0) {
    return std::nullopt;
  }

  std::vector<llama_token> tokens(candidate_value.size() + 8);
  const int n_tokens = llama_tokenize(
      vocab_, candidate_value.data(),
      static_cast<int32_t>(candidate_value.size()), tokens.data(),
      static_cast<int32_t>(tokens.size()), /*add_special=*/false,
      /*parse_special=*/false);
  if (n_tokens <= 0) {
    return std::nullopt;
  }
  tokens.resize(n_tokens);

  // First candidate token is scored against the cached prefix logits (they
  // only depend on the shared prefix, so they are identical for every
  // candidate in this segment and were computed once by the caller).
  double nll_sum = NegLogLikelihood(cached_prefix_logits.data(), n_vocab_, tokens[0]);

  if (n_tokens > 1) {
    // Feed tokens[0 .. n_tokens-2] (n_tokens-1 tokens); their output logits
    // predict tokens[1 .. n_tokens-1] respectively. The last token's own
    // continuation is not scored (no EOS is appended/scored).
    const int n_feed = n_tokens - 1;
    llama_batch batch = llama_batch_init(n_feed, /*embd=*/0, /*n_seq_max=*/1);
    batch.n_tokens = n_feed;
    for (int i = 0; i < n_feed; ++i) {
      batch.token[i] = tokens[i];
      batch.pos[i] = prefix_len + i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = 1;  // logits needed at every fed position this time.
    }
    const int ret = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (ret != 0) {
      LOG(WARNING) << "ZenzRerankRewriter: llama_decode failed for candidate "
                      "continuation (ret="
                   << ret << ").";
      return std::nullopt;
    }
    const float* logits_base = llama_get_logits(ctx_);
    for (int i = 0; i < n_feed; ++i) {
      const float* row = logits_base + static_cast<size_t>(i) * n_vocab_;
      nll_sum += NegLogLikelihood(row, n_vocab_, tokens[i + 1]);
    }
  }

  return nll_sum / static_cast<double>(char_len);
}

bool ZenzRerankRewriter::RerankSegment(absl::string_view left_context,
                                       Segment* segment) const {
  const size_t score_count = std::min(kTopN, segment->candidates_size());
  if (score_count < 2) {
    return false;
  }

  const std::string katakana_reading =
      japanese::HiraganaToKatakana(segment->key());

  std::string prefix;
  if (!left_context.empty()) {
    prefix.append(kContextTag.data(), kContextTag.size());
    prefix.append(left_context.data(), left_context.size());
  }
  prefix.append(kInputTag.data(), kInputTag.size());
  prefix.append(katakana_reading);
  prefix.append(kOutputTag.data(), kOutputTag.size());

  std::vector<llama_token> prefix_tokens(prefix.size() + 8);
  int n_prefix = llama_tokenize(
      vocab_, prefix.data(), static_cast<int32_t>(prefix.size()),
      prefix_tokens.data(), static_cast<int32_t>(prefix_tokens.size()),
      /*add_special=*/false, /*parse_special=*/true);
  if (n_prefix <= 0) {
    return false;
  }
  prefix_tokens.resize(n_prefix);

  // Fresh start: drop any leftover KV state from a previous segment/call.
  llama_kv_cache_seq_rm(ctx_, /*seq_id=*/0, /*p0=*/0, /*p1=*/-1);

  llama_batch prefix_batch = llama_batch_init(n_prefix, /*embd=*/0, /*n_seq_max=*/1);
  prefix_batch.n_tokens = n_prefix;
  for (int i = 0; i < n_prefix; ++i) {
    prefix_batch.token[i] = prefix_tokens[i];
    prefix_batch.pos[i] = i;
    prefix_batch.n_seq_id[i] = 1;
    prefix_batch.seq_id[i][0] = 0;
    prefix_batch.logits[i] = (i == n_prefix - 1) ? 1 : 0;
  }
  const int decode_ret = llama_decode(ctx_, prefix_batch);
  llama_batch_free(prefix_batch);
  if (decode_ret != 0) {
    LOG(WARNING) << "ZenzRerankRewriter: llama_decode failed for shared "
                    "prefix (ret="
                 << decode_ret << ").";
    return false;
  }

  const float* last_logits = llama_get_logits(ctx_);
  const std::vector<float> cached_prefix_logits(last_logits,
                                                 last_logits + n_vocab_);

  struct Scored {
    size_t index;
    double nll_per_char;
  };
  std::vector<Scored> scored;
  scored.reserve(score_count);
  for (size_t c = 0; c < score_count; ++c) {
    const converter::Candidate& candidate =
        segment->candidate(static_cast<int>(c));
    const std::optional<double> nll = ScoreCandidateNllPerChar(
        candidate.value, n_prefix, cached_prefix_logits);
    if (nll.has_value()) {
      scored.push_back({c, *nll});
    }
  }
  if (scored.size() < 2) {
    // Not enough successfully-scored candidates to derive a meaningful
    // relative correction.
    return false;
  }

  const double nll_min =
      std::min_element(scored.begin(), scored.end(),
                       [](const Scored& a, const Scored& b) {
                         return a.nll_per_char < b.nll_per_char;
                       })
          ->nll_per_char;

  bool any_correction = false;
  for (const Scored& s : scored) {
    const double correction =
        kWeight * std::clamp(s.nll_per_char - nll_min, 0.0, kNllSpan);
    if (correction > 0.0) {
      converter::Candidate* candidate =
          segment->mutable_candidate(static_cast<int>(s.index));
      candidate->cost += static_cast<int32_t>(std::lround(correction));
      any_correction = true;
    }
  }
  if (!any_correction) {
    // All scored candidates tied at nll_min: costs unchanged, order
    // unchanged. Nothing further to do.
    return false;
  }

  // Re-derive display order among the scored candidates by (now-adjusted)
  // cost, stable on ties. Candidates are identified by value (matches the
  // approach used elsewhere in this codebase, e.g.
  // UserSegmentHistoryRewriter::SortCandidates) since move_candidate() can
  // shift other candidates' indices as we go.
  std::vector<std::string> target_order_values;
  target_order_values.reserve(scored.size());
  {
    std::vector<Scored> by_cost = scored;
    std::stable_sort(by_cost.begin(), by_cost.end(),
                     [segment](const Scored& a, const Scored& b) {
                       return segment->candidate(static_cast<int>(a.index)).cost <
                              segment->candidate(static_cast<int>(b.index)).cost;
                     });
    for (const Scored& s : by_cost) {
      target_order_values.push_back(
          segment->candidate(static_cast<int>(s.index)).value);
    }
  }

  bool modified = false;
  size_t next_pos = 0;
  for (const std::string& value : target_order_values) {
    int found_index = -1;
    for (size_t l = next_pos; l < segment->candidates_size(); ++l) {
      if (segment->candidate(static_cast<int>(l)).value == value) {
        found_index = static_cast<int>(l);
        break;
      }
    }
    if (found_index < 0) {
      continue;  // Should not happen; be defensive rather than crash.
    }
    if (static_cast<size_t>(found_index) != next_pos) {
      segment->move_candidate(found_index, static_cast<int>(next_pos));
      modified = true;
    }
    ++next_pos;
  }

  return modified;
}

bool ZenzRerankRewriter::Rewrite(const ConversionRequest& request,
                                 Segments* segments) const {
  if (!IsAvailable() || segments == nullptr) {
    return false;
  }

  bool modified = false;
  for (size_t i = 0; i < segments->conversion_segments_size(); ++i) {
    Segment* segment = segments->mutable_conversion_segment(i);
    if (segment->segment_type() == Segment::FIXED_VALUE) {
      continue;
    }
    if (segment->candidates_size() <= 1) {
      continue;
    }
    const std::string left_context = BuildLeftContext(*segments, i);
    modified |= RerankSegment(left_context, segment);
  }
  return modified;
}

}  // namespace mozc
