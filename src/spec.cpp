/* Speculative-decoding driver (MTP first). Adapted from llama.cpp's
 * examples/speculative-simple, with two changes for embedded-MTP heads:
 *   - the draft context comes from common_speculative_init_from_params (the head
 *     lives in the target model), not a separate draft model;
 *   - the "evaluate on the draft model" step is common_speculative_process()
 *     (the MTP head consumes the target's next-n embeddings), not a plain decode.
 * Correctness is guaranteed by the TARGET verify (common_sampler_sample_and_accept_n
 * samples from the target's logits), so committed tokens match plain decoding
 * regardless of how good the drafts are — that's the whole point of spec-decode. */
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"

#include "spec.h"

#include "ggml.h"

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>

struct basi_spec {
    llama_model   *model   = nullptr;
    llama_context *ctx_tgt = nullptr;
    llama_context *ctx_dft = nullptr;

    common_params params;                             // kept alive: init reads it
    common_speculative_init_result_ptr spec_init;     // owns ctx_dft
    common_speculative *spec = nullptr;
    common_sampler     *smpl = nullptr;

    llama_batch batch_tgt;

    long n_drafted = 0;
    long n_accept  = 0;
};

extern "C" basi_spec *basi_spec_init(struct llama_model *model,
                                     struct llama_context *ctx_tgt,
                                     const char *model_path,
                                     int n_ctx, int n_gpu_layers,
                                     const char *spec_type, int n_max,
                                     unsigned int seed, float temp) {
    if (!model || !ctx_tgt || !spec_type || !model_path) return nullptr;

    std::vector<enum common_speculative_type> types =
        common_speculative_types_from_names({ std::string(spec_type) });
    if (types.empty() || types[0] == COMMON_SPECULATIVE_TYPE_NONE) return nullptr;

    basi_spec *s = new basi_spec();
    s->model   = model;
    s->ctx_tgt = ctx_tgt;

    // mirror how ctx_tgt was built so the MTP draft context matches
    s->params.model.path    = model_path;
    s->params.n_ctx         = n_ctx;
    s->params.n_gpu_layers  = n_gpu_layers;
    s->params.speculative.types       = types;
    s->params.speculative.draft.n_max = n_max > 0 ? n_max : 1;

    // Build a draft context only for the draft-* types (MTP/Eagle/DFlash) — they
    // need one, and init_from_params ABORTS without a draft model. ngram-* types
    // draft from the prompt/context and need no ctx_dft.
    const bool needs_dft = (strncmp(spec_type, "draft-", 6) == 0);
    if (needs_dft) {
        fprintf(stderr, "[dbg] init: common_speculative_init_from_params...\n");
        s->spec_init = common_speculative_init_from_params(s->params, model, ctx_tgt);
        if (!s->spec_init) { fprintf(stderr, "[dbg] init: from_params NULL\n"); delete s; return nullptr; }
        s->ctx_dft = s->spec_init->context();   // may still be null
        fprintf(stderr, "[dbg] init: ctx_dft=%p\n", (void*)s->ctx_dft);
    } else {
        fprintf(stderr, "[dbg] init: ngram type, no ctx_dft\n");
    }

    s->params.speculative.draft.ctx_tgt = ctx_tgt;
    s->params.speculative.draft.ctx_dft = s->ctx_dft;   // null for ngram

    fprintf(stderr, "[dbg] init: common_speculative_init...\n");
    s->spec = common_speculative_init(s->params.speculative, 1);
    if (!s->spec) { fprintf(stderr, "[dbg] init: spec NULL\n"); delete s; return nullptr; }
    fprintf(stderr, "[dbg] init: spec ok\n");

    // note: common_speculative_init constructs the draft-mtp impl, whose ctor
    // already enables next-n embeddings on ctx_tgt/ctx_dft — we don't set them.

    // target sampler: greedy when temp<=0 (the lossless setting). No grammar yet
    // (milestone 1); the grammar milestone adds it via sparams.grammar.
    common_params_sampling sp;
    sp.seed = seed;
    sp.temp = temp;
    if (temp <= 0.0f) {
        // pure argmax: only top_k=1 in the chain, penalties disabled — so the
        // verifier matches a plain raw-argmax decode (the lossless reference).
        sp.top_k = 1;
        sp.penalty_repeat = 1.0f;
        sp.penalty_last_n = 0;
        sp.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    }
    s->smpl = common_sampler_init(model, sp);
    if (!s->smpl) { basi_spec_free(s); return nullptr; }

    s->batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    return s;
}

extern "C" int basi_spec_run(basi_spec *s,
                             const llama_token *prompt, int n_prompt,
                             long cap, basi_spec_emit emit, void *ud) {
    if (!s || !prompt || n_prompt <= 0) return -1;

    const llama_vocab *vocab = llama_model_get_vocab(s->model);
    const llama_seq_id seq_id = 0;

    fprintf(stderr, "[dbg] run: prefill (%d toks)...\n", n_prompt);
    // Prefill the target with all but the last prompt token. NOTE: do NOT call
    // common_speculative_process() on a llama_batch_get_one() batch — that helper
    // leaves seq_id/n_seq_id/pos NULL and process() dereferences them. Priming the
    // MTP head is not needed for correctness (the target verify guarantees the
    // committed tokens); the loop's process() calls (on fully-built batches) prime
    // it from the first iteration.
    {
        llama_batch pb = llama_batch_get_one(const_cast<llama_token *>(prompt), n_prompt - 1);
        if (llama_decode(s->ctx_tgt, pb) != 0) return -1;
        fprintf(stderr, "[dbg] run: prefill decoded\n");
    }

    llama_token id_last = prompt[n_prompt - 1];

    llama_tokens prompt_tgt(prompt, prompt + n_prompt - 1);
    prompt_tgt.reserve(llama_n_ctx(s->ctx_tgt));

    int  n_past    = n_prompt - 1;
    long n_emitted = 0;
    bool has_eos   = false;

    common_speculative_begin(s->spec, seq_id, prompt_tgt);
    fprintf(stderr, "[dbg] run: begin done, entering loop\n");

    const bool use_ckpt_tgt = (common_context_can_seq_rm(s->ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    const bool use_ckpt_dft = s->ctx_dft && (common_context_can_seq_rm(s->ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    fprintf(stderr, "[dbg] run: use_ckpt tgt=%d dft=%d\n", use_ckpt_tgt, use_ckpt_dft);

    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    while (true) {
        if (draft.empty()) {
            ckpt.update_pos(prompt_tgt.size(),
                            llama_memory_seq_pos_min(llama_get_memory(s->ctx_tgt), seq_id),
                            llama_memory_seq_pos_max(llama_get_memory(s->ctx_tgt), seq_id));
            if (use_ckpt_dft) {
                ckpt.update_dft(s->ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            common_speculative_get_draft_params(s->spec, seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ -1,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt_tgt,
                /* .result   = */ &draft,
            };
            common_speculative_draft(s->spec);

            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(s->ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (s->ctx_dft && use_ckpt_dft) {
                ckpt.load_dft(s->ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(llama_get_memory(s->ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        }

        // target batch: [id_last, draft0, draft1, ...]
        common_batch_clear(s->batch_tgt);
        common_batch_add(s->batch_tgt, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(s->batch_tgt, draft[i], n_past + (int) i, { seq_id }, true);
        }

        if (llama_decode(s->ctx_tgt, s->batch_tgt) != 0) return -1;

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) smpl_save.reset(common_sampler_clone(s->smpl));

        // Read the TARGET logits (verify + accept) BEFORE priming the head. For MTP
        // the draft context shares the target's compute buffers, so calling
        // common_speculative_process() first would overwrite the very logits that
        // accept-n reads -> garbage after the first token.
        std::vector<llama_token> ids = common_sampler_sample_and_accept_n(s->smpl, s->ctx_tgt, draft);

        // MTP: now feed the batch to the speculator (extracts next-n embeddings,
        // runs the draft head) so the NEXT draft is primed.
        common_speculative_process(s->spec, s->batch_tgt);
        if (ids.empty()) return -1;

        // partial acceptance without full seq-rm support: restore + carry as next draft
        if (use_ckpt_tgt && ids.size() - 1 < draft.size()) {
            draft = std::move(ids);
            ckpt.load_tgt(s->ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(s->ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            if (s->ctx_dft && use_ckpt_dft) {
                ckpt.load_dft(s->ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(llama_get_memory(s->ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
            prompt_tgt.resize(ckpt.n_tokens);
            s->smpl = smpl_save.release();  // adopt the restored sampler
            n_past  = (int) prompt_tgt.size();
            continue;
        }

        common_speculative_accept(s->spec, seq_id, (uint16_t)(ids.size() - 1));

        n_past       += (int) ids.size() - 1;
        s->n_drafted += (long) draft.size();
        s->n_accept  += (long) ids.size() - 1;

        // commit accepted tokens
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            n_emitted++;
            const bool eog = llama_vocab_is_eog(vocab, id_last);
            if (eog) has_eos = true;
            if (emit) emit(id_last, eog ? 1 : 0, ud);
            if (eog) break;
            if (cap > 0 && n_emitted >= cap) break;
        }

        draft.clear();
        llama_memory_seq_rm(llama_get_memory(s->ctx_tgt), seq_id, n_past, -1);
        if (s->ctx_dft) llama_memory_seq_rm(llama_get_memory(s->ctx_dft), seq_id, n_past, -1);

        if (has_eos) break;
        if (cap > 0 && n_emitted >= cap) break;
        if ((uint32_t) n_past + 1 >= llama_n_ctx(s->ctx_tgt)) break;  // context full
    }

    return (int) n_emitted;
}

extern "C" void basi_spec_stats(const basi_spec *s, long *drafted, long *accepted) {
    if (!s) { if (drafted) *drafted = 0; if (accepted) *accepted = 0; return; }
    if (drafted)  *drafted  = s->n_drafted;
    if (accepted) *accepted = s->n_accept;
}

extern "C" void basi_spec_free(basi_spec *s) {
    if (!s) return;
    if (s->batch_tgt.token || s->batch_tgt.embd) llama_batch_free(s->batch_tgt);
    if (s->smpl) common_sampler_free(s->smpl);
    if (s->spec) common_speculative_free(s->spec);
    // ctx_dft is owned by spec_init (unique_ptr) — freed on destruct
    delete s;
}

/* ---- self-test: plain-greedy vs spec-greedy A/B on the same tokens ---------- */

static std::vector<llama_token> plain_greedy(llama_context *ctx, llama_model *model,
                                             const std::vector<llama_token> &prompt, int max_gen) {
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> out;
    llama_batch pb = llama_batch_get_one(const_cast<llama_token *>(prompt.data()), (int) prompt.size());
    fprintf(stderr, "[dbg] plain: decoding prompt (%zu toks)...\n", prompt.size());
    if (llama_decode(ctx, pb) != 0) { fprintf(stderr, "[dbg] plain: prompt decode failed\n"); return out; }
    fprintf(stderr, "[dbg] plain: prompt decoded, n_vocab=%d\n", n_vocab);
    int n_past = (int) prompt.size();
    for (int i = 0; i < max_gen; ++i) {
        const float *logits = llama_get_logits_ith(ctx, -1);
        if (!logits) { fprintf(stderr, "[dbg] plain: logits NULL at i=%d\n", i); break; }
        llama_token best = 0; float bestv = logits[0];
        for (int t = 1; t < n_vocab; ++t) if (logits[t] > bestv) { bestv = logits[t]; best = t; }
        if (llama_vocab_is_eog(vocab, best)) break;
        out.push_back(best);
        llama_batch nb = llama_batch_get_one(&best, 1);
        if (llama_decode(ctx, nb) != 0) break;
        if ((uint32_t) ++n_past + 1 >= llama_n_ctx(ctx)) break;
    }
    return out;
}

extern "C" void basi_spec_selftest(llama_model *model, llama_context *ctx,
                                   const char *model_path, int n_ctx, int n_gpu_layers) {
    const char *type = getenv("BASI_SPEC");        if (!type || !*type) type = "draft-mtp";
    int n_max = 1;    { const char *e = getenv("BASI_SPEC_NMAX"); if (e && *e) n_max = atoi(e); }
    int max_gen = 128;{ const char *e = getenv("BASI_SPEC_N");    if (e && *e) max_gen = atoi(e); }
    const char *prompt_str = getenv("BASI_SPEC_PROMPT");
    if (!prompt_str || !*prompt_str)
        prompt_str = "Write a C function that reverses a string in place. Explain your approach first.";

    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n = -llama_tokenize(vocab, prompt_str, (int) strlen(prompt_str), NULL, 0, true, true);
    if (n <= 0) { fprintf(stderr, "[spec-selftest] tokenize failed\n"); return; }
    std::vector<llama_token> prompt(n);
    llama_tokenize(vocab, prompt_str, (int) strlen(prompt_str), prompt.data(), n, true, true);

    fprintf(stderr, "\n=== spec self-test: type=%s n_max=%d gen=%d prompt_toks=%d ===\n",
            type, n_max, max_gen, n);

    // plain greedy reference
    fprintf(stderr, "[dbg] clearing memory before plain...\n");
    llama_memory_clear(llama_get_memory(ctx), true);
    fprintf(stderr, "[dbg] memory cleared, starting plain greedy...\n");
    int64_t t0 = ggml_time_us();
    std::vector<llama_token> A = plain_greedy(ctx, model, prompt, max_gen);
    fprintf(stderr, "[dbg] plain greedy done (%zu toks)\n", A.size());
    double ta = (ggml_time_us() - t0) / 1e6;

    // spec greedy
    llama_memory_clear(llama_get_memory(ctx), true);
    basi_spec *s = basi_spec_init(model, ctx, model_path, n_ctx, n_gpu_layers, type, n_max, 0, 0.0f);
    if (!s) { fprintf(stderr, "[spec-selftest] basi_spec_init returned NULL (type unsupported / no head?)\n"); return; }
    std::vector<llama_token> B;
    int64_t t1 = ggml_time_us();
    basi_spec_run(s, prompt.data(), (int) prompt.size(), max_gen,
                  [](llama_token tok, int eog, void *ud) {
                      if (!eog) ((std::vector<llama_token> *) ud)->push_back(tok);
                  }, &B);
    double tb = (ggml_time_us() - t1) / 1e6;

    long drafted = 0, accepted = 0; basi_spec_stats(s, &drafted, &accepted);

    // compare
    bool identical = (A.size() == B.size());
    size_t diverge = A.size();
    if (identical) {
        for (size_t i = 0; i < A.size(); ++i) if (A[i] != B[i]) { identical = false; diverge = i; break; }
    } else {
        size_t m = A.size() < B.size() ? A.size() : B.size();
        for (size_t i = 0; i < m; ++i) if (A[i] != B[i]) { diverge = i; break; }
    }

    fprintf(stderr, "plain : %zu tok  %.2f s  %.2f tok/s\n", A.size(), ta, A.size() / (ta > 0 ? ta : 1));
    fprintf(stderr, "spec  : %zu tok  %.2f s  %.2f tok/s   (accept %ld/%ld = %.1f%%)\n",
            B.size(), tb, B.size() / (tb > 0 ? tb : 1),
            accepted, drafted, drafted ? 100.0 * accepted / drafted : 0.0);
    fprintf(stderr, "speedup: %.2fx    LOSSLESS: %s%s\n",
            (ta > 0 && tb > 0) ? ta / tb : 0.0,
            identical ? "YES (byte-identical)" : "NO",
            identical ? "" : "  (first divergence at token index shown below)");
    if (!identical) fprintf(stderr, "  diverge@%zu\n", diverge);

    // dump first tokens of each for diagnosis
    auto dump = [&](const char *lbl, const std::vector<llama_token> &v) {
        fprintf(stderr, "%s ids:", lbl);
        for (size_t i = 0; i < v.size() && i < 10; ++i) fprintf(stderr, " %d", v[i]);
        fprintf(stderr, "\n%s txt: '", lbl);
        for (size_t i = 0; i < v.size() && i < 20; ++i) { char b[128]; int m = llama_token_to_piece(vocab, v[i], b, sizeof b, 0, false); if (m > 0) fwrite(b, 1, m, stderr); }
        fprintf(stderr, "'\n");
    };
    dump("A(plain)", A);
    dump("B(spec) ", B);

    basi_spec_free(s);
}
