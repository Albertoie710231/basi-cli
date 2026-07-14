#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <termios.h>
#include <poll.h>
#include <signal.h>
#include <time.h>
#include <errno.h>

#include "llama.h"
#include "ggml-backend.h"

#include "util.h"
#include "globals.h"
#include "model.h"
#include "hwinfo.h"
#include "chat_tmpl.h"
#include "md.h"
#include "srvgen.h"
#include "srvchat.h"

void model_init(void) {
    extern void log_callback(enum ggml_log_level level, const char *text, void *user_data);
    llama_log_set(log_callback, NULL);
    ggml_backend_load_all();
}

/* ── Spinner frames ────────────────────────────────────────────────── */

static const char *spinner_frames[] = {
    "\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9", "\xe2\xa0\xb8",
    "\xe2\xa0\xbc", "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7",
    "\xe2\xa0\x87", "\xe2\xa0\x8f"
};
#define SPINNER_COUNT 10
/* ── Extract <tool>...</tool> from text ────────────────────────────── */

const char *extract_tool_call(const char *text, size_t *out_len) {
    const char *start = strstr(text, "<tool>");
    if (!start) return NULL;
    start += 6; /* strlen("<tool>") */
    const char *end = strstr(start, "</tool>");
    if (!end) return NULL;
    *out_len = end - start;
    return start;
}

/* ── Thinking animation ────────────────────────────────────────────── */

static void draw_thinking_box(size_t frame) {
    if (generate_quiet) return;
    const char *spinner = spinner_frames[frame % SPINNER_COUNT];
    printf("\r\033[K\033[90m\xe2\x94\x8c\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x90\033[0m\r\n");
    printf("\033[90m\xe2\x94\x82\033[0m \033[36m%s thinking...\033[0m   "
           "\033[90m\xe2\x94\x82\033[0m\r\n", spinner);
    printf("\033[90m\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x98\033[0m");
    printf("\033[2A");
    fflush(stdout);
}

static void clear_thinking_box(void) {
    if (generate_quiet) return;
    printf("\r\033[K\n\033[K\n\033[K\033[2A\r");
    fflush(stdout);
}

/* ── UTF-8 helpers ─────────────────────────────────────────────────── */

static size_t utf8_seq_len(unsigned char b) {
    if ((b & 0x80) == 0)    return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

/* ── Thinking state machine ────────────────────────────────────────── */

typedef enum {
    STATE_NORMAL,
    STATE_MAYBE_OPEN,
    STATE_THINKING,
    STATE_MAYBE_CLOSE,
    STATE_SUPPRESS,     /* inside a native tool call: kept in text, hidden from display */
} ThinkingState;

/* ── Generate response ─────────────────────────────────────────────── */


/* ── Server-backed generation (Pi-style) ─────────────────────────────────────
 * When basi_srv_port>0, generate() delegates token generation to a spawned
 * llama-server over its /completion SSE stream instead of decoding in-process.
 * Set by main.c after srvgen_spawn(). basi_srv_model is a vocab_only handle used
 * to derive the tool grammar. */
int basi_srv_port = 0;
const struct llama_model *basi_srv_model = NULL;
/* Defaults mirror BASI's native sampler chain (main.c overwrites in server mode):
   temp 0.4, repeat_penalty 1.1 over 256 tokens, min_p 0.05, top_k/top_p/seed off. */
SrvSampling basi_srv_sampling = { .temperature = 0.4, .repeat_penalty = 1.1, .repeat_last_n = 256,
                                  .min_p = 0.05, .top_k = 0, .top_p = 1.0, .seed = -1 };
/* When set, generate_server() omits the tool-call grammar. deepsearch drives its
   own ReAct loop with a private sampler chain that (in native mode) carries no
   grammar; this flag reproduces that on the server path so the main tool grammar
   can't leak into deepsearch's rounds or its final synthesis. */
int basi_srv_suppress_grammar = 0;

/* Remove <think>…</think> spans from a response (unless generate_keep_think).
 * Returns a malloc'd copy. */
static char *strip_thinking_dup(const char *s) {
    if (generate_keep_think) return strdup(s);
    const char *open = "<think>", *close = "</think>";
    const char *o = NULL, *c = NULL;
    if (basi_thinking_tags(&o, &c) && o && c && *o && *c) { open = o; close = c; }
    size_t olen = strlen(open), clen = strlen(close);
    StringBuf out; sb_init(&out);
    const char *p = s;
    while (*p) {
        const char *t = strstr(p, open);
        if (!t) { sb_append_str(&out, p); break; }
        sb_append(&out, p, (size_t)(t - p));
        const char *e = strstr(t + olen, close);
        if (!e) break;                 /* unterminated → drop the trailing think */
        p = e + clen;
    }
    return sb_to_str(&out);
}

/* ── Server-path streaming display filter ────────────────────────────────────
 * The server streams raw /completion content — thinking and tool-call markup
 * included. This runs the SAME hide-thinking / hide-tool-markup state machine as
 * native generate() over the SSE chunks, so the live display matches: the
 * reasoning span collapses to a spinner box (or dim [thinking] text under
 * Ctrl+T), native tool-call markup is suppressed, and answer text is
 * markdown-rendered. Display only — the returned/stored text is the full
 * accumulation srvgen_complete() builds independently (markup intact for the
 * parser; strip_thinking_dup() drops the reasoning). */
enum { SRV_OPEN_THINK = 1, SRV_OPEN_TOOLCALL = 2 };

typedef struct {
    ThinkingState state;
    char   tag_buf[64];
    size_t tag_len;
    struct { const char *s; size_t len; int act; } openers[3];
    int    n_openers;
    char   think_open_buf[48], think_close_buf[48];
    const char *think_open, *think_close;
    size_t think_open_len, think_close_len;
    int    md;
    size_t spinner_frame;
    double last_spinner;
    bool   thinking_box_shown;
    unsigned char utf8_buf[4];
    size_t utf8_len;
    bool   stdin_is_tty;
} SrvDisplay;

/* Reveal / tear down the thinking box (or its dim [thinking] header). */
static void srv_think_open_ui(SrvDisplay *d) {
    if (generate_quiet) { d->thinking_box_shown = true; return; }
    if (show_thinking) { printf("\033[90m[thinking] "); fflush(stdout); }
    else { draw_thinking_box(d->spinner_frame); d->last_spinner = time_now(); }
    d->thinking_box_shown = true;
}
static void srv_think_close_ui(SrvDisplay *d) {
    if (!d->thinking_box_shown) return;
    if (!generate_quiet) { if (show_thinking) printf("\033[0m\n"); else clear_thinking_box(); fflush(stdout); }
    d->thinking_box_shown = false;
}

/* Emit answer text: through the markdown renderer, or raw with a colour reset. */
static void srv_emit_normal(SrvDisplay *d, const char *p, size_t n) {
    if (n == 0) return;
    if (d->md) md_feed(p, n);
    else if (!generate_quiet) { printf("\033[0m"); fwrite(p, 1, n, stdout); fflush(stdout); }
}

static void srv_display_init(SrvDisplay *d, const char *prompt, size_t prompt_len) {
    memset(d, 0, sizeof *d);
    d->state = STATE_NORMAL;
    d->md = (generate_markdown && !generate_quiet) ? 1 : 0;
    d->stdin_is_tty = isatty(STDIN_FILENO);

    /* Reasoning delimiters from THIS model's template (falls back to <think>). */
    d->think_open = "<think>"; d->think_close = "</think>";
    const char *o = NULL, *c = NULL;
    if (basi_thinking_tags(&o, &c) && o && c &&
        strlen(o) > 0 && strlen(o) < sizeof d->think_open_buf &&
        strlen(c) > 0 && strlen(c) < sizeof d->think_close_buf) {
        strcpy(d->think_open_buf,  o); d->think_open  = d->think_open_buf;
        strcpy(d->think_close_buf, c); d->think_close = d->think_close_buf;
    }
    d->think_open_len  = strlen(d->think_open);
    d->think_close_len = strlen(d->think_close);

    d->openers[d->n_openers].s = d->think_open; d->openers[d->n_openers].len = d->think_open_len;
    d->openers[d->n_openers].act = SRV_OPEN_THINK; d->n_openers++;
    if (generate_native_tools) {
        d->openers[d->n_openers].s = "<tool_call>";  d->openers[d->n_openers].len = 11; d->openers[d->n_openers].act = SRV_OPEN_TOOLCALL; d->n_openers++;
        d->openers[d->n_openers].s = "<|tool_call>"; d->openers[d->n_openers].len = 12; d->openers[d->n_openers].act = SRV_OPEN_TOOLCALL; d->n_openers++;
    }

    /* Forced-open thinking: Qwen3.x-style templates inject the opening think tag
       into the PROMPT, so the model streams only the reasoning CONTENT and then
       the close tag. If the rendered prompt ends with the open tag, start INSIDE
       the thinking block (exactly as if we'd just matched it). */
    if (d->think_open_len > 0 && prompt) {
        size_t pl = prompt_len;
        while (pl > 0 && (prompt[pl-1]=='\n' || prompt[pl-1]=='\r' ||
                          prompt[pl-1]==' '  || prompt[pl-1]=='\t')) pl--;
        if (pl >= d->think_open_len &&
            memcmp(prompt + pl - d->think_open_len, d->think_open, d->think_open_len) == 0) {
            d->state = STATE_THINKING;
            srv_think_open_ui(d);
        }
    }
}

/* srvgen_complete() emit callback: run one SSE content chunk through the state
   machine. ud is the SrvDisplay*. */
static void srv_display_feed(const char *chunk, void *ud) {
    SrvDisplay *d = (SrvDisplay *) ud;
    size_t n = strlen(chunk);

    /* Ctrl+T toggles the thinking display (box <-> dim text), like native. Only
       on an interactive TTY (a pipe would steal the next prompt's bytes). */
    if (!generate_quiet && d->stdin_is_tty) {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
        if (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) {
            unsigned char key;
            if (read(STDIN_FILENO, &key, 1) == 1 && key == 0x14) {   /* Ctrl+T */
                show_thinking = !show_thinking;
                if (d->state == STATE_THINKING || d->state == STATE_MAYBE_CLOSE) {
                    if (show_thinking) {
                        if (d->thinking_box_shown) { clear_thinking_box(); }
                        printf("\033[90m[thinking] "); fflush(stdout);
                        d->thinking_box_shown = true;
                    } else {
                        printf("\033[0m\n");
                        draw_thinking_box(d->spinner_frame);
                        d->thinking_box_shown = true;
                        d->last_spinner = time_now();
                    }
                }
            }
        }
    }

    size_t piece_start = 0;
    for (size_t idx = 0; idx < n; idx++) {
        char ch = chunk[idx];
        switch (d->state) {
        case STATE_NORMAL:
            if (ch == '<') {
                if (idx > piece_start) srv_emit_normal(d, chunk + piece_start, idx - piece_start);
                d->state = STATE_MAYBE_OPEN;
                d->tag_len = 0;
                d->tag_buf[d->tag_len++] = ch;
                piece_start = idx + 1;
            }
            break;

        case STATE_MAYBE_OPEN: {
            if (d->tag_len < sizeof d->tag_buf) d->tag_buf[d->tag_len++] = ch;
            int matched = 0; bool alive = false;
            for (int oi = 0; oi < d->n_openers; oi++) {
                if (d->tag_len <= d->openers[oi].len &&
                    memcmp(d->tag_buf, d->openers[oi].s, d->tag_len) == 0) {
                    alive = true;
                    if (d->tag_len == d->openers[oi].len) { matched = d->openers[oi].act; break; }
                }
            }
            if (matched == SRV_OPEN_THINK) {
                d->state = STATE_THINKING; d->tag_len = 0; piece_start = idx + 1;
                srv_think_open_ui(d);
            } else if (matched == SRV_OPEN_TOOLCALL) {
                if (d->md) md_end();   /* close the answer before the hidden markup */
                d->state = STATE_SUPPRESS; d->tag_len = 0; piece_start = idx + 1;
            } else if (!alive) {
                srv_emit_normal(d, d->tag_buf, d->tag_len);
                d->state = STATE_NORMAL; d->tag_len = 0; piece_start = idx + 1;
            }
            /* else: still a viable prefix of some opener — keep buffering */
            break;
        }

        case STATE_SUPPRESS:
            /* Native tool-call markup: kept in the returned text (parser needs
               it), never displayed. Stays hidden until generation ends. */
            piece_start = idx + 1;
            break;

        case STATE_THINKING:
            if (ch == '<') {
                d->state = STATE_MAYBE_CLOSE; d->tag_len = 0; d->tag_buf[d->tag_len++] = ch;
            } else {
                if (show_thinking) { if (!generate_quiet) { printf("\033[90m%c", ch); fflush(stdout); } }
                else {
                    double now = time_now();
                    if (now - d->last_spinner > 0.08) {
                        d->spinner_frame++; draw_thinking_box(d->spinner_frame); d->last_spinner = now;
                    }
                }
            }
            piece_start = idx + 1;
            break;

        case STATE_MAYBE_CLOSE:
            if (d->tag_len < sizeof d->tag_buf) d->tag_buf[d->tag_len++] = ch;
            if (d->tag_len <= d->think_close_len && d->tag_buf[d->tag_len-1] == d->think_close[d->tag_len-1]) {
                if (d->tag_len == d->think_close_len) {
                    d->state = STATE_NORMAL; d->tag_len = 0; piece_start = idx + 1;
                    srv_think_close_ui(d);
                }
            } else {
                /* '<…' inside the reasoning, not the close tag — keep it hidden */
                if (show_thinking && !generate_quiet) {
                    printf("\033[90m"); fwrite(d->tag_buf, 1, d->tag_len, stdout); fflush(stdout);
                }
                d->state = STATE_THINKING; d->tag_len = 0; piece_start = idx + 1;
            }
            break;
        }
    }

    /* Flush the normal-state tail, holding back only a trailing incomplete UTF-8
       sequence so a chunk boundary can't split a multibyte char. Server chunks
       are normally already whole-UTF-8, so utf8_len stays 0 in practice. */
    if (d->state == STATE_NORMAL && piece_start < n) {
        const unsigned char *slice = (const unsigned char *) chunk + piece_start;
        size_t slice_len = n - piece_start;

        /* Complete a sequence left dangling from the previous chunk first. */
        if (d->utf8_len > 0) {
            size_t need = utf8_seq_len(d->utf8_buf[0]);
            while (d->utf8_len < need && slice_len > 0) { d->utf8_buf[d->utf8_len++] = *slice++; slice_len--; }
            if (d->utf8_len == need) { srv_emit_normal(d, (const char *) d->utf8_buf, need); d->utf8_len = 0; }
            else return;   /* still incomplete — wait for the next chunk */
        }

        size_t boundary = 0, pos = 0;
        while (pos < slice_len) {
            size_t slen = utf8_seq_len(slice[pos]);
            if (pos + slen <= slice_len) { boundary = pos + slen; pos += slen; }
            else break;
        }
        if (boundary > 0) srv_emit_normal(d, (const char *) slice, boundary);
        if (boundary < slice_len) {
            size_t rem = slice_len - boundary;
            if (rem > sizeof d->utf8_buf) rem = sizeof d->utf8_buf;   /* never overflow */
            memcpy(d->utf8_buf, slice + boundary, rem);
            d->utf8_len = rem;
        }
    }
}

/* Flush any buffered UTF-8 and close an open thinking box at end of stream. */
static void srv_display_finish(SrvDisplay *d) {
    if (d->utf8_len > 0) { srv_emit_normal(d, (const char *) d->utf8_buf, d->utf8_len); d->utf8_len = 0; }
    if (d->thinking_box_shown) srv_think_close_ui(d);
}

static GenerateResult generate_server(const char *prompt) {
    GenerateResult res = { NULL, 0, 0, 0.0, 0.0 };

    char *frag = (basi_srv_model && !basi_srv_suppress_grammar)
                     ? basi_tool_grammar_json(basi_srv_model) : NULL;

    long cap = 0; { const char *e = getenv("BASI_MAX_TOKENS"); if (e && *e) cap = atol(e); }
    int n_predict = cap > 0 ? (int) cap : -1;

    SrvDisplay disp;
    srv_display_init(&disp, prompt, strlen(prompt));

    if (disp.md) md_begin();
    double t0 = time_now();
    double tps = 0; int ntok = 0;
    char *full = srvgen_complete(basi_srv_port, prompt, n_predict, &basi_srv_sampling,
                                 frag, srv_display_feed, &disp, &tps, &ntok);
    srv_display_finish(&disp);
    if (disp.md) md_end();
    if (!generate_quiet) { printf("\033[0m\n"); fflush(stdout); }
    free(frag);

    res.gen_time_s = time_now() - t0;
    res.gen_tokens = (size_t) ntok;
    res.text = full ? strip_thinking_dup(full) : strdup("[server generation failed]");
    free(full);
    return res;
}

/* ── Chat-completions path (item 6b): server templates from messages, owns the
 * tool grammar, and returns STRUCTURED tool_calls + separated reasoning. The
 * display is simpler than the /completion state machine — reasoning arrives on a
 * distinct stream, so no <think> tag parsing: reasoning → thinking box (or dim
 * text under Ctrl+T), content → the markdown answer. ─────────────────────────── */
typedef struct {
    int    md;
    bool   thinking_box_shown;
    bool   md_started;
    size_t spinner_frame;
    double last_spinner;
} ChatDisplay;

static void chat_on_reasoning(const char *chunk, void *ud) {
    ChatDisplay *d = (ChatDisplay *) ud;
    if (generate_quiet) return;
    if (show_thinking) {
        printf("\033[90m%s\033[0m", chunk); fflush(stdout);
        d->thinking_box_shown = true;   /* so a newline is emitted when it closes */
    } else {
        double now = time_now();
        if (!d->thinking_box_shown || now - d->last_spinner > 0.08) {
            d->spinner_frame++; draw_thinking_box(d->spinner_frame); d->last_spinner = now;
            d->thinking_box_shown = true;
        }
    }
}

static void chat_on_content(const char *chunk, void *ud) {
    ChatDisplay *d = (ChatDisplay *) ud;
    if (generate_quiet) return;
    if (d->thinking_box_shown) {   /* reasoning finished — tear the box down first */
        if (show_thinking) printf("\033[0m\n"); else clear_thinking_box();
        d->thinking_box_shown = false;
    }
    if (d->md) {
        if (!d->md_started) { md_begin(); d->md_started = true; }   /* open the answer stream once */
        md_feed(chunk, strlen(chunk));
    } else { printf("\033[0m"); fputs(chunk, stdout); fflush(stdout); }
}

/* Server-chat generation. Serializes `messages` (+ registered tools) to OpenAI
 * JSON, streams /v1/chat/completions, and returns the answer text plus STRUCTURED
 * tool calls in tc_out/n_tc_out (caller frees via basi_free_tool_calls).
 * res.prompt_tokens carries the server's exact prompt count for ctx accounting. */
GenerateResult generate_chat(const struct llama_chat_message *messages, size_t msg_count,
                             BasiToolCall **tc_out, int *n_tc_out) {
    GenerateResult res = { NULL, 0, 0, 0.0, 0.0 };
    if (tc_out) *tc_out = NULL;
    if (n_tc_out) *n_tc_out = 0;

    char *mj = basi_messages_to_json(messages, (int) msg_count);
    if (!mj) { res.text = strdup("[chat serialization failed]"); return res; }
    /* deepsearch clears g_tools for its own ReAct loop → basi_tools_to_json()
       returns NULL there, which is exactly what we want (no tools advertised). */
    char *tj = basi_tools_to_json();

    long cap = 0; { const char *e = getenv("BASI_MAX_TOKENS"); if (e && *e) cap = atol(e); }
    int n_predict = cap > 0 ? (int) cap : -1;

    ChatDisplay disp = { (generate_markdown && !generate_quiet) ? 1 : 0, false, false, 0, 0.0 };
    double t0 = time_now();
    SrvChatResult *r = srvchat_complete(basi_srv_port, mj, tj, &basi_srv_sampling, n_predict,
                                        chat_on_content, chat_on_reasoning, &disp);
    free(mj); free(tj);

    if (!r) {
        if (disp.thinking_box_shown && !generate_quiet) { if (show_thinking) printf("\033[0m\n"); else clear_thinking_box(); }
        if (disp.md && disp.md_started) md_end();
        res.text = strdup("[server chat failed]");
        return res;
    }

    if (getenv("BASI_DEBUG_CHAT"))
        fprintf(stderr, "\n[chat] finish=%s content=%zub reasoning=%zub tool_calls=%d\n  reasoning=[%.400s]\n  content=[%.200s]\n",
                r->finish_reason ? r->finish_reason : "?",
                r->content ? strlen(r->content) : 0,
                r->reasoning ? strlen(r->reasoning) : 0, r->n_tool_calls,
                r->reasoning ? r->reasoning : "", r->content ? r->content : "");

    /* Qwen3.x sometimes keeps a brief final answer INSIDE the reasoning stream
       (thinks the answer, closes, stops) → content comes back empty. When there's
       no answer content and no tool call, promote the reasoning to the answer so
       the turn isn't blank; it was shown live as a hidden box, so reveal it now. */
    bool answer_in_reasoning = (!r->content || !r->content[0]) && r->n_tool_calls == 0
                               && r->reasoning && r->reasoning[0];
    const char *answer = answer_in_reasoning ? r->reasoning : (r->content ? r->content : "");

    if (disp.thinking_box_shown && !generate_quiet) {   /* tear the reasoning box down */
        if (show_thinking) printf("\033[0m\n"); else clear_thinking_box();
        disp.thinking_box_shown = false;
    }
    if (answer_in_reasoning && !generate_quiet) {        /* box hid the answer — reveal it */
        if (disp.md) { md_begin(); md_feed(answer, strlen(answer)); md_end(); }
        else { printf("\033[0m"); fputs(answer, stdout); }
    } else if (disp.md && disp.md_started) {
        md_end();                                        /* close the answer stream */
    }
    if (!generate_quiet) { printf("\033[0m\n"); fflush(stdout); }

    res.text          = strdup(answer);
    res.prompt_tokens = (size_t) r->prompt_tokens;
    res.gen_tokens    = (size_t) r->completion_tokens;
    res.gen_time_s    = (r->tps > 0) ? r->completion_tokens / r->tps : (time_now() - t0);

    if (r->n_tool_calls > 0 && tc_out) {
        BasiToolCall *arr = calloc((size_t) r->n_tool_calls, sizeof(BasiToolCall));
        if (arr) {
            for (int i = 0; i < r->n_tool_calls; i++) {
                arr[i].name      = strdup(r->tool_calls[i].name      ? r->tool_calls[i].name      : "");
                arr[i].arguments = strdup(r->tool_calls[i].arguments ? r->tool_calls[i].arguments : "{}");
            }
            *tc_out = arr;
            if (n_tc_out) *n_tc_out = r->n_tool_calls;
        }
    }
    srvchat_free(r);
    return res;
}

GenerateResult generate(
    struct llama_context *ctx,
    const struct llama_vocab *vocab,
    struct llama_sampler *smpl,
    const char *prompt,
    size_t prompt_len)
{
    if (basi_srv_port > 0) return generate_server(prompt);

    GenerateResult res = { NULL, 0, 0, 0.0, 0.0 };
    StringBuf response;
    sb_init(&response);

    /* Live markdown rendering of the answer stream (interactive TTY only). When
       on, visible answer text is routed through md_feed() instead of raw
       printf; md_end() closes it at the end / before tool markup. */
    const bool md = generate_markdown && !generate_quiet;
    if (md) md_begin();

    /* Check if first generation */
    llama_memory_t memory = llama_get_memory(ctx);
    bool is_first = (llama_memory_seq_pos_max(memory, 0) == -1);

    /* Defensive backstop: an underflowed delta (prev_len > render) yields a huge
       prompt_len whose (int) cast crashes llama_tokenize with std::length_error.
       Callers guard the delta, but never let a bogus length abort the process. */
    if (prompt_len == 0 || prompt_len > (size_t)(64 * 1024 * 1024)) {
        res.text = strdup("[Invalid prompt length]");
        return res;
    }

    /* Tokenize */
    int n_prompt_tokens = -llama_tokenize(vocab, prompt, (int)prompt_len,
                                           NULL, 0, is_first, true);
    if (n_prompt_tokens <= 0) {
        res.text = strdup("[Tokenization failed]");
        return res;
    }

    llama_token *tokens = malloc(n_prompt_tokens * sizeof(llama_token));
    llama_tokenize(vocab, prompt, (int)prompt_len,
                   tokens, n_prompt_tokens, is_first, true);

    res.prompt_tokens = n_prompt_tokens;

    /* Create batch */
    struct llama_batch batch = llama_batch_get_one(tokens, n_prompt_tokens);
    bool is_prompt_phase = true;
    double timer_start = time_now();

    /* Thinking state */
    ThinkingState state = STATE_NORMAL;
    char tag_buf[64];
    size_t tag_len = 0;

    /* Reasoning delimiters: use the ones common_chat derives from THIS model's
       template, so Gemma's "<|channel>thought"/"<channel|>", Qwen's
       "<think>"/"</think>", etc. are all hidden automatically — falling back to
       <think>/</think> when the model declares none. Copied locally so the
       pointers stay valid for the whole generation, and length-bounded so they
       can never overflow tag_buf. */
    const char *think_open = "<think>", *think_close = "</think>";
    char think_open_buf[48], think_close_buf[48];
    {
        const char *o = NULL, *c = NULL;
        if (basi_thinking_tags(&o, &c) && o && c &&
            strlen(o) > 0 && strlen(o) < sizeof(think_open_buf) &&
            strlen(c) > 0 && strlen(c) < sizeof(think_close_buf)) {
            strcpy(think_open_buf, o);
            strcpy(think_close_buf, c);
            think_open  = think_open_buf;
            think_close = think_close_buf;
        }
    }
    const size_t think_open_len  = strlen(think_open);
    const size_t think_close_len = strlen(think_close);

    /* Openers the STATE_MAYBE_OPEN matcher recognizes: the reasoning tag (→ hide
       as a thinking box) plus, in native tool mode, the tool-call markers (→
       hide the raw JSON; the [Executing:] line is the clean indicator). The
       tool-call markers are a display-only heuristic covering the families in
       use — a miss just means the markup shows, never a parse/behaviour change. */
    enum { OPEN_THINK = 1, OPEN_TOOLCALL = 2 };
    struct { const char *s; size_t len; int act; } openers[3];
    int n_openers = 0;
    openers[n_openers].s = think_open; openers[n_openers].len = think_open_len; openers[n_openers].act = OPEN_THINK; n_openers++;
    if (generate_native_tools) {
        openers[n_openers].s = "<tool_call>";  openers[n_openers].len = 11; openers[n_openers].act = OPEN_TOOLCALL; n_openers++;
        openers[n_openers].s = "<|tool_call>"; openers[n_openers].len = 12; openers[n_openers].act = OPEN_TOOLCALL; n_openers++;
    }
    size_t spinner_frame = 0;
    double last_spinner = 0;
    bool thinking_box_shown = false;

    /* UTF-8 buffer */
    unsigned char utf8_buf[4];
    size_t utf8_len = 0;

    /* Only poll stdin for Ctrl+T when it's an interactive terminal. On piped /
       scripted stdin there is always input pending, so the poll-and-read below
       would steal bytes from the next prompt — corrupting multi-turn scripted
       sessions. You also can't press Ctrl+T through a pipe, so there's nothing
       to detect there. */
    const bool stdin_is_tty = isatty(STDIN_FILENO);

    /* Forced-open thinking: some templates (Qwen3.x) inject the opening think tag
       into the generation PROMPT, so the model emits only the thinking CONTENT
       and then the closing tag — the open tag never appears in the output. Left
       alone the state machine stays NORMAL and the reasoning LEAKS as visible
       text (and isn't stripped from the stored turn). If the rendered prompt ends
       with the think-open tag (modulo trailing whitespace), start INSIDE the
       thinking block, exactly as if we'd just matched the open tag. */
    bool forced_open_think = false;
    if (think_open_len > 0) {
        size_t pl = prompt_len;
        while (pl > 0 && (prompt[pl-1] == '\n' || prompt[pl-1] == '\r' ||
                          prompt[pl-1] == ' '  || prompt[pl-1] == '\t')) pl--;
        if (pl >= think_open_len &&
            memcmp(prompt + pl - think_open_len, think_open, think_open_len) == 0) {
            state = STATE_THINKING;
            if (generate_keep_think) sb_append_str(&response, think_open);
            /* Reveal the box AFTER prefill (in the is_prompt_phase block below),
               not here — drawing it before the loop would freeze the spinner on
               frame 0 for the entire prompt decode (which blocks the loop). */
            forced_open_think = true;
        }
    }

    /* Optional hard cap on generated tokens (env BASI_MAX_TOKENS; 0/unset =
       unlimited, preserving default behavior). Without it, a model that never
       emits end-of-turn generates until the context fills — this bounds
       non-interactive and benchmark runs. */
    long gen_cap = 0;
    { const char *e = getenv("BASI_MAX_TOKENS"); if (e && *e) gen_cap = atol(e); }
    long n_generated = 0;

    /* Generation loop */
    while (1) {
        /* Check context space */
        uint32_t n_ctx = llama_n_ctx(ctx);
        uint32_t n_ctx_used = (uint32_t)(llama_memory_seq_pos_max(memory, 0) + 1);
        if (n_ctx_used + (uint32_t)batch.n_tokens > n_ctx) {
            if (thinking_box_shown) { if (!generate_quiet) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } } thinking_box_shown = false; }
            if (md) md_end();
            if (!generate_quiet) printf("\n[Context limit reached]\n");
            fflush(stdout);
            break;
        }

        if (llama_decode(ctx, batch) != 0) {
            if (thinking_box_shown) { if (!generate_quiet) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } } thinking_box_shown = false; }
            if (md) md_end();
            if (!generate_quiet) printf("\n[Decode error]\n");
            fflush(stdout);
            break;
        }

        if (is_prompt_phase) {
            res.prompt_time_s = time_now() - timer_start;
            timer_start = time_now();
            is_prompt_phase = false;
            /* Prefill is done — now reveal a forced-open thinking box (deferred
               from before the loop) so its spinner animates from the first
               generated token instead of sitting frozen through the prompt
               decode. The first token is already suppressed because state was
               set to STATE_THINKING up front. */
            if (forced_open_think && !thinking_box_shown && !generate_quiet) {
                if (show_thinking) { printf("\033[90m[thinking] "); fflush(stdout); }
                else { draw_thinking_box(spinner_frame); last_spinner = time_now(); }
                thinking_box_shown = true;
            }
        }

        /* Tick the pinned status bar so its ctx meter climbs live as the KV
           fills (throttled internally; no-op when the bar is inactive/quiet). */
        if (!generate_quiet) statusbar_tick();

        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) { if (!generate_quiet) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } } thinking_box_shown = false; }
            break;
        }

        if (gen_cap > 0 && ++n_generated >= gen_cap) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) { if (!generate_quiet) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } } thinking_box_shown = false; }
            if (md) md_end();
            if (!generate_quiet) { printf("\n[max tokens reached]\n"); fflush(stdout); }
            break;
        }

        if (generation_interrupted) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) { if (!generate_quiet) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } } thinking_box_shown = false; }
            if (md) md_end();
            if (!generate_quiet) printf("\n\033[90m[interrupted]\033[0m");
            fflush(stdout);
            break;
        }

        /* Check for Ctrl+T (toggle thinking display) via non-blocking read.
           Skipped when quiet (deepsearch internal rounds) or when stdin is not a
           TTY (piped/scripted — reading would eat the next prompt). */
        if (!generate_quiet && stdin_is_tty) {
            struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
            if (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) {
                unsigned char key;
                if (read(STDIN_FILENO, &key, 1) == 1 && key == 0x14) { /* Ctrl+T */
                    show_thinking = !show_thinking;
                    if (state == STATE_THINKING || state == STATE_MAYBE_CLOSE) {
                        if (show_thinking) {
                            /* Switching from box to text */
                            if (thinking_box_shown) { if (!generate_quiet) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } } thinking_box_shown = false; }
                            printf("\033[90m[thinking] ");
                            fflush(stdout);
                        } else {
                            /* Switching from text to box */
                            printf("\033[0m\n");
                            draw_thinking_box(spinner_frame);
                            thinking_box_shown = true;
                            last_spinner = time_now();
                        }
                    }
                }
            }
        }

        res.gen_tokens++;

        /* Token to text */
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n < 0) continue;

        /* Process through thinking state machine */
        size_t piece_start = 0;
        for (int idx = 0; idx < n; idx++) {
            char ch = buf[idx];
            switch (state) {
            case STATE_NORMAL:
                if (ch == '<') {
                    /* Flush text before '<' */
                    if ((size_t)idx > piece_start) {
                        if (md) md_feed(buf + piece_start, idx - piece_start);
                        else if (!generate_quiet) {
                            printf("\033[0m");
                            fwrite(buf + piece_start, 1, idx - piece_start, stdout);
                            fflush(stdout);
                        }
                        sb_append(&response, buf + piece_start, idx - piece_start);
                    }
                    state = STATE_MAYBE_OPEN;
                    tag_len = 0;
                    tag_buf[tag_len++] = ch;
                    piece_start = idx + 1;
                }
                break;

            case STATE_MAYBE_OPEN: {
                tag_buf[tag_len++] = ch;
                /* Match the accumulating tag against every opener; an opener is
                   "alive" while tag_buf is a prefix of it. */
                int matched = 0;       /* action of a fully-matched opener */
                bool alive = false;
                for (int oi = 0; oi < n_openers; oi++) {
                    if (tag_len <= openers[oi].len &&
                        memcmp(tag_buf, openers[oi].s, tag_len) == 0) {
                        alive = true;
                        if (tag_len == openers[oi].len) { matched = openers[oi].act; break; }
                    }
                }
                if (matched == OPEN_THINK) {
                    state = STATE_THINKING;
                    tag_len = 0;
                    piece_start = idx + 1;
                    if (generate_keep_think) sb_append_str(&response, think_open);
                    if (show_thinking) {
                        if (!generate_quiet) { printf("\033[90m[thinking] "); fflush(stdout); }
                    } else {
                        draw_thinking_box(spinner_frame);
                        last_spinner = time_now();
                    }
                    thinking_box_shown = true;
                } else if (matched == OPEN_TOOLCALL) {
                    /* Keep the markup in the returned text (the parser needs it)
                       but stop displaying it — the [Executing:] line is shown
                       after the call is parsed. */
                    sb_append(&response, tag_buf, tag_len);
                    if (md) md_end();   /* close the answer before the hidden tool markup */
                    state = STATE_SUPPRESS;
                    tag_len = 0;
                    piece_start = idx + 1;
                } else if (!alive) {
                    /* Matched no opener — flush the buffer as normal text. */
                    if (md) md_feed(tag_buf, tag_len);
                    else if (!generate_quiet) {
                        printf("\033[0m");
                        fwrite(tag_buf, 1, tag_len, stdout);
                        fflush(stdout);
                    }
                    sb_append(&response, tag_buf, tag_len);
                    state = STATE_NORMAL;
                    tag_len = 0;
                    piece_start = idx + 1;
                }
                /* else: still a viable prefix of some opener — keep buffering */
                break;
            }

            case STATE_SUPPRESS:
                /* Native tool-call markup: retained in `response` for parsing,
                   never displayed. Remains until the generation ends. */
                sb_append_char(&response, ch);
                piece_start = idx + 1;
                break;

            case STATE_THINKING: {
                if (ch == '<') {
                    state = STATE_MAYBE_CLOSE;
                    tag_len = 0;
                    tag_buf[tag_len++] = ch;
                } else {
                    if (generate_keep_think) sb_append_char(&response, ch);
                    if (show_thinking) {
                        if (!generate_quiet) { printf("\033[90m%c", ch); fflush(stdout); }
                    } else {
                        /* Spinner animation */
                        double now = time_now();
                        if (now - last_spinner > 0.08) {
                            spinner_frame++;
                            draw_thinking_box(spinner_frame);
                            last_spinner = now;
                        }
                    }
                }
                piece_start = idx + 1;
                break;
            }

            case STATE_MAYBE_CLOSE: {
                tag_buf[tag_len++] = ch;
                const char *target = think_close;
                if (tag_len <= think_close_len && tag_buf[tag_len - 1] == target[tag_len - 1]) {
                    if (tag_len == think_close_len) {
                        state = STATE_NORMAL;
                        tag_len = 0;
                        piece_start = idx + 1;
                        if (generate_keep_think) sb_append_str(&response, think_close);
                        if (show_thinking) {
                            if (!generate_quiet) printf("\033[0m\n");
                        } else {
                            clear_thinking_box();
                        }
                        fflush(stdout);
                        thinking_box_shown = false;
                    }
                } else {
                    if (generate_keep_think) sb_append(&response, tag_buf, tag_len);
                    if (show_thinking && !generate_quiet) {
                        printf("\033[90m");
                        fwrite(tag_buf, 1, tag_len, stdout);
                        fflush(stdout);
                    }
                    state = STATE_THINKING;
                    tag_len = 0;
                    piece_start = idx + 1;
                }
                break;
            }
            }
        }

        /* Output remaining content in normal state with UTF-8 buffering */
        if (state == STATE_NORMAL && piece_start < (size_t)n) {
            const char *slice = buf + piece_start;
            size_t slice_len = n - piece_start;

            unsigned char combined[260];
            size_t combined_len = 0;

            /* Prepend buffered UTF-8 bytes */
            memcpy(combined, utf8_buf, utf8_len);
            combined_len = utf8_len;
            utf8_len = 0;

            memcpy(combined + combined_len, slice, slice_len);
            combined_len += slice_len;

            /* Find complete UTF-8 boundary */
            size_t output_end = 0;
            size_t pos = 0;
            while (pos < combined_len) {
                size_t slen = utf8_seq_len(combined[pos]);
                if (pos + slen <= combined_len) {
                    output_end = pos + slen;
                    pos += slen;
                } else {
                    break;
                }
            }

            if (output_end > 0) {
                if (md) md_feed((const char *)combined, output_end);
                else if (!generate_quiet) {
                    printf("\033[0m");
                    fwrite(combined, 1, output_end, stdout);
                    fflush(stdout);
                }
                sb_append(&response, (const char *)combined, output_end);
            }

            /* Buffer incomplete trailing bytes */
            if (output_end < combined_len) {
                memcpy(utf8_buf, combined + output_end, combined_len - output_end);
                utf8_len = combined_len - output_end;
            }
        }

        /* Stop as soon as a complete <tool>...</tool> has been emitted, so the
           model can't chain dozens of speculative tool calls in one response.
           Gated to STATE_NORMAL so a "</tool>" inside a kept <think> block
           (generate_keep_think) can't trip it. */
        if (state == STATE_NORMAL && response.len >= 7 &&
            memcmp(response.data + response.len - 7, "</tool>", 7) == 0) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) {
                if (!generate_quiet) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } }
                thinking_box_shown = false;
            }
            break;
        }

        /* Next token */
        llama_token single = new_token;
        batch = llama_batch_get_one(&single, 1);
    }

    /* Flush remaining UTF-8 */
    if (utf8_len > 0) {
        if (md) md_feed((const char *)utf8_buf, utf8_len);
        else if (!generate_quiet) {
            printf("\033[0m");
            fwrite(utf8_buf, 1, utf8_len, stdout);
        }
        sb_append(&response, (const char *)utf8_buf, utf8_len);
    }

    if (md) md_end();   /* render any trailing partial line + close open spans */
    if (!generate_quiet) printf("\033[0m\n");
    fflush(stdout);

    free(tokens);
    res.text = sb_to_str(&response);
    return res;
}

/* ── Log callback (suppress non-errors) ────────────────────────────── */

void log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

/* ── Custom ChatML template ────────────────────────────────────────── */

/*
 * Format messages as ChatML. Same format for all models:
 *   <|im_start|>role\ncontent<|im_end|>\n
 *
 * If add_generation_prompt is true, appends <|im_start|>assistant\n
 * If buf is NULL, returns the required length without writing.
 */
static int apply_chatml(
    const struct llama_chat_message *msgs, size_t n_msgs,
    bool add_gen_prompt,
    char *buf, size_t buf_size)
{
    size_t total = 0;

    #define CHATML_WRITE(s, len) do { \
        if (buf && total + (len) < buf_size) \
            memcpy(buf + total, (s), (len)); \
        total += (len); \
    } while(0)
    #define CHATML_STR(s) CHATML_WRITE(s, strlen(s))

    for (size_t i = 0; i < n_msgs; i++) {
        CHATML_STR("<|im_start|>");
        CHATML_STR(msgs[i].role);
        CHATML_WRITE("\n", 1);
        if (msgs[i].content)
            CHATML_STR(msgs[i].content);
        CHATML_STR("<|im_end|>\n");
    }

    if (add_gen_prompt) {
        CHATML_STR("<|im_start|>assistant\n");
    }

    #undef CHATML_WRITE
    #undef CHATML_STR

    if (buf && total < buf_size) buf[total] = '\0';
    return (int)total;
}

/*
 * Apply chat template with fallback. The C-API llama_chat_apply_template only
 * matches a hardcoded list of templates (no Jinja parser). Modern HF GGUFs ship
 * custom Jinja templates that it rejects with -1. When that happens, fall back
 * to plain ChatML, which is the de-facto format for Qwen/Phi/etc.
 */
int apply_template(
    const struct llama_model *model,
    const struct llama_chat_message *msgs, size_t n_msgs,
    bool add_gen_prompt,
    char *buf, size_t buf_size)
{
    bool dbg = getenv("BASI_DEBUG_TEMPLATE") != NULL;

    /* 1) Native: render the model's actual chat template via the jinja engine.
          This drives every model in its real format (Gemma/DeepSeek/custom). */
    char *rendered = basi_render_chat(model, msgs, n_msgs, add_gen_prompt);
    if (rendered) {
        size_t len = strlen(rendered);
        if (dbg) fprintf(stderr, "[tmpl] native jinja render: %zu bytes\n", len);
        if (!buf) { free(rendered); return (int)len; }       /* length-only query */
        if (len < buf_size) { memcpy(buf, rendered, len + 1); free(rendered); return (int)len; }
        free(rendered);   /* doesn't fit this buffer — fall through to legacy */
        if (dbg) fprintf(stderr, "[tmpl] rendered prompt too big for buffer; falling back\n");
    }

    /* 2) Legacy fallback: llama.cpp's C-API detection, then ChatML. */
    const char *tmpl = model ? llama_model_chat_template(model, NULL) : NULL;
    if (tmpl) {
        int r = llama_chat_apply_template(tmpl, msgs, n_msgs, add_gen_prompt, buf, buf_size);
        if (dbg) fprintf(stderr, "[tmpl] fallback llama_chat_apply_template -> %d\n", r);
        if (r >= 0) return r;
    }
    return apply_chatml(msgs, n_msgs, add_gen_prompt, buf, buf_size);
}

/* ── Model picker ──────────────────────────────────────────────────── */

#define MODEL_DIRS_MAX 4
static const char *model_search_dirs[] = {
    NULL, /* [0] filled at runtime: ~/.cache/huggingface/hub */
    NULL, /* [1] filled at runtime: $HOME/models (where /cookbook downloads) */
    ".",
    NULL
};

/* Resolve the two $HOME-relative search dirs at runtime: [0] the HF hub cache,
 * [1] the ~/models dir /cookbook downloads into. No-op if $HOME is unset. */
static void init_model_search_dirs(void) {
    static char cache_dir[512];
    static char models_dir[512];
    const char *home = getenv("HOME");
    if (!home) return;
    snprintf(cache_dir,  sizeof(cache_dir),  "%s/.cache/huggingface/hub", home);
    snprintf(models_dir, sizeof(models_dir), "%s/models", home);
    model_search_dirs[0] = cache_dir;
    model_search_dirs[1] = models_dir;
}

/* GGUF arch metadata used to size offload + KV cache. */
typedef struct {
    int n_ctx_train;   /* <arch>.context_length         */
    int n_layers;      /* <arch>.block_count            */
    int n_embd;        /* <arch>.embedding_length       */
    int n_head;        /* <arch>.attention.head_count   */
    int n_head_kv;     /* <arch>.attention.head_count_kv scalar (0 if array/absent) */

    /* Explicit KV head dimensions. When present (e.g. Gemma, DeepSeek) these
     * override the head_dim = n_embd/n_head assumption, which is otherwise
     * wrong. 0 means "fall back to n_embd/n_head". */
    int key_length;          /* <arch>.attention.key_length        */
    int value_length;        /* <arch>.attention.value_length      */
    int key_length_swa;      /* …key_length_swa   (sliding-window layers) */
    int value_length_swa;    /* …value_length_swa */
    int sliding_window;      /* …sliding_window — 0 means no SWA   */

    /* Per-layer arrays (size n_layers) when the GGUF stores them as arrays —
     * Gemma varies KV heads and local/global attention per layer. NULL when
     * the model uses uniform scalars. */
    int  *head_kv_per_layer; /* …head_count_kv as an array */
    bool *is_swa_per_layer;  /* …sliding_window_pattern (true = windowed) */

    /* Exact per-layer weight bytes from walking the tensor table. NULL if
     * tensor walk failed; estimator falls back to file_size/n_layers. */
    double *layer_weight_mb;     /* size n_layers */
    double  fixed_weight_mb;     /* token_embd, output, output_norm, etc. */
} GGUFArch;

/* True when `key` ends with `suffix` — avoids the substring trap where
 * ".embedding_length" also matches ".embedding_length_per_layer_input". */
static bool key_suffix_is(const char *key, const char *suffix) {
    size_t kl = strlen(key), sl = strlen(suffix);
    return kl >= sl && strcmp(key + kl - sl, suffix) == 0;
}

/* Bytes per element for a given GGML tensor dtype. Returns 4.0 for unknown
 * types so we don't grossly underestimate. Block sizes match ggml-quants. */
static double ggml_dtype_bytes_per_elem(uint32_t dtype) {
    switch (dtype) {
    case 0:  return 4.0;             /* F32   */
    case 1:  return 2.0;             /* F16   */
    case 2:  return 18.0 / 32.0;     /* Q4_0  */
    case 3:  return 20.0 / 32.0;     /* Q4_1  */
    case 6:  return 22.0 / 32.0;     /* Q5_0  */
    case 7:  return 24.0 / 32.0;     /* Q5_1  */
    case 8:  return 34.0 / 32.0;     /* Q8_0  */
    case 9:  return 40.0 / 32.0;     /* Q8_1  */
    case 10: return 84.0 / 256.0;    /* Q2_K  */
    case 11: return 110.0 / 256.0;   /* Q3_K  */
    case 12: return 144.0 / 256.0;   /* Q4_K  */
    case 13: return 176.0 / 256.0;   /* Q5_K  */
    case 14: return 210.0 / 256.0;   /* Q6_K  */
    case 15: return 292.0 / 256.0;   /* Q8_K  */
    case 16: return 66.0  / 256.0;   /* IQ2_XXS */
    case 17: return 74.0  / 256.0;   /* IQ2_XS  */
    case 18: return 110.0 / 256.0;   /* IQ3_XXS */
    case 19: return 50.0  / 256.0;   /* IQ1_S   */
    case 20: return 18.0  / 32.0;    /* IQ4_NL  */
    case 21: return 110.0 / 256.0;   /* IQ3_S   */
    case 22: return 82.0  / 256.0;   /* IQ2_S   */
    case 23: return 136.0 / 256.0;   /* IQ4_XS  */
    case 24: return 1.0;             /* I8      */
    case 25: return 2.0;             /* I16     */
    case 26: return 4.0;             /* I32     */
    case 27: return 8.0;             /* I64     */
    case 28: return 8.0;             /* F64     */
    case 29: return 56.0  / 256.0;   /* IQ1_M   */
    case 30: return 2.0;             /* BF16    */
    /* Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8 share Q4_0 layout. */
    case 31: case 32: case 33: return 18.0 / 32.0;
    case 34: return 26.0 / 256.0;    /* TQ1_0 */
    case 35: return 64.0 / 256.0;    /* TQ2_0 */
    default: return 4.0;             /* unknown — overestimate slightly */
    }
}

/* Parse a tensor name like "blk.13.ffn_down_exps.weight" → 13.
 * Returns -1 for non-block tensors (token_embd, output, output_norm, …). */
static int tensor_layer_index(const char *name) {
    if (strncmp(name, "blk.", 4) != 0) return -1;
    int idx = 0;
    const char *p = name + 4;
    if (*p < '0' || *p > '9') return -1;
    while (*p >= '0' && *p <= '9') { idx = idx * 10 + (*p - '0'); p++; }
    return idx;
}

/* Read GGUF metadata fields needed for VRAM estimation. Missing fields stay 0. */
static GGUFArch read_gguf_arch(const char *path) {
    GGUFArch r = {0};
    FILE *f = fopen(path, "rb");
    if (!f) return r;

    /* GGUF header: magic(4) + version(4) + tensor_count(8) + metadata_count(8) */
    uint32_t magic, version;
    uint64_t tensor_count, metadata_count;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return r; }
    if (fread(&version, 4, 1, f) != 1) { fclose(f); return r; }
    if (fread(&tensor_count, 8, 1, f) != 1) { fclose(f); return r; }
    if (fread(&metadata_count, 8, 1, f) != 1) { fclose(f); return r; }

    /* Per-layer arrays captured during the metadata walk; mapped to n_layers
     * after the loop (metadata order isn't guaranteed). */
    int     *tmp_kv     = NULL; uint64_t tmp_kv_len  = 0;
    bool    *tmp_swa    = NULL; uint64_t tmp_swa_len = 0;
    const uint64_t ARR_CAP = 4096;  /* sane upper bound on layer count */

    bool metadata_clean = true;
    for (uint64_t i = 0; i < metadata_count; i++) {
        /* key: len(8) + data */
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) { metadata_clean = false; break; }
        char key[257] = {0};
        if (key_len > 256) {
            /* Skip the entire over-sized key; we don't care about its name. */
            if (fseek(f, (long)key_len, SEEK_CUR) != 0) { metadata_clean = false; break; }
        } else {
            if (fread(key, 1, key_len, f) != key_len) { metadata_clean = false; break; }
            key[key_len] = '\0';
        }

        /* value type */
        uint32_t vtype;
        if (fread(&vtype, 4, 1, f) != 1) { metadata_clean = false; break; }

        if (vtype == 4 || vtype == 5) { /* uint32 / int32 */
            uint32_t val;
            if (fread(&val, 4, 1, f) != 1) { metadata_clean = false; break; }
            /* Exact-suffix matching: ".embedding_length" must NOT match
             * ".embedding_length_per_layer_input" (=0), which would zero
             * n_embd and silently drop the entire KV-cache estimate. */
            if      (key_suffix_is(key, ".context_length"))             r.n_ctx_train = (int)val;
            else if (key_suffix_is(key, ".block_count"))                r.n_layers    = (int)val;
            else if (key_suffix_is(key, ".embedding_length"))           r.n_embd      = (int)val;
            else if (key_suffix_is(key, ".attention.head_count_kv"))    r.n_head_kv   = (int)val;
            else if (key_suffix_is(key, ".attention.head_count"))       r.n_head      = (int)val;
            else if (key_suffix_is(key, ".attention.key_length_swa"))   r.key_length_swa   = (int)val;
            else if (key_suffix_is(key, ".attention.value_length_swa")) r.value_length_swa = (int)val;
            else if (key_suffix_is(key, ".attention.key_length"))       r.key_length   = (int)val;
            else if (key_suffix_is(key, ".attention.value_length"))     r.value_length = (int)val;
            else if (key_suffix_is(key, ".attention.sliding_window"))   r.sliding_window = (int)val;
        } else if (vtype == 6) { fseek(f, 4, SEEK_CUR); }
        else if (vtype == 0 || vtype == 1 || vtype == 7) { fseek(f, 1, SEEK_CUR); }
        else if (vtype == 2 || vtype == 3) { fseek(f, 2, SEEK_CUR); }
        else if (vtype == 10 || vtype == 12) { fseek(f, 8, SEEK_CUR); }
        else if (vtype == 8) { /* string */
            uint64_t slen;
            if (fread(&slen, 8, 1, f) != 1) { metadata_clean = false; break; }
            fseek(f, (long)slen, SEEK_CUR);
        } else if (vtype == 9) { /* array */
            uint32_t atype;
            uint64_t alen;
            if (fread(&atype, 4, 1, f) != 1) { metadata_clean = false; break; }
            if (fread(&alen, 8, 1, f) != 1)  { metadata_clean = false; break; }

            /* Capture the two per-layer arrays Gemma stores as arrays: KV head
             * counts (int) and the local/global attention pattern (bool). */
            int  *cap_kv  = NULL;
            bool *cap_swa = NULL;
            if (alen > 0 && alen <= ARR_CAP) {
                if ((atype == 4 || atype == 5) &&
                    key_suffix_is(key, ".attention.head_count_kv")) {
                    cap_kv = calloc((size_t)alen, sizeof(int));
                } else if (atype == 7 &&
                           key_suffix_is(key, ".attention.sliding_window_pattern")) {
                    cap_swa = calloc((size_t)alen, sizeof(bool));
                }
            }

            bool array_clean = true;
            for (uint64_t a = 0; a < alen; a++) {
                if (atype == 8) {
                    uint64_t slen;
                    if (fread(&slen, 8, 1, f) != 1) { array_clean = false; break; }
                    fseek(f, (long)slen, SEEK_CUR);
                } else if (atype == 4 || atype == 5 || atype == 6) {
                    uint32_t v;
                    if (fread(&v, 4, 1, f) != 1) { array_clean = false; break; }
                    if (cap_kv) cap_kv[a] = (int)v;
                } else if (atype == 0 || atype == 1 || atype == 7) {
                    uint8_t v;
                    if (fread(&v, 1, 1, f) != 1) { array_clean = false; break; }
                    if (cap_swa) cap_swa[a] = (v != 0);
                } else if (atype == 2 || atype == 3) { fseek(f, 2, SEEK_CUR); }
                else if (atype == 10 || atype == 12) { fseek(f, 8, SEEK_CUR); }
                else { array_clean = false; break; }
            }
            if (!array_clean) {
                free(cap_kv); free(cap_swa);
                metadata_clean = false; break;
            }
            if (cap_kv)  { free(tmp_kv);  tmp_kv  = cap_kv;  tmp_kv_len  = alen; }
            if (cap_swa) { free(tmp_swa); tmp_swa = cap_swa; tmp_swa_len = alen; }
        } else {
            metadata_clean = false; /* unknown type — file position is now lost */
            break;
        }
    }

    /* Map captured per-layer arrays onto n_layers (clamp if lengths differ). */
    if (r.n_layers > 0) {
        if (tmp_kv && tmp_kv_len > 0) {
            r.head_kv_per_layer = calloc((size_t)r.n_layers, sizeof(int));
            if (r.head_kv_per_layer)
                for (uint64_t i = 0; i < (uint64_t)r.n_layers; i++)
                    r.head_kv_per_layer[i] = tmp_kv[i < tmp_kv_len ? i : tmp_kv_len - 1];
        }
        if (tmp_swa && tmp_swa_len > 0) {
            r.is_swa_per_layer = calloc((size_t)r.n_layers, sizeof(bool));
            if (r.is_swa_per_layer)
                for (uint64_t i = 0; i < (uint64_t)r.n_layers; i++)
                    r.is_swa_per_layer[i] = tmp_swa[i < tmp_swa_len ? i : tmp_swa_len - 1];
        }
    }
    free(tmp_kv);
    free(tmp_swa);

    /* Walk tensor info section: each entry is name(str) + n_dims(u32)
     * + dims[n_dims](u64) + dtype(u32) + offset(u64). Exact per-layer
     * weight bytes matter for MoE models where layers are wildly uneven. */
    if (metadata_clean && r.n_layers > 0 && tensor_count > 0 && tensor_count < 1000000) {
        r.layer_weight_mb = calloc((size_t)r.n_layers, sizeof(double));
        if (r.layer_weight_mb) {
            bool tensor_clean = true;
            for (uint64_t t = 0; t < tensor_count; t++) {
                uint64_t name_len;
                if (fread(&name_len, 8, 1, f) != 1) { tensor_clean = false; break; }
                if (name_len > 1024) { tensor_clean = false; break; }
                char name[1025] = {0};
                if (fread(name, 1, name_len, f) != name_len) { tensor_clean = false; break; }
                name[name_len] = '\0';

                uint32_t n_dims;
                if (fread(&n_dims, 4, 1, f) != 1) { tensor_clean = false; break; }
                if (n_dims == 0 || n_dims > 8) { tensor_clean = false; break; }

                uint64_t elements = 1;
                for (uint32_t d = 0; d < n_dims; d++) {
                    uint64_t dim;
                    if (fread(&dim, 8, 1, f) != 1) { tensor_clean = false; break; }
                    elements *= dim;
                }
                if (!tensor_clean) break;

                uint32_t dtype;
                if (fread(&dtype, 4, 1, f) != 1) { tensor_clean = false; break; }
                uint64_t off;
                if (fread(&off, 8, 1, f) != 1)   { tensor_clean = false; break; }

                double mb = (double)elements * ggml_dtype_bytes_per_elem(dtype)
                            / (1024.0 * 1024.0);
                int layer = tensor_layer_index(name);
                if (layer >= 0 && layer < r.n_layers) {
                    r.layer_weight_mb[layer] += mb;
                } else {
                    r.fixed_weight_mb += mb;
                }
            }
            if (!tensor_clean) {
                free(r.layer_weight_mb);
                r.layer_weight_mb = NULL;
                r.fixed_weight_mb = 0.0;
            }
        }
    }

    fclose(f);
    return r;
}

/* Extract a display name from a gguf filename */
static const char *model_display_name(const char *path) {
    const char *base = strrchr(path, '/');
    return base ? base + 1 : path;
}

/* Get file size in MB */
static double file_size_mb(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) return st.st_size / (1024.0 * 1024.0);
    return 0;
}

/* Launch config */

/* Recursively collect .gguf files under `root` (skips mmproj weights). */
static void scan_gguf_recursive(const char *root, char ***list, int *count, int *cap) {
    DIR *dir = opendir(root);
    if (!dir) return;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;  /* skip ., .., hidden */
        char fullpath[1024];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", root, ent->d_name);

        struct stat st;
        if (lstat(fullpath, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            scan_gguf_recursive(fullpath, list, count, cap);
            continue;
        }
        /* Resolve symlinks for regular-file checks (HF stores blobs via symlinks). */
        if (S_ISLNK(st.st_mode) && stat(fullpath, &st) != 0) continue;
        if (!S_ISREG(st.st_mode) && !S_ISLNK(st.st_mode)) continue;

        size_t nlen = strlen(ent->d_name);
        if (nlen < 5) continue;
        if (strcmp(ent->d_name + nlen - 5, ".gguf") != 0) continue;
        if (strstr(ent->d_name, "mmproj") != NULL) continue;

        if (*count >= *cap) {
            *cap = *cap ? *cap * 2 : 16;
            *list = realloc(*list, *cap * sizeof(char *));
        }
        (*list)[(*count)++] = strdup(fullpath);
    }
    closedir(dir);
}

/* Public: scan the model search dirs and return every .gguf path found
 * (malloc'd array of malloc'd strings). Caller frees each entry then the array.
 * Returns the count (0 with *out=NULL if none). Used by /model to resolve a
 * substring like "qwen3.6" to a concrete path without the full picker TUI. */
int basi_list_models(char ***out) {
    init_model_search_dirs();
    char **models = NULL;
    int count = 0, cap = 0;
    for (int d = 0; d < MODEL_DIRS_MAX && model_search_dirs[d]; d++)
        scan_gguf_recursive(model_search_dirs[d], &models, &count, &cap);
    *out = models;
    return count;
}

/* Settings values for ←/→ adjustment.
 * GPU_LAYER_AUTO (-1) means: auto-fit to available VRAM each render.
 * Manual values step by 1 in [0, model's n_layers]. */
#define GPU_LAYER_AUTO (-1)
static const float temp_opts[]     = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
#define N_TEMP_OPTS (int)(sizeof(temp_opts)/sizeof(temp_opts[0]))
#define CTX_STEP    1024  /* slider step size */
#define CTX_MIN     1024
#define CTX_DEFAULT 32768

/* Base Vulkan allocator + command pool + small fixed buffers.
 * Compute scratch on top of this is computed dynamically from ctx × n_embd. */
#define VRAM_OVERHEAD_BASE_MB 600.0
/* Fallback only: when the driver can't report live VRAM budget
 * (VK_EXT_memory_budget absent), assume this much is held by the
 * display/compositor and subtract it from the total heap size. */
#define VRAM_RESERVE_MB 2048.0
/* Safety margin held back from the driver's live free-VRAM figure, to absorb
 * allocator fragmentation and growth during a long generation. */
#define VRAM_SAFETY_MB 512.0

/* Memory split between VRAM and system RAM for a given offload setting. */
typedef struct { double vram_mb; double ram_mb; } MemorySplit;

/* Estimate weights + KV cache split given the model arch and offload count.
 * Uses exact per-layer tensor bytes when the GGUF tensor walk succeeded;
 * otherwise falls back to file_size / n_layers. */
static MemorySplit estimate_memory(double file_size_mb, GGUFArch arch,
                                   int gpu_layers, int ctx) {
    MemorySplit s = {0, 0};
    int total_layers = arch.n_layers > 0 ? arch.n_layers : 32;
    if (gpu_layers > total_layers) gpu_layers = total_layers;
    if (gpu_layers < 0) gpu_layers = 0;

    /* Weights — exact per-layer if the tensor walker succeeded, else even split. */
    if (arch.layer_weight_mb) {
        for (int i = 0; i < gpu_layers; i++)        s.vram_mb += arch.layer_weight_mb[i];
        for (int i = gpu_layers; i < total_layers; i++) s.ram_mb += arch.layer_weight_mb[i];
        /* Fixed tensors (token_embd, output, output_norm) live on CPU when not
         * fully offloaded. llama.cpp puts them on GPU only when n_gpu_layers
         * exceeds n_blocks, which is out of our slider's range. */
        s.ram_mb += arch.fixed_weight_mb;
    } else {
        double weights_per_layer = file_size_mb / (double)total_layers;
        s.vram_mb += gpu_layers * weights_per_layer;
        s.ram_mb  += (total_layers - gpu_layers) * weights_per_layer;
    }

    /* KV cache — fp16, summed exactly per layer. KV head count and the cached
     * token window can vary across layers: Gemma alternates local sliding-
     * window layers (which cache only `sliding_window` tokens, with smaller
     * head dims) against full-context global layers, and uses explicit
     * key/value head dimensions that differ from n_embd/n_head. */
    {
        int n_head    = arch.n_head    > 0 ? arch.n_head    : 1;
        int kdim_full = arch.key_length   > 0 ? arch.key_length
                        : (arch.n_embd > 0 ? arch.n_embd / n_head : 0);
        int vdim_full = arch.value_length > 0 ? arch.value_length
                        : (arch.n_embd > 0 ? arch.n_embd / n_head : 0);
        int kdim_swa  = arch.key_length_swa   > 0 ? arch.key_length_swa   : kdim_full;
        int vdim_swa  = arch.value_length_swa > 0 ? arch.value_length_swa : vdim_full;
        int scalar_kv = arch.n_head_kv > 0 ? arch.n_head_kv : n_head;

        if (kdim_full > 0) {
            for (int i = 0; i < total_layers; i++) {
                int kvh = arch.head_kv_per_layer ? arch.head_kv_per_layer[i] : scalar_kv;
                if (kvh <= 0) kvh = scalar_kv;
                bool swa = arch.is_swa_per_layer ? arch.is_swa_per_layer[i] : false;
                int tokens = ctx, kdim = kdim_full, vdim = vdim_full;
                if (swa && arch.sliding_window > 0) {
                    if (arch.sliding_window < tokens) tokens = arch.sliding_window;
                    kdim = kdim_swa;
                    vdim = vdim_swa;
                }
                /* K and V caches: tokens × kv_heads × head_dim × 2 bytes. */
                double layer_kv_mb = (double)tokens * (double)kvh
                                     * ((double)kdim + (double)vdim) * 2.0
                                     / (1024.0 * 1024.0);
                if (i < gpu_layers) s.vram_mb += layer_kv_mb;
                else                s.ram_mb  += layer_kv_mb;
            }
        }
    }

    /* Compute scratch: dominated by ctx × n_embd intermediate tensors during
     * prompt processing. Coefficient of 6 calibrated empirically against
     * Qwen3-Coder-30B-A3B at ctx=262144 on Intel Arc B580 (Vulkan); lower
     * values under-predict at very long contexts and trigger OOM. */
    if (gpu_layers > 0) {
        double n_embd_for_scratch = arch.n_embd > 0 ? (double)arch.n_embd : 2048.0;
        double compute_scratch_mb = (double)ctx * n_embd_for_scratch * 6.0
                                    / (1024.0 * 1024.0);
        s.vram_mb += VRAM_OVERHEAD_BASE_MB + compute_scratch_mb;
    }
    return s;
}

/* Largest gpu_layers (in [0, n_layers]) whose VRAM footprint fits `vram_budget_mb`.
 * Returns 0 if even the smallest non-zero offload spills. */
static int auto_fit_layers(double file_size_mb, GGUFArch arch,
                           int ctx, double vram_budget_mb) {
    int total = arch.n_layers > 0 ? arch.n_layers : 32;
    for (int g = total; g >= 0; g--) {
        MemorySplit s = estimate_memory(file_size_mb, arch, g, ctx);
        if (s.vram_mb <= vram_budget_mb) return g;
    }
    return 0;
}

/* VRAM freed by a just-exited model (e.g. after a /model re-exec) can lag in the
 * driver's live budget for a few hundred ms. Sample the probe until the free
 * figure stops climbing (or a short timeout), so the picker's VRAM math reflects
 * the fully-offloaded GPU rather than a mid-teardown snapshot. Cheap when nothing
 * is changing: the first two reads already agree and it returns after one tick. */
static HwInfo hw_probe_settled(void) {
    HwInfo prev = hw_probe();
    if (!prev.has_gpu || !prev.vram_budget_known) return prev;
    for (int i = 0; i < 16; i++) {                 /* up to ~2.4s */
        usleep(150000);
        HwInfo cur = hw_probe();
        if (!cur.vram_budget_known) return cur;
        long long climbed = (long long)cur.vram_avail_mb - (long long)prev.vram_avail_mb;
        prev = cur;
        if (climbed <= 32) break;                  /* free stopped rising → settled */
    }
    return prev;
}

/*
 * Scan directories for .gguf files, show interactive menu with settings.
 * Returns filled LaunchConfig, or model_path=NULL on cancel.
 */
LaunchConfig pick_model(void) {
    LaunchConfig cfg = { NULL, 99, CONTEXT_SIZE, 0.4f, 0, 0 };

    /* Build search dirs */
    init_model_search_dirs();

    /* Collect .gguf files (recursive: HF hub nests files under models--ORG--NAME/snapshots/HASH/) */
    char **models = NULL;
    int count = 0, cap = 0;

    for (int d = 0; d < MODEL_DIRS_MAX && model_search_dirs[d]; d++) {
        scan_gguf_recursive(model_search_dirs[d], &models, &count, &cap);
    }

    if (count == 0) {
        fprintf(stderr, "No .gguf models found in search directories.\n");
        free(models);
        return cfg;
    }

    /* Raw terminal */
    struct termios orig;
    bool raw = false;
    if (tcgetattr(STDIN_FILENO, &orig) == 0) {
        struct termios t = orig;
        t.c_lflag &= ~(ICANON | ECHO);
        t.c_cc[VMIN] = 1;
        t.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &t);
        raw = true;
    }

    /* Read GGUF arch + cache sizes per model */
    GGUFArch *model_arch    = calloc(count, sizeof(GGUFArch));
    double   *model_size_mb = calloc(count, sizeof(double));
    for (int i = 0; i < count; i++) {
        model_arch[i]    = read_gguf_arch(models[i]);
        model_size_mb[i] = file_size_mb(models[i]);
    }

    /* Hardware probe. Wait for VRAM to settle first (a model offloaded by the
       /model re-exec may still be clearing), then let 'r' re-probe on demand. */
    HwInfo hw = hw_probe_settled();

    /* Menu state */
    enum { SECTION_MODEL, SECTION_GPU, SECTION_CTX, SECTION_TEMP,
           SECTION_SPEC, SECTION_FA, SECTION_LAUNCH, SECTION_COUNT };
    int section = SECTION_MODEL;
    int model_sel = 0;
    int gpu_setting = GPU_LAYER_AUTO;  /* -1 = auto, else absolute layer count */
    int ctx_val = CTX_DEFAULT;     /* free slider value */
    int temp_idx = 4;              /* default: 0.4 */
    /* llama-server launch flags (baked into .basi/run-llama-server.sh). Until the
       user toggles them, they auto-follow the selected model's MTP-ness (an MTP
       head gives lossless spec-decode); spec is forced off for a non-MTP model
       where draft-mtp has nothing to draft. */
    int spec_on = 0, fa_on = 0;
    int spec_touched = 0, fa_touched = 0;

    while (1) {
        printf("\033[2J\033[H");
        printf("\033[1;36m╔══════════════════════════════════════════════════════════════╗\033[0m\n");
        printf("\033[1;36m║           BASI-CLI — Model Configuration                    ║\033[0m\n");
        printf("\033[1;36m╚══════════════════════════════════════════════════════════════╝\033[0m\n\n");

        /* Model selection */
        printf("%s MODEL %s\n",
               section == SECTION_MODEL ? "\033[1;33m▸" : "  \033[90m",
               "\033[0m");
        for (int i = 0; i < count; i++) {
            double mb = file_size_mb(models[i]);
            if (i == model_sel) {
                printf("    \033[1;36m● %s\033[90m (%.0f MB)\033[0m\n",
                       model_display_name(models[i]), mb);
            } else {
                printf("    \033[90m○ %s (%.0f MB)\033[0m\n",
                       model_display_name(models[i]), mb);
            }
        }

        printf("\n");

        /* Clamp ctx_val to selected model's training context */
        int max_ctx = model_arch[model_sel].n_ctx_train > 0
                      ? model_arch[model_sel].n_ctx_train : 131072;
        if (ctx_val > max_ctx) ctx_val = max_ctx;

        /* Clamp manual setting to current model's layer count */
        int model_max_layers = model_arch[model_sel].n_layers > 0
                               ? model_arch[model_sel].n_layers : 99;
        if (gpu_setting > model_max_layers) gpu_setting = model_max_layers;

        /* Usable VRAM budget for offload. Prefer the driver's live free figure
         * (VK_EXT_memory_budget) — it already excludes the compositor and any
         * other process holding VRAM, so no fixed display reserve is guessed.
         * Fall back to total-minus-reserve only when the driver can't report
         * it. A small safety margin absorbs allocator growth during inference. */
        double vram_usable_mb = 0.0;
        if (hw.has_gpu) {
            if (hw.vram_budget_known)
                vram_usable_mb = (double)hw.vram_avail_mb - VRAM_SAFETY_MB;
            else
                vram_usable_mb = (double)hw.vram_total_mb - VRAM_RESERVE_MB;
            if (vram_usable_mb < 0) vram_usable_mb = 0;
        }

        /* Resolve effective GPU layer count (auto-fit when sentinel is selected) */
        int gpu_effective;
        if (gpu_setting == GPU_LAYER_AUTO) {
            gpu_effective = auto_fit_layers(model_size_mb[model_sel],
                                            model_arch[model_sel],
                                            ctx_val, vram_usable_mb);
        } else {
            gpu_effective = gpu_setting;
        }

        /* GPU layers */
        printf("%s GPU LAYERS    ",
               section == SECTION_GPU ? "\033[1;33m▸" : "  \033[90m");
        if (gpu_setting == GPU_LAYER_AUTO) {
            printf("\033[1mAUTO\033[0m \033[90m(%d / %d)\033[0m",
                   gpu_effective, model_max_layers);
        } else {
            printf("\033[1m%d\033[0m \033[90m/ %d\033[0m",
                   gpu_setting, model_max_layers);
        }
        if (section == SECTION_GPU) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        /* Context size with slider bar */
        {
            printf("%s CONTEXT SIZE  \033[1m%-6d\033[0m",
                   section == SECTION_CTX ? "\033[1;33m▸" : "  \033[90m",
                   ctx_val);

            int bar_width = 20;
            int filled = max_ctx > 0 ? (int)((long)ctx_val * bar_width / max_ctx) : 0;
            if (filled > bar_width) filled = bar_width;
            printf(" \033[90m[");
            for (int b = 0; b < bar_width; b++) {
                if (b < filled) printf("\033[36m█");
                else printf("\033[90m░");
            }
            printf("\033[90m] max:%d\033[0m", max_ctx);
            if (section == SECTION_CTX) printf("  \033[90m← →\033[0m");
            printf("\n");
        }

        /* MEMORY row: shows VRAM/RAM split for the current settings */
        {
            MemorySplit ms = estimate_memory(model_size_mb[model_sel],
                                             model_arch[model_sel],
                                             gpu_effective, ctx_val);
            bool fits_gpu = hw.has_gpu && ms.vram_mb <= vram_usable_mb;
            bool spilling = ms.ram_mb > 0.5;  /* anything not on GPU */

            const char *gpu_color = !hw.has_gpu     ? "\033[90m"
                                  : !fits_gpu       ? "\033[1;31m"   /* red: doesn't fit */
                                  : spilling        ? "\033[1;33m"   /* yellow: partial */
                                                    : "\033[1;32m";  /* green: full GPU */
            const char *ram_color = spilling        ? "\033[1;33m" : "\033[90m";

            printf("    \033[90mMEMORY        \033[0m");
            if (hw.has_gpu) {
                printf("%s[%s %.1f / %.1f GB]\033[0m  ",
                       gpu_color, hw_vendor_label(hw.vendor_id),
                       ms.vram_mb / 1024.0, vram_usable_mb / 1024.0);
            } else {
                printf("\033[90m[no GPU detected]\033[0m  ");
            }
            printf("%s[RAM %.1f / %.1f GB]\033[0m\n",
                   ram_color, ms.ram_mb / 1024.0,
                   (double)hw.ram_total_mb / 1024.0);
            if (hw.has_gpu && hw.vram_budget_known) {
                printf("    \033[90m%s  %.1f GB free of %.1f GB (live)\033[0m\n",
                       hw.gpu_name, (double)hw.vram_avail_mb / 1024.0,
                       (double)hw.vram_total_mb / 1024.0);
            } else if (hw.has_gpu) {
                printf("    \033[90m%s  (%.1f GB reserved for display, estimated)\033[0m\n",
                       hw.gpu_name, VRAM_RESERVE_MB / 1024.0);
            }
        }

        /* Temperature */
        printf("%s TEMPERATURE   \033[1m%-6.1f\033[0m",
               section == SECTION_TEMP ? "\033[1;33m▸" : "  \033[90m",
               temp_opts[temp_idx]);
        if (section == SECTION_TEMP) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        /* llama-server launch flags — these become the .basi/run-llama-server.sh
           command. Spec-decode needs an MTP head, so it's n/a for non-MTP models. */
        int cur_mtp = (strstr(models[model_sel], "MTP") || strstr(models[model_sel], "mtp")) ? 1 : 0;
        if (!spec_touched) spec_on = cur_mtp;   /* auto-follow model until toggled */
        if (!fa_touched)   fa_on   = cur_mtp;
        if (!cur_mtp) spec_on = 0;              /* draft-mtp needs an MTP head */
        printf("%s SPEC-DECODE   \033[1m%s\033[0m",
               section == SECTION_SPEC ? "\033[1;33m▸" : "  \033[90m",
               !cur_mtp ? "n/a (no MTP head)" : (spec_on ? "draft-mtp (n-max 1)" : "off"));
        if (section == SECTION_SPEC && cur_mtp) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        printf("%s FLASH-ATTN    \033[1m%s\033[0m",
               section == SECTION_FA ? "\033[1;33m▸" : "  \033[90m",
               fa_on ? "on" : "off");
        if (section == SECTION_FA) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        printf("\n");

        /* Launch button */
        if (section == SECTION_LAUNCH) {
            printf("  \033[1;32m▸ [ LAUNCH ]\033[0m\n");
        } else {
            printf("    \033[90m[ LAUNCH ]\033[0m\n");
        }

        printf("\n\033[90m↑/↓ navigate  ←/→ adjust  r refresh VRAM  Enter select/launch  q quit\033[0m\n");
        fflush(stdout);

        /* Read key */
        unsigned char ch;
        if (read(STDIN_FILENO, &ch, 1) != 1) break;

        if (ch == 'q' || ch == 'Q' || ch == 3) break;

        if (ch == 'r' || ch == 'R') {          /* re-probe live VRAM on demand */
            hw = hw_probe_settled();
            continue;
        }

        if (ch == '\n' || ch == '\r') {
            if (section == SECTION_LAUNCH || section == SECTION_MODEL) {
                if (section == SECTION_MODEL) {
                    /* Enter on model goes to next section */
                    section = SECTION_GPU;
                    continue;
                }
                /* Launch */
                cfg.model_path = strdup(models[model_sel]);
                cfg.gpu_layers = gpu_effective;
                cfg.ctx_size = ctx_val;
                cfg.temperature = temp_opts[temp_idx];
                cfg.spec_draft_mtp = spec_on;
                cfg.flash_attn = fa_on;
                break;
            } else {
                /* Enter on setting goes to next section */
                section++;
            }
            continue;
        }

        if (ch == 27) {
            unsigned char seq[2];
            if (read(STDIN_FILENO, seq, 2) == 2 && seq[0] == '[') {
                switch (seq[1]) {
                case 'A': /* Up */
                    if (section == SECTION_MODEL && model_sel > 0) model_sel--;
                    else if (section > SECTION_MODEL) section--;
                    break;
                case 'B': /* Down */
                    if (section == SECTION_MODEL && model_sel < count - 1) model_sel++;
                    else if (section < SECTION_LAUNCH) section++;
                    break;
                case 'C': /* Right */
                    if (section == SECTION_GPU) {
                        int max_l = model_arch[model_sel].n_layers > 0
                                    ? model_arch[model_sel].n_layers : 99;
                        if (gpu_setting < max_l) gpu_setting++;
                    }
                    if (section == SECTION_CTX) {
                        int mc = model_arch[model_sel].n_ctx_train > 0
                                 ? model_arch[model_sel].n_ctx_train : 131072;
                        ctx_val += CTX_STEP;
                        if (ctx_val > mc) ctx_val = mc;
                    }
                    if (section == SECTION_TEMP && temp_idx < N_TEMP_OPTS - 1) temp_idx++;
                    if (section == SECTION_SPEC && cur_mtp) { spec_on = 1; spec_touched = 1; }
                    if (section == SECTION_FA) { fa_on = 1; fa_touched = 1; }
                    break;
                case 'D': /* Left */
                    if (section == SECTION_GPU && gpu_setting > GPU_LAYER_AUTO) gpu_setting--;
                    if (section == SECTION_CTX) {
                        ctx_val -= CTX_STEP;
                        if (ctx_val < CTX_MIN) ctx_val = CTX_MIN;
                    }
                    if (section == SECTION_TEMP && temp_idx > 0) temp_idx--;
                    if (section == SECTION_SPEC) { spec_on = 0; spec_touched = 1; }
                    if (section == SECTION_FA) { fa_on = 0; fa_touched = 1; }
                    break;
                }
            }
        }
    }

    if (raw) tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig);
    printf("\033[2J\033[H");
    fflush(stdout);

    for (int i = 0; i < count; i++) {
        free(models[i]);
        free(model_arch[i].layer_weight_mb);
        free(model_arch[i].head_kv_per_layer);
        free(model_arch[i].is_swa_per_layer);
    }
    free(models);
    free(model_arch);
    free(model_size_mb);
    return cfg;
}
