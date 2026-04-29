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
} ThinkingState;

/* ── Generate response ─────────────────────────────────────────────── */


GenerateResult generate(
    struct llama_context *ctx,
    const struct llama_vocab *vocab,
    struct llama_sampler *smpl,
    const char *prompt,
    size_t prompt_len)
{
    GenerateResult res = { NULL, 0, 0, 0.0, 0.0 };
    StringBuf response;
    sb_init(&response);

    /* Check if first generation */
    llama_memory_t memory = llama_get_memory(ctx);
    bool is_first = (llama_memory_seq_pos_max(memory, 0) == -1);

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
    char tag_buf[16];
    size_t tag_len = 0;
    size_t spinner_frame = 0;
    double last_spinner = 0;
    bool thinking_box_shown = false;

    /* UTF-8 buffer */
    unsigned char utf8_buf[4];
    size_t utf8_len = 0;

    /* Generation loop */
    while (1) {
        /* Check context space */
        uint32_t n_ctx = llama_n_ctx(ctx);
        uint32_t n_ctx_used = (uint32_t)(llama_memory_seq_pos_max(memory, 0) + 1);
        if (n_ctx_used + (uint32_t)batch.n_tokens > n_ctx) {
            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
            printf("\n[Context limit reached]\n");
            fflush(stdout);
            break;
        }

        if (llama_decode(ctx, batch) != 0) {
            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
            printf("\n[Decode error]\n");
            fflush(stdout);
            break;
        }

        if (is_prompt_phase) {
            res.prompt_time_s = time_now() - timer_start;
            timer_start = time_now();
            is_prompt_phase = false;
        }

        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
            break;
        }

        if (generation_interrupted) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
            printf("\n\033[90m[interrupted]\033[0m");
            fflush(stdout);
            break;
        }

        /* Check for Ctrl+T (toggle thinking display) via non-blocking read */
        {
            struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
            if (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) {
                unsigned char key;
                if (read(STDIN_FILENO, &key, 1) == 1 && key == 0x14) { /* Ctrl+T */
                    show_thinking = !show_thinking;
                    if (state == STATE_THINKING || state == STATE_MAYBE_CLOSE) {
                        if (show_thinking) {
                            /* Switching from box to text */
                            if (thinking_box_shown) { if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); } thinking_box_shown = false; }
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
                        printf("\033[33m");
                        fwrite(buf + piece_start, 1, idx - piece_start, stdout);
                        fflush(stdout);
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
                const char *target = "<think>";
                if (tag_len <= 7 && tag_buf[tag_len - 1] == target[tag_len - 1]) {
                    if (tag_len == 7) {
                        state = STATE_THINKING;
                        tag_len = 0;
                        piece_start = idx + 1;
                        if (show_thinking) {
                            printf("\033[90m[thinking] ");
                            fflush(stdout);
                        } else {
                            draw_thinking_box(spinner_frame);
                            last_spinner = time_now();
                        }
                        thinking_box_shown = true;
                    }
                } else {
                    /* Not <think>, flush tag buffer */
                    printf("\033[33m");
                    fwrite(tag_buf, 1, tag_len, stdout);
                    fflush(stdout);
                    sb_append(&response, tag_buf, tag_len);
                    state = STATE_NORMAL;
                    tag_len = 0;
                    piece_start = idx + 1;
                }
                break;
            }

            case STATE_THINKING: {
                if (ch == '<') {
                    state = STATE_MAYBE_CLOSE;
                    tag_len = 0;
                    tag_buf[tag_len++] = ch;
                } else if (show_thinking) {
                    printf("\033[90m%c", ch);
                    fflush(stdout);
                } else {
                    /* Spinner animation */
                    double now = time_now();
                    if (now - last_spinner > 0.08) {
                        spinner_frame++;
                        draw_thinking_box(spinner_frame);
                        last_spinner = now;
                    }
                }
                piece_start = idx + 1;
                break;
            }

            case STATE_MAYBE_CLOSE: {
                tag_buf[tag_len++] = ch;
                const char *target = "</think>";
                if (tag_len <= 8 && tag_buf[tag_len - 1] == target[tag_len - 1]) {
                    if (tag_len == 8) {
                        state = STATE_NORMAL;
                        tag_len = 0;
                        piece_start = idx + 1;
                        if (show_thinking) {
                            printf("\033[0m\n");
                        } else {
                            clear_thinking_box();
                        }
                        fflush(stdout);
                        thinking_box_shown = false;
                    }
                } else {
                    if (show_thinking) {
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
                printf("\033[33m");
                fwrite(combined, 1, output_end, stdout);
                fflush(stdout);
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
           Only normal-state text is appended to `response`, so this tail match
           won't trigger inside <think> blocks. */
        if (response.len >= 7 &&
            memcmp(response.data + response.len - 7, "</tool>", 7) == 0) {
            res.gen_time_s = time_now() - timer_start;
            if (thinking_box_shown) {
                if (show_thinking) { printf("\033[0m\n"); } else { clear_thinking_box(); }
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
        printf("\033[33m");
        fwrite(utf8_buf, 1, utf8_len, stdout);
        sb_append(&response, (const char *)utf8_buf, utf8_len);
    }

    printf("\033[0m\n");
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
    const char *tmpl,
    const struct llama_chat_message *msgs, size_t n_msgs,
    bool add_gen_prompt,
    char *buf, size_t buf_size)
{
    if (tmpl) {
        int r = llama_chat_apply_template(tmpl, msgs, n_msgs, add_gen_prompt, buf, buf_size);
        if (r >= 0) return r;
    }
    return apply_chatml(msgs, n_msgs, add_gen_prompt, buf, buf_size);
}

/* ── Model picker ──────────────────────────────────────────────────── */

#define MODEL_DIRS_MAX 4
static const char *model_search_dirs[] = {
    NULL, /* filled at runtime: ~/.cache/llama.cpp */
    "/home/alberto/models",
    ".",
    NULL
};

/* Read max context length from GGUF metadata (returns 0 if not found) */
static int read_gguf_context_length(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    /* GGUF header: magic(4) + version(4) + tensor_count(8) + metadata_count(8) */
    uint32_t magic, version;
    uint64_t tensor_count, metadata_count;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return 0; }
    if (fread(&version, 4, 1, f) != 1) { fclose(f); return 0; }
    if (fread(&tensor_count, 8, 1, f) != 1) { fclose(f); return 0; }
    if (fread(&metadata_count, 8, 1, f) != 1) { fclose(f); return 0; }

    int result = 0;
    for (uint64_t i = 0; i < metadata_count; i++) {
        /* key: len(8) + data */
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) break;
        if (key_len > 256) { /* skip huge keys */ fseek(f, key_len, SEEK_CUR); }
        char key[257] = {0};
        size_t rlen = key_len < 256 ? key_len : 256;
        if (fread(key, 1, rlen, f) != rlen) break;
        if (key_len > 256) key[256] = '\0';

        /* value type */
        uint32_t vtype;
        if (fread(&vtype, 4, 1, f) != 1) break;

        if (vtype == 4) { /* uint32 */
            uint32_t val;
            if (fread(&val, 4, 1, f) != 1) break;
            if (strstr(key, "context_length")) {
                result = (int)val;
                break;
            }
        } else if (vtype == 5 || vtype == 6) { fseek(f, 4, SEEK_CUR); }
        else if (vtype == 0 || vtype == 1 || vtype == 7) { fseek(f, 1, SEEK_CUR); }
        else if (vtype == 2 || vtype == 3) { fseek(f, 2, SEEK_CUR); }
        else if (vtype == 10 || vtype == 12) { fseek(f, 8, SEEK_CUR); }
        else if (vtype == 8) { /* string */
            uint64_t slen;
            if (fread(&slen, 8, 1, f) != 1) break;
            fseek(f, slen, SEEK_CUR);
        } else if (vtype == 9) { /* array */
            uint32_t atype;
            uint64_t alen;
            if (fread(&atype, 4, 1, f) != 1) break;
            if (fread(&alen, 8, 1, f) != 1) break;
            for (uint64_t a = 0; a < alen; a++) {
                if (atype == 8) {
                    uint64_t slen;
                    if (fread(&slen, 8, 1, f) != 1) break;
                    fseek(f, slen, SEEK_CUR);
                } else if (atype == 4 || atype == 5 || atype == 6) { fseek(f, 4, SEEK_CUR); }
                else if (atype == 0 || atype == 1 || atype == 7) { fseek(f, 1, SEEK_CUR); }
                else if (atype == 2 || atype == 3) { fseek(f, 2, SEEK_CUR); }
                else if (atype == 10 || atype == 12) { fseek(f, 8, SEEK_CUR); }
            }
        } else {
            break; /* unknown type, bail */
        }
    }
    fclose(f);
    return result;
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

/* Settings values for ←/→ adjustment */
static const int gpu_layer_opts[]  = { 0, 10, 20, 30, 40, 50, 60, 80, 99 };
static const float temp_opts[]     = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
#define N_GPU_OPTS  (int)(sizeof(gpu_layer_opts)/sizeof(gpu_layer_opts[0]))
#define N_TEMP_OPTS (int)(sizeof(temp_opts)/sizeof(temp_opts[0]))
#define CTX_STEP    1024  /* slider step size */
#define CTX_MIN     1024
#define CTX_DEFAULT 32768

/*
 * Scan directories for .gguf files, show interactive menu with settings.
 * Returns filled LaunchConfig, or model_path=NULL on cancel.
 */
LaunchConfig pick_model(void) {
    LaunchConfig cfg = { NULL, 99, CONTEXT_SIZE, 0.4f };

    /* Build search dirs */
    static char cache_dir[512];
    const char *home = getenv("HOME");
    if (home) {
        snprintf(cache_dir, sizeof(cache_dir), "%s/.cache/huggingface/hub", home);
        model_search_dirs[0] = cache_dir;
    }

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

    /* Read max context for first model */
    int *model_max_ctx = calloc(count, sizeof(int));
    for (int i = 0; i < count; i++)
        model_max_ctx[i] = read_gguf_context_length(models[i]);

    /* Menu state */
    enum { SECTION_MODEL, SECTION_GPU, SECTION_CTX, SECTION_TEMP, SECTION_LAUNCH, SECTION_COUNT };
    int section = SECTION_MODEL;
    int model_sel = 0;
    int gpu_idx = N_GPU_OPTS - 1;  /* default: 99 */
    int ctx_val = CTX_DEFAULT;     /* free slider value */
    int temp_idx = 4;              /* default: 0.4 */

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

        /* GPU layers */
        printf("%s GPU LAYERS    \033[1m%-6d\033[0m",
               section == SECTION_GPU ? "\033[1;33m▸" : "  \033[90m",
               gpu_layer_opts[gpu_idx]);
        if (section == SECTION_GPU) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        /* Context size with slider bar */
        {
            int max_ctx = model_max_ctx[model_sel] > 0 ? model_max_ctx[model_sel] : 131072;
            /* Clamp ctx_val to max */
            if (ctx_val > max_ctx) ctx_val = max_ctx;

            printf("%s CONTEXT SIZE  \033[1m%-6d\033[0m",
                   section == SECTION_CTX ? "\033[1;33m▸" : "  \033[90m",
                   ctx_val);

            /* Draw slider bar */
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

        /* Temperature */
        printf("%s TEMPERATURE   \033[1m%-6.1f\033[0m",
               section == SECTION_TEMP ? "\033[1;33m▸" : "  \033[90m",
               temp_opts[temp_idx]);
        if (section == SECTION_TEMP) printf("  \033[90m← →\033[0m");
        printf("\033[0m\n");

        printf("\n");

        /* Launch button */
        if (section == SECTION_LAUNCH) {
            printf("  \033[1;32m▸ [ LAUNCH ]\033[0m\n");
        } else {
            printf("    \033[90m[ LAUNCH ]\033[0m\n");
        }

        printf("\n\033[90m↑/↓ navigate  ←/→ adjust  Enter select/launch  q quit\033[0m\n");
        fflush(stdout);

        /* Read key */
        unsigned char ch;
        if (read(STDIN_FILENO, &ch, 1) != 1) break;

        if (ch == 'q' || ch == 'Q' || ch == 3) break;

        if (ch == '\n' || ch == '\r') {
            if (section == SECTION_LAUNCH || section == SECTION_MODEL) {
                if (section == SECTION_MODEL) {
                    /* Enter on model goes to next section */
                    section = SECTION_GPU;
                    continue;
                }
                /* Launch */
                cfg.model_path = strdup(models[model_sel]);
                cfg.gpu_layers = gpu_layer_opts[gpu_idx];
                cfg.ctx_size = ctx_val;
                cfg.temperature = temp_opts[temp_idx];
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
                    if (section == SECTION_GPU && gpu_idx < N_GPU_OPTS - 1) gpu_idx++;
                    if (section == SECTION_CTX) {
                        int max_ctx = model_max_ctx[model_sel] > 0 ? model_max_ctx[model_sel] : 131072;
                        ctx_val += CTX_STEP;
                        if (ctx_val > max_ctx) ctx_val = max_ctx;
                    }
                    if (section == SECTION_TEMP && temp_idx < N_TEMP_OPTS - 1) temp_idx++;
                    break;
                case 'D': /* Left */
                    if (section == SECTION_GPU && gpu_idx > 0) gpu_idx--;
                    if (section == SECTION_CTX) {
                        ctx_val -= CTX_STEP;
                        if (ctx_val < CTX_MIN) ctx_val = CTX_MIN;
                    }
                    if (section == SECTION_TEMP && temp_idx > 0) temp_idx--;
                    break;
                }
            }
        }
    }

    if (raw) tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig);
    printf("\033[2J\033[H");
    fflush(stdout);

    for (int i = 0; i < count; i++) free(models[i]);
    free(models);
    free(model_max_ctx);
    return cfg;
}
