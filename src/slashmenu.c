/* slashmenu.c — slash-command autocomplete dropdown. See slashmenu.h. */
#include "slashmenu.h"

#include <string.h>
#include <strings.h>
#include <unistd.h>

/* ── command table ──────────────────────────────────────────────────── */

static const SlashCmd TABLE[] = {
    { "/help",        "this help",                              false },
    { "/model",       "switch model (picker or name)",          true  },
    { "/cookbook",    "download & manage models",               true  },
    { "/clear",       "drop conversation history",              false },
    { "/cost",        "show session token usage",               false },
    { "/save",        "export transcript as JSONL",             true  },
    { "/memory",      "edit ./BASI.md in $EDITOR",              false },
    { "/note",        "append a note to the knowledge base",    true  },
    { "/edit",        "edit a knowledge-base file",             true  },
    { "/permissions", "show or set permission mode",            true  },
    { "/plan",        "planning workflow (draft/accept/off)",   true  },
    { "/premortem",   "rewrite the plan with a pre-mortem",     false },
    { "/deepsearch",  "multi-round deep research",              true  },
    { "/study",       "run experiments until a question is settled", true  },
};
#define TABLE_N ((int)(sizeof(TABLE) / sizeof(TABLE[0])))

const SlashCmd *slashmenu_table(int *count) {
    if (count) *count = TABLE_N;
    return TABLE;
}

/* ── pure logic ─────────────────────────────────────────────────────── */

bool slashmenu_active(const char *line, size_t len, size_t cursor,
                      size_t *prefix_len) {
    if (len == 0 || line[0] != '/') return false;
    size_t upto = cursor < len ? cursor : len;
    for (size_t i = 0; i < upto; i++)
        if (line[i] == ' ') return false;      /* past the command word */
    if (prefix_len) *prefix_len = upto > 0 ? upto - 1 : 0;  /* chars after '/' */
    return true;
}

int slashmenu_filter(const char *prefix, size_t prefix_len, int *idx, int max) {
    int n = 0;
    for (int i = 0; i < TABLE_N; i++) {
        if (strncasecmp(prefix, TABLE[i].name + 1, prefix_len) == 0) {
            if (n < max) idx[n] = i;
            n++;
        }
    }
    return n;
}

int slashmenu_visible_width(const char *s) {
    int w = 0;
    for (const char *p = s; *p; ) {
        if (*p == '\033') {                    /* skip a CSI … final-byte run */
            p++;
            if (*p == '[') { p++; while (*p && !(*p >= 0x40 && *p <= 0x7e)) p++; if (*p) p++; }
            continue;
        }
        w++; p++;
    }
    return w;
}

/* ── rendering ──────────────────────────────────────────────────────── */

/* Paint one menu row's content (already positioned + cleared). Content is
 * truncated to the terminal width; the selected row gets a full-width bg. */
static void render_row(FILE *out, const SlashCmd *c, bool sel, int cols) {
    if (cols < 12) cols = 12;
    if (cols > 400) cols = 400;
    int budget = cols - 1;                     /* keep the last column free */

    /* layout: "  <name>  <desc>", name column padded to 13 */
    int namecol = 13;
    char left[64];
    int ln = snprintf(left, sizeof left, "  %-*s  ", namecol, c->name);
    if (ln > budget) ln = budget;

    int descbudget = budget - ln;
    if (descbudget < 0) descbudget = 0;

    if (sel) {
        /* full-width highlight bar */
        char row[600];
        snprintf(row, sizeof row, "%.*s%.*s", ln, left, descbudget, c->desc);
        int pad = budget - slashmenu_visible_width(row);
        fprintf(out, "\033[48;5;24m\033[97m%s%*s\033[0m", row, pad > 0 ? pad : 0, "");
    } else {
        fprintf(out, "  \033[36m%-*s\033[0m  \033[38;5;244m%.*s\033[0m",
                namecol, c->name, descbudget, c->desc);
    }
}

void slashmenu_close(FILE *out, SlashMenuState *st, int promptw, int cursorcol) {
    if (!st->open) return;
    for (int k = 0; k < st->rows; k++)
        fprintf(out, "\033[%d;1H\033[2K", st->r0 + 1 + k);
    fprintf(out, "\033[%d;%dH", st->r0, 1 + promptw + cursorcol);
    st->open = false;
    st->rows = 0;
    fflush(out);
}

void slashmenu_draw(FILE *out, SlashMenuState *st,
                    int r0, int region_bottom, int cols,
                    const int *idx, int matchc, int selected,
                    int promptw, int cursorcol) {
    /* erase a previously painted menu first (positions are still valid — no
     * output has scrolled the region since the last paint) */
    if (st->open)
        for (int k = 0; k < st->rows; k++)
            fprintf(out, "\033[%d;1H\033[2K", st->r0 + 1 + k);

    int need = matchc < SLASHMENU_MAX_VISIBLE ? matchc : SLASHMENU_MAX_VISIBLE;
    if (need <= 0) {                           /* nothing to show */
        fprintf(out, "\033[%d;%dH", r0, 1 + promptw + cursorcol);
        st->open = false; st->rows = 0;
        fflush(out);
        return;
    }

    /* scroll the region up if the menu would overrun the bottom edge */
    int avail = region_bottom - r0;
    if (need > avail) {
        int scroll = need - avail;
        fprintf(out, "\033[%d;1H", region_bottom);
        for (int i = 0; i < scroll; i++) fputc('\n', out);
        r0 -= scroll;
    }

    /* window the matches so the selection stays visible */
    int wstart = 0;
    if (matchc > need) {
        wstart = selected - need / 2;
        if (wstart < 0) wstart = 0;
        if (wstart > matchc - need) wstart = matchc - need;
    }

    for (int k = 0; k < need; k++) {
        int mi = idx[wstart + k];
        fprintf(out, "\033[%d;1H\033[2K", r0 + 1 + k);
        render_row(out, &TABLE[mi], (wstart + k) == selected, cols);
    }

    fprintf(out, "\033[%d;%dH", r0, 1 + promptw + cursorcol);
    st->open = true; st->r0 = r0; st->rows = need;
    fflush(out);
}

/* ── terminal helpers ───────────────────────────────────────────────── */

static unsigned char g_pb[64];
static int g_pb_len = 0;

static void pb_push(unsigned char c) {
    if (g_pb_len < (int)sizeof g_pb) g_pb[g_pb_len++] = c;
}

long slashmenu_read_byte(unsigned char *c) {
    if (g_pb_len > 0) {
        *c = g_pb[0];
        memmove(g_pb, g_pb + 1, (size_t)(--g_pb_len));
        return 1;
    }
    return (long)read(STDIN_FILENO, c, 1);
}

int slashmenu_cpr_row(void) {
    if (write(STDOUT_FILENO, "\033[6n", 4) != 4) return -1;
    /* reply: ESC [ <row> ; <col> R — tolerate leading stray bytes by pushing
     * them back for the editor to consume. */
    int state = 0, row = 0;
    for (int guard = 0; guard < 64; guard++) {
        unsigned char c;
        if (read(STDIN_FILENO, &c, 1) != 1) return -1;
        switch (state) {
        case 0: if (c == 27) state = 1; else pb_push(c); break;
        case 1: if (c == '[') state = 2; else { pb_push(c); state = 0; } break;
        case 2:
            if (c >= '0' && c <= '9') row = row * 10 + (c - '0');
            else if (c == ';') state = 3;
            else if (c == 'R') return row;
            else return -1;
            break;
        case 3: if (c == 'R') return row; break;   /* ignore the column */
        }
    }
    return -1;
}
