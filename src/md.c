/* Streaming markdown → ANSI renderer. See md.h.
 *
 * Model: a per-line classifier decides the line kind from its first few bytes
 * (heading / blockquote / list / rule / fenced code / paragraph), emitting any
 * block prefix, then the rest of the line streams live through an inline
 * scanner that toggles bold / italic / code and renders [text](url) links.
 * Paragraph text therefore streams token-by-token (only the tiny line-start
 * prefix is buffered); short block lines settle as their line completes.
 *
 * It only styles; it never drops or reorders visible characters — a construct
 * it doesn't recognise falls back to literal text. All state is module-global
 * and cleared by md_begin(). */

#include "md.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* ── styles (256-colour) ─────────────────────────────────────────────── */
#define RESET      "\033[0m"
#define A_BOLD     "\033[1m"
#define A_ITAL     "\033[3m"
#define S_CODE_IN  "\033[38;5;216m\033[48;5;236m"   /* inline code: amber on dark */
#define S_CODE_BL  "\033[38;5;252m"                  /* code-block text */
#define S_GUTTER   "\033[38;5;240m\xe2\x94\x82 \033[0m"  /* "│ " code gutter */
#define S_QUOTEBAR "\033[38;5;108m\xe2\x96\x8e\033[0m "  /* "▎ " */
#define S_QUOTETXT "\033[38;5;245m"
#define S_BULLET   "\033[38;5;111m\xe2\x80\xa2\033[0m " /* "• " */
#define S_NUM      "\033[38;5;111m"
#define S_RULE     "\033[38;5;240m"
#define S_LINK     "\033[4;38;5;111m"
#define S_URL      "\033[38;5;240m"

static const char *heading_style(int level) {
    switch (level) {
        case 1:  return "\033[1;4;38;5;231m";  /* bold underline bright */
        case 2:  return "\033[1;38;5;111m";    /* bold blue */
        case 3:  return "\033[1;38;5;108m";    /* bold teal */
        default: return "\033[1;38;5;145m";    /* bold grey */
    }
}

/* ── state ───────────────────────────────────────────────────────────── */
enum { L_START, L_BODY, L_HR, L_CODE, L_SKIP, L_TABLE };
static int   lstate;
static bool  in_fence;             /* inside a ``` code block */
static int   g_cols = 80;

/* Table buffering: a table needs all its rows before it can render (column
 * widths span every row), so it is the one buffered construct. Rows starting
 * with '|' accumulate here until a non-table line ends the block. */
#define TBL_ROWS 48
#define TBL_COLS 1024
#define TBL_MAXCOL 16
static char  tblbuf[TBL_ROWS][TBL_COLS];
static int   tbl_n;                /* index of the row currently filling */
static int   tbl_len;             /* fill length of the current row */
static bool  tbl_rowstart;        /* between rows: next byte decides continue/end */

static char  base[48];             /* SGR active for the current line body */
static bool  i_bold, i_ital, i_code;
static int   pending_star;         /* saw one '*', awaiting the next byte */
static bool  skip_one_space;       /* swallow one leading space (after '>') */

static bool  fed_any;              /* emit a clean RESET before the first output */
static char  sb[64];  static size_t sblen;       /* line-start classify buffer */
static char  hrbuf[128]; static size_t hrlen; static char hrchar;  /* L_HR */

/* link capture: 0 off, 1 in text, 2 saw ']', 3 in url */
static int   linkcap;
static char  ltext[512]; static size_t ltlen;
static char  lurl[1024];  static size_t lulen;

static void out(const char *s) { fputs(s, stdout); }
static void outn(const char *s, size_t n) { fwrite(s, 1, n, stdout); }
static void outc(char c) { putchar(c); }

/* Re-emit the full active SGR (base + inline attrs) after a RESET. */
static void apply_style(void) {
    out(RESET);
    if (base[0]) out(base);
    if (i_code)  out(S_CODE_IN);
    if (i_bold)  out(A_BOLD);
    if (i_ital)  out(A_ITAL);
}

/* ── inline scanner (paragraph / list / quote / heading body) ─────────── */
static void link_flush_literal(void) {
    /* Emit a partial/aborted link capture as plain text. */
    outc('[');
    if (ltlen) outn(ltext, ltlen);
    if (linkcap >= 2) outc(']');
    if (linkcap >= 3) { outc('('); if (lulen) outn(lurl, lulen); }
    linkcap = 0; ltlen = 0; lulen = 0;
}

static void body_char(char c);

static void link_char(char c) {
    if (linkcap == 1) {                       /* collecting link text */
        if (c == ']') { linkcap = 2; return; }
        if (c == '[' || ltlen + 1 >= sizeof ltext) { link_flush_literal(); body_char(c); return; }
        ltext[ltlen++] = c; return;
    }
    if (linkcap == 2) {                        /* after ']', want '(' */
        if (c == '(') { linkcap = 3; lulen = 0; return; }
        link_flush_literal(); body_char(c); return;
    }
    /* linkcap == 3: collecting url */
    if (c == ')') {
        out(S_LINK); outn(ltext, ltlen); apply_style();
        if (lulen) { out(" "); out(S_URL); outn(lurl, lulen); apply_style(); }
        linkcap = 0; ltlen = 0; lulen = 0; return;
    }
    if (lulen + 1 >= sizeof lurl) { link_flush_literal(); body_char(c); return; }
    lurl[lulen++] = c;
}

static void body_char(char c) {
    if (linkcap) { link_char(c); return; }
    if (pending_star) {
        pending_star = 0;
        if (c == '*') { i_bold = !i_bold; apply_style(); return; }
        /* a single '*' resolved: emphasis toggle, or literal before a space */
        if (i_ital)                       { i_ital = false; apply_style(); }
        else if (c != ' ' && c != '\t')   { i_ital = true;  apply_style(); }
        else                              { outc('*'); }
        /* fall through to handle c */
    }
    if (c == '*')  { pending_star = 1; return; }
    if (c == '`')  { i_code = !i_code; apply_style(); return; }
    if (c == '[')  { linkcap = 1; ltlen = 0; return; }
    outc(c);
}

/* ── line/block helpers ──────────────────────────────────────────────── */
static void emit_indent(size_t n) { for (size_t i = 0; i < n; i++) outc(' '); }

static void close_inline(void) {
    if (pending_star) { outc('*'); pending_star = 0; }
    if (linkcap) link_flush_literal();
    i_bold = i_ital = i_code = false;
    out(RESET);
    base[0] = 0;
}

static void end_line(void) {          /* end an L_BODY / L_CODE line */
    close_inline();
    outc('\n');
    lstate = L_START; sblen = 0; skip_one_space = false;
}

static void emit_rule(void) {
    int w = g_cols > 4 ? g_cols - 1 : 40;
    if (w > 120) w = 120;
    out(S_RULE);
    for (int i = 0; i < w; i++) out("\xe2\x94\x80");   /* ─ */
    out(RESET);
}

/* ── line-start classifier ───────────────────────────────────────────── */
static void to_para(size_t ind) {     /* replay buffered bytes as paragraph */
    emit_indent(ind);
    base[0] = 0;
    lstate = L_BODY;
    for (size_t i = ind; i < sblen; i++) body_char(sb[i]);
    sblen = 0;
}

static void start_hr(char ch) {       /* buffer the rest of the line to decide */
    lstate = L_HR; hrchar = ch; hrlen = 0;
    hrbuf[hrlen++] = ch; hrbuf[hrlen++] = ch;   /* the two markers already seen */
    sblen = 0;
}

static void handle_start(char c) {
    if (c == '\n') {                  /* blank / prefix-only line */
        if (sblen) outn(sb, sblen);
        outc('\n'); sblen = 0; return;
    }
    if (sblen < sizeof sb) sb[sblen++] = c;
    else { to_para(0); body_char(c); return; }   /* runaway: give up, stream */

    size_t ind = 0; while (ind < sblen && sb[ind] == ' ') ind++;
    if (ind == sblen) return;                     /* only spaces so far */
    const char *q = sb + ind; size_t m = sblen - ind;
    char c0 = q[0];

    /* fenced code ``` (open or close) */
    if (c0 == '`') {
        if (m < 3) { if (q[m - 1] != '`') { to_para(ind); return; } return; }
        if (strncmp(q, "```", 3) == 0) { in_fence = !in_fence; lstate = L_SKIP; sblen = 0; return; }
        to_para(ind); return;
    }
    if (in_fence) {                   /* inside a code block: this is a code line */
        out(S_GUTTER); emit_indent(ind); out(S_CODE_BL);
        lstate = L_CODE;
        for (size_t i = ind; i < sblen; i++) outc(sb[i]);
        sblen = 0; return;
    }
    /* table row: a line starting with '|' — buffer it; L_TABLE collects the rest */
    if (c0 == '|') {
        lstate = L_TABLE; tbl_n = 0; tbl_rowstart = false; tbl_len = 0;
        for (size_t k = ind; k < sblen && tbl_len < TBL_COLS - 1; k++) tblbuf[0][tbl_len++] = sb[k];
        tblbuf[0][tbl_len] = 0;
        sblen = 0; return;
    }
    /* heading */
    if (c0 == '#') {
        size_t h = 0; while (h < m && q[h] == '#') h++;
        if (h > 6) { to_para(ind); return; }
        if (h == m) return;           /* still counting '#'s */
        if (q[h] == ' ') {
            emit_indent(ind);
            snprintf(base, sizeof base, "%s", heading_style((int)h));
            out(base);
            lstate = L_BODY; sblen = 0; return;   /* '#'s + space consumed */
        }
        to_para(ind); return;         /* "#tag" */
    }
    /* blockquote */
    if (c0 == '>') {
        emit_indent(ind);
        out(S_QUOTEBAR);
        snprintf(base, sizeof base, "%s", S_QUOTETXT); apply_style();
        lstate = L_BODY; skip_one_space = true; sblen = 0; return;
    }
    /* unordered list / rule */
    if (c0 == '-' || c0 == '+' || c0 == '*') {
        if (m < 2) return;
        char c1 = q[1];
        if (c1 == ' ') {              /* "- " list item */
            emit_indent(ind); out(S_BULLET); base[0] = 0; lstate = L_BODY; sblen = 0; return;
        }
        if (c0 == '-' && c1 == '-') { start_hr('-'); return; }   /* maybe --- */
        to_para(ind); return;         /* **bold, *ital, -x … */
    }
    if (c0 == '_') {
        if (m < 2) return;
        if (q[1] == '_') { start_hr('_'); return; }              /* maybe ___ */
        to_para(ind); return;
    }
    /* ordered list "N. " */
    if (c0 >= '0' && c0 <= '9') {
        size_t d = 0; while (d < m && q[d] >= '0' && q[d] <= '9') d++;
        if (d == m) return;
        if (q[d] == '.' || q[d] == ')') {
            if (d + 1 == m) return;
            if (q[d + 1] == ' ') {
                emit_indent(ind);
                out(S_NUM); outn(q, d); outc(q[d]); out(RESET); outc(' ');
                base[0] = 0; lstate = L_BODY; sblen = 0; return;
            }
        }
        to_para(ind); return;
    }
    to_para(ind);                     /* plain paragraph */
}

/* ── tables ──────────────────────────────────────────────────────────── */
static size_t dwidth(const char *s, size_t n) {   /* display columns (codepoints) */
    size_t w = 0;
    for (size_t i = 0; i < n; i++) if (((unsigned char)s[i] & 0xC0) != 0x80) w++;
    return w;
}

/* A GitHub table separator row: only |-:  and whitespace, with at least one '-'. */
static bool is_separator(const char *row) {
    bool dash = false;
    for (const char *p = row; *p; p++) {
        char c = *p;
        if (c == '-') dash = true;
        else if (c != '|' && c != ':' && c != ' ' && c != '\t') return false;
    }
    return dash;
}

/* Strip the inline markers we can't style inside a cell, in place: **bold**,
 * `code`, ~~strike~~ → their text; [text](url) → text. Single * / _ are left
 * literal (they're often real characters in a data cell, e.g. n*log n). Keeps
 * the cell readable and keeps width/wrap correct. */
static void strip_md(char *s) {
    char *w = s, *r = s;
    while (*r) {
        if (r[0] == '*' && r[1] == '*') { r += 2; continue; }
        if (r[0] == '~' && r[1] == '~') { r += 2; continue; }
        if (r[0] == '`') { r += 1; continue; }
        if (r[0] == '[') {
            char *close = strchr(r, ']');
            if (close && close[1] == '(') {
                for (char *p = r + 1; p < close; p++) *w++ = *p;
                char *paren = strchr(close, ')');
                r = paren ? paren + 1 : close + 1;
                continue;
            }
        }
        *w++ = *r++;
    }
    *w = 0;
}

/* Split a "| a | b |" row into trimmed cells. Returns cell count. */
static int split_row(const char *row, char cells[][256], int maxc) {
    int nc = 0;
    const char *s = row;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '|') s++;                       /* drop optional leading pipe */
    const char *start = s;
    while (1) {
        if (*s == '|' || *s == '\0') {
            const char *a = start, *b = s;
            while (a < b && (*a == ' ' || *a == '\t')) a++;
            while (b > a && (b[-1] == ' ' || b[-1] == '\t')) b--;
            size_t len = (size_t)(b - a); if (len > 255) len = 255;
            if (nc < maxc) { memcpy(cells[nc], a, len); cells[nc][len] = 0; strip_md(cells[nc]); nc++; }
            if (*s == '\0') break;
            start = ++s;
            if (*s == '\0') break;            /* trailing pipe → no empty tail cell */
        } else s++;
    }
    return nc;
}

/* Greedy word-wrap `text` to `w` columns into out[] (≤ maxlines). */
static int wrap_cell(const char *text, int w, char out[][256], int maxlines) {
    int nl = 0; char cur[256]; int curlen = 0, curw = 0; cur[0] = 0;
    const char *p = text;
    if (w < 1) w = 1;
    while (*p && nl < maxlines) {
        while (*p == ' ') p++;
        if (!*p) break;
        const char *ws = p; while (*p && *p != ' ') p++;
        int wl = (int)(p - ws), ww = (int)dwidth(ws, (size_t)wl);
        if (ww > w) {                          /* hard-break an over-long word */
            if (curlen) { strcpy(out[nl++], cur); curlen = 0; curw = 0; cur[0] = 0; }
            const char *q = ws;
            while (q < p && nl < maxlines) {
                int cw = 0; const char *r = q;
                while (r < p && cw < w) { r++; if (((unsigned char)*r & 0xC0) != 0x80) cw++; }
                size_t seg = (size_t)(r - q); if (seg > 255) seg = 255;
                memcpy(cur, q, seg); cur[seg] = 0; strcpy(out[nl++], cur);
                cur[0] = 0; curlen = 0; curw = 0; q = r;
            }
            continue;
        }
        if (curw == 0) { memcpy(cur, ws, (size_t)wl); curlen = wl; cur[curlen] = 0; curw = ww; }
        else if (curw + 1 + ww <= w) { cur[curlen++] = ' '; memcpy(cur + curlen, ws, (size_t)wl); curlen += wl; cur[curlen] = 0; curw += 1 + ww; }
        else { strcpy(out[nl++], cur); memcpy(cur, ws, (size_t)wl); curlen = wl; cur[curlen] = 0; curw = ww; }
    }
    if (curlen && nl < maxlines) strcpy(out[nl++], cur);
    if (nl == 0) { out[0][0] = 0; nl = 1; }
    return nl;
}

static void emit_cell(const char *s, int w, int align, bool bold) {
    int tw = (int)dwidth(s, strlen(s)); if (tw > w) tw = w;
    int pad = w - tw, left, right;
    if (align == 1)      { left = pad; right = 0; }        /* right */
    else if (align == 2) { left = pad / 2; right = pad - pad / 2; }  /* center */
    else                 { left = 0; right = pad; }        /* left */
    for (int i = 0; i < left; i++) outc(' ');
    if (bold) out(A_BOLD);
    out(s);
    if (bold) out(RESET);
    for (int i = 0; i < right; i++) outc(' ');
}

static void tborder(const int *colw, int ncols, const char *l, const char *mid, const char *r) {
    out(S_RULE); out(l);
    for (int j = 0; j < ncols; j++) {
        for (int k = 0; k < colw[j] + 2; k++) out("\xe2\x94\x80");   /* ─ */
        out(j < ncols - 1 ? mid : r);
    }
    out(RESET); outc('\n');
}

static void trow(char cells[][256], int nc, const int *colw, const int *align,
                 int ncols, bool bold) {
    static char lines[TBL_MAXCOL][8][256]; int nlines[TBL_MAXCOL], maxl = 1;
    for (int j = 0; j < ncols; j++) {
        nlines[j] = wrap_cell(j < nc ? cells[j] : "", colw[j], lines[j], 8);
        if (nlines[j] > maxl) maxl = nlines[j];
    }
    for (int li = 0; li < maxl; li++) {
        out(S_RULE); out("\xe2\x94\x82"); out(RESET);          /* │ */
        for (int j = 0; j < ncols; j++) {
            outc(' ');
            emit_cell(li < nlines[j] ? lines[j][li] : "", colw[j], align[j], bold);
            outc(' ');
            out(S_RULE); out("\xe2\x94\x82"); out(RESET);
        }
        outc('\n');
    }
}

/* Re-emit the buffered rows as plain text (a "|"-line block that wasn't a table). */
static void flush_table_as_text(void) {
    for (int r = 0; r <= tbl_n; r++) { out(tblbuf[r]); outc('\n'); }
}

static void render_table(void) {
    if (tbl_n < 1 || !is_separator(tblbuf[1])) { flush_table_as_text(); return; }
    char header[TBL_MAXCOL][256]; int nh = split_row(tblbuf[0], header, TBL_MAXCOL);
    char sepc[TBL_MAXCOL][256];   int ns = split_row(tblbuf[1], sepc, TBL_MAXCOL);
    int ncols = nh; if (ncols < 1) { flush_table_as_text(); return; }
    if (ncols > TBL_MAXCOL) ncols = TBL_MAXCOL;

    int align[TBL_MAXCOL];
    for (int j = 0; j < ncols; j++) {
        const char *c = j < ns ? sepc[j] : "-"; size_t L = strlen(c);
        bool l = c[0] == ':', r = L && c[L - 1] == ':';
        align[j] = (l && r) ? 2 : (r ? 1 : 0);
    }

    static char data[TBL_ROWS][TBL_MAXCOL][256]; int dnc[TBL_ROWS], drows = 0;
    for (int r = 2; r <= tbl_n && drows < TBL_ROWS; r++)
        dnc[drows] = split_row(tblbuf[r], data[drows], TBL_MAXCOL), drows++;

    int colw[TBL_MAXCOL];
    for (int j = 0; j < ncols; j++) {
        size_t w = j < nh ? dwidth(header[j], strlen(header[j])) : 1;
        for (int r = 0; r < drows; r++)
            if (j < dnc[r]) { size_t cw = dwidth(data[r][j], strlen(data[r][j])); if (cw > w) w = cw; }
        if (w < 3) w = 3;
        if (w > 48) w = 48;
        colw[j] = (int)w;
    }
    /* shrink widest columns until the table fits the terminal */
    int budget = g_cols - 1;
    for (;;) {
        int tot = 1; for (int j = 0; j < ncols; j++) tot += colw[j] + 3;
        if (tot <= budget) break;
        int mj = 0; for (int j = 1; j < ncols; j++) if (colw[j] > colw[mj]) mj = j;
        if (colw[mj] <= 8) break;
        colw[mj]--;
    }

    tborder(colw, ncols, "\xe2\x94\x8c", "\xe2\x94\xac", "\xe2\x94\x90");  /* ┌ ┬ ┐ */
    trow(header, nh, colw, align, ncols, true);
    tborder(colw, ncols, "\xe2\x94\x9c", "\xe2\x94\xbc", "\xe2\x94\xa4");  /* ├ ┼ ┤ */
    for (int r = 0; r < drows; r++) {
        trow(data[r], dnc[r], colw, align, ncols, false);
        if (r < drows - 1)                                                /* rule between rows */
            tborder(colw, ncols, "\xe2\x94\x9c", "\xe2\x94\xbc", "\xe2\x94\xa4");
    }
    tborder(colw, ncols, "\xe2\x94\x94", "\xe2\x94\xb4", "\xe2\x94\x98");  /* └ ┴ ┘ */
}

static void table_finish(void) { render_table(); tbl_n = 0; tbl_len = 0; tbl_rowstart = false; }

/* ── public API ──────────────────────────────────────────────────────── */
void md_begin(void) {
    lstate = L_START; in_fence = false;
    base[0] = 0; i_bold = i_ital = i_code = false;
    pending_star = 0; skip_one_space = false;
    sblen = 0; hrlen = 0; linkcap = 0; ltlen = 0; lulen = 0; fed_any = false;
    tbl_n = 0; tbl_len = 0; tbl_rowstart = false;
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) g_cols = ws.ws_col;
    else g_cols = 80;
}

void md_feed(const char *s, size_t n) {
    if (n && !fed_any) { fed_any = true; out(RESET); }   /* clean slate */
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        switch (lstate) {
        case L_START:
            handle_start(c);
            break;
        case L_BODY:
            if (c == '\n') { end_line(); }
            else if (skip_one_space && c == ' ') { skip_one_space = false; }
            else { skip_one_space = false; body_char(c); }
            break;
        case L_HR:
            if (c == '\n') {
                /* rule iff every buffered byte is the marker or a space, ≥3 markers */
                size_t marks = 0; bool ok = true;
                for (size_t k = 0; k < hrlen; k++) {
                    if (hrbuf[k] == hrchar) marks++;
                    else if (hrbuf[k] != ' ') { ok = false; break; }
                }
                if (ok && marks >= 3) { emit_rule(); outc('\n'); }
                else { lstate = L_BODY; for (size_t k = 0; k < hrlen; k++) body_char(hrbuf[k]); end_line(); }
                lstate = L_START; sblen = 0; hrlen = 0;
            } else if (hrlen < sizeof hrbuf) { hrbuf[hrlen++] = c; }
            break;
        case L_CODE:
            if (c == '\n') { out(RESET); outc('\n'); lstate = L_START; sblen = 0; }
            else outc(c);
            break;
        case L_SKIP:                   /* remainder of a ``` fence line: swallow */
            if (c == '\n') { lstate = L_START; sblen = 0; }
            break;
        case L_TABLE:
            if (tbl_rowstart) {                       /* between rows: continue or end? */
                if (c == ' ' || c == '\t') break;
                if (c == '|') {                        /* another row */
                    tbl_rowstart = false;
                    if (tbl_n + 1 < TBL_ROWS) { tbl_n++; tbl_len = 0; tblbuf[tbl_n][tbl_len++] = c; tblbuf[tbl_n][tbl_len] = 0; }
                    else { table_finish(); lstate = L_START; }
                    break;
                }
                table_finish(); lstate = L_START;      /* table block ended */
                if (c == '\n') outc('\n');
                else handle_start(c);                  /* reprocess this line fresh */
                break;
            }
            if (c == '\n') {
                tblbuf[tbl_n][tbl_len] = 0;
                if (tbl_n == 1 && !is_separator(tblbuf[1])) {   /* row 2 not a separator → not a table */
                    flush_table_as_text(); tbl_n = 0; tbl_len = 0; lstate = L_START;
                } else {
                    tbl_rowstart = true;
                }
            } else if (tbl_len < TBL_COLS - 1) { tblbuf[tbl_n][tbl_len++] = c; tblbuf[tbl_n][tbl_len] = 0; }
            break;
        }
    }
    fflush(stdout);
}

void md_end(void) {
    switch (lstate) {
    case L_BODY:  close_inline(); break;
    case L_CODE:  out(RESET); break;
    case L_HR:    lstate = L_BODY; for (size_t k = 0; k < hrlen; k++) body_char(hrbuf[k]); close_inline(); break;
    case L_TABLE: tblbuf[tbl_n][tbl_len] = 0; table_finish(); break;
    case L_START: if (sblen) outn(sb, sblen); break;
    case L_SKIP:  break;
    }
    lstate = L_START; sblen = 0; hrlen = 0;
    fflush(stdout);
}
