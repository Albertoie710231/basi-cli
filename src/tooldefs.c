#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "tooldefs.h"

/* ── Native tool table ──────────────────────────────────────────────────
 * One entry per existing BASI tool (kept 1:1 with the legacy <tool> set).
 * Schemas are deliberately FLAT — a single object, primitive properties, an
 * explicit `required` array, no $ref/anyOf/unbounded ints — because older /
 * smaller models silently fail to emit a call when a schema carries
 * constructs they can't parse (opencode-doc "schema sanitization"). */

#define OBJ(props, req) "{\"type\":\"object\",\"properties\":{" props "},\"required\":" req "}"
#define STR(name, desc) "\"" name "\":{\"type\":\"string\",\"description\":\"" desc "\"}"
#define INT(name, desc) "\"" name "\":{\"type\":\"integer\",\"description\":\"" desc "\"}"

static const BasiToolDef TOOLS[] = {
    { "read", "Read an entire small file (<~2000 tokens). For larger files use head/grep.",
      OBJ(STR("file", "Path to the file"), "[\"file\"]") },
    { "head", "Read the first N lines of a file.",
      OBJ(STR("file", "Path to the file") "," INT("lines", "Number of lines (default 10)"), "[\"file\"]") },
    { "tail", "Read the last N lines of a file.",
      OBJ(STR("file", "Path to the file") "," INT("lines", "Number of lines (default 10)"), "[\"file\"]") },
    { "grep", "Search a file for a pattern; returns matching lines with line numbers.",
      OBJ(STR("pattern", "Text or regex to search for") "," STR("file", "Path to the file") ","
          INT("context", "Lines of surrounding context to include (optional)"), "[\"pattern\",\"file\"]") },
    { "wc", "Count lines, words and characters in a file.",
      OBJ(STR("file", "Path to the file"), "[\"file\"]") },
    { "bash", "Run a shell command via 'bash -c'. Requires user approval. Use for builds, tests, git, or anything the other tools do not cover. To list or COUNT the functions in a C/C++ file, use 'ctags -x --c-kinds=f <file>' (append '| wc -l' to count) — it is exact. Do NOT try to match function definitions with grep/awk regexes; multi-line signatures, macros and return types on their own line make that unreliable.",
      OBJ(STR("command", "The shell command to run"), "[\"command\"]") },
    { "edit", "Create or modify a file. Read it first. To change part of a file, give the exact current text in 'search' and the new text in 'replace'. To create a NEW file, or to overwrite an existing file wholesale (e.g. fill in a stub), leave 'search' empty and put the full file content in 'replace'.",
      OBJ(STR("file", "Relative path to the file") "," STR("search", "Exact current text to replace (verbatim); empty to create a new file") ","
          STR("replace", "The replacement text"), "[\"file\",\"replace\"]") },
    { "scaffold", "Materialize a code template into a destination directory. Requires user approval. Use 'scaffold list' (name=list) to see templates.",
      OBJ(STR("name", "Template name, or 'list' to list templates") "," STR("dest", "Destination directory (default .)"), "[\"name\"]") },
    { "symbols", "List what a source file DEFINES, with exact counts by kind (functions, structs, macros...). Use this for any 'what is in this file' / 'how many functions' / 'where is X defined' question, and to find a symbol name before calling code_context. Exact — never count or locate definitions with grep/awk regexes, which multi-line signatures and macros defeat. Works for any language ctags parses (C/C++/Python/Go/JS/Rust/...).",
      OBJ(STR("file", "Path to the source file") "," STR("kind", "Optional filter: function, struct, macro, variable, ..."), "[\"file\"]") },
    { "code_context", "Return clangd's structural info (signature, type, doc) for a top-level C symbol. Authoritative for C — prefer it over grep when you need a symbol's signature, type or definition site. Requires the symbol NAME; use 'symbols' first to discover names.",
      OBJ(STR("file", "Path to the C file") "," STR("symbol", "Top-level identifier name"), "[\"file\",\"symbol\"]") },
    { "web_search", "Search the web. Use for any current/latest/version/price/news question. Returns ranked results plus the full text of the top pages.",
      OBJ(STR("query", "The search query") "," STR("recency", "Optional recency filter: day|week|month|year"), "[\"query\"]") },
    { "web_fetch", "Fetch and extract the readable text of one web page.",
      OBJ(STR("url", "The URL to fetch (http/https)"), "[\"url\"]") },
    { "readfile", "Read a local document (pdf/docx/odt/epub/text). Only when the user gave a concrete path.",
      OBJ(STR("path", "Path to the local document") "," STR("regex", "Optional regex to narrow output"), "[\"path\"]") },
    { "docs_toc", "List every document in the project knowledge base (no bodies).", "{}" },
    { "docs_get", "Read one knowledge-base document or section by its path[#anchor].",
      OBJ(STR("path", "Document path as shown by docs_toc, optionally with #anchor"), "[\"path\"]") },
    { "docs_search", "Literal substring grep across the whole knowledge base.",
      OBJ(STR("keyword", "Substring to search for"), "[\"keyword\"]") },
    { "docs_recent_notes", "Show all user notes for this project (newest first).", "{}" },
    { "docs_vector_search", "Semantic similarity search across the corpus (last resort when docs_search finds nothing).",
      OBJ(STR("query", "Natural-language query"), "[\"query\"]") },
    { "plan_write", "Save a Proposal-A3 plan artifact (drafting/premortem phase). Body is YAML frontmatter plus the seven A3 sections.",
      OBJ(STR("body", "The full plan document"), "[\"body\"]") },
    { "assumptions", "List unverified things the plan depends on (drafting phase), one per line.",
      OBJ(STR("items", "Newline-separated '- <item>' list"), "[\"items\"]") },
    { "spike_write", "Persist a spike artifact (spike phase): Question / Findings / Decision.",
      OBJ(STR("body", "The full spike document"), "[\"body\"]") },
    { "study_write", "Save a study: a hypothesis paired with a COMMAND that measures it. Use when a question can be settled by running something rather than by reasoning about it. Body is YAML frontmatter (metric, extract regex, runs, decision_rule) plus ## Question/Hypothesis/Experiment/Results/Verdict, with one ```arm <name> fenced block per arm. Then run `basi study run <slug>`: the verdict is computed from the measured numbers, not from your reading of them.",
      OBJ(STR("body", "The full study document"), "[\"body\"]") },
    { "study_run", "Execute a study written with study_write and return the measured results plus the computed verdict (SUPPORTED / REFUTED / INCONCLUSIVE). Run this after study_write, and after any change you want to measure. The verdict is computed from the numbers by the pre-registered decision rule — you cannot argue with it, so use it: keep the change if SUPPORTED, revert it if REFUTED, and if INCONCLUSIVE the effect was not distinguishable from noise, so do NOT report it as an improvement.",
      OBJ(STR("slug", "Slug of the study to run"), "[\"slug\"]") },
    { "plan_verify", "Run the verify clause of every Implementation Plan row (active phase), or one row by id.",
      OBJ(STR("id", "Optional row id, e.g. 1.2"), "[]") },
};

const BasiToolDef *basi_tool_defs(int *n) {
    if (n) *n = (int)(sizeof(TOOLS) / sizeof(TOOLS[0]));
    return TOOLS;
}

/* ── JSON args → execute_tool command string ───────────────────────────── */

static void append_arg(StringBuf *sb, const char *json, const char *key, bool quote) {
    char *v = jx_get_string(json, key);
    if (!v) return;
    sb_append_char(sb, ' ');
    if (quote) sb_append_char(sb, '"');
    sb_append_str(sb, v);
    if (quote) sb_append_char(sb, '"');
    free(v);
}

char *basi_build_command(const char *name, const char *args) {
    if (!name) return NULL;
    if (!args) args = "{}";

    StringBuf sb;
    sb_init(&sb);

    if (strcmp(name, "read") == 0) {
        sb_append_str(&sb, "read");
        append_arg(&sb, args, "file", false);
    } else if (strcmp(name, "head") == 0 || strcmp(name, "tail") == 0) {
        sb_append_str(&sb, name);
        long lines = jx_get_int(args, "lines");
        if (lines > 0) { char b[32]; snprintf(b, sizeof(b), " -n %ld", lines); sb_append_str(&sb, b); }
        append_arg(&sb, args, "file", false);
    } else if (strcmp(name, "grep") == 0) {
        sb_append_str(&sb, "grep -n");
        long ctx = jx_get_int(args, "context");
        if (ctx > 0) { char b[32]; snprintf(b, sizeof(b), " -C %ld", ctx); sb_append_str(&sb, b); }
        append_arg(&sb, args, "pattern", true);
        append_arg(&sb, args, "file", false);
    } else if (strcmp(name, "wc") == 0) {
        sb_append_str(&sb, "wc");
        append_arg(&sb, args, "file", false);
    } else if (strcmp(name, "bash") == 0) {
        sb_append_str(&sb, "bash");
        append_arg(&sb, args, "command", false);   /* raw rest-of-line, no quoting */
    } else if (strcmp(name, "edit") == 0) {
        char *file    = jx_get_string(args, "file");
        char *search  = jx_get_string(args, "search");
        char *replace = jx_get_string(args, "replace");
        sb_append_str(&sb, "edit ");
        sb_append_str(&sb, file ? file : "");
        sb_append_str(&sb, "\n<<<<<<< SEARCH\n");
        sb_append_str(&sb, search ? search : "");
        sb_append_str(&sb, "\n=======\n");
        sb_append_str(&sb, replace ? replace : "");
        sb_append_str(&sb, "\n>>>>>>> REPLACE\n");
        free(file); free(search); free(replace);
    } else if (strcmp(name, "scaffold") == 0) {
        sb_append_str(&sb, "scaffold");
        append_arg(&sb, args, "name", false);
        append_arg(&sb, args, "dest", false);
    } else if (strcmp(name, "code_context") == 0) {
        sb_append_str(&sb, "code_context");
        append_arg(&sb, args, "file", false);
        append_arg(&sb, args, "symbol", false);
    } else if (strcmp(name, "web_search") == 0) {
        sb_append_str(&sb, "web_search");
        append_arg(&sb, args, "query", true);
        append_arg(&sb, args, "recency", false);
    } else if (strcmp(name, "web_fetch") == 0) {
        sb_append_str(&sb, "web_fetch");
        append_arg(&sb, args, "url", true);
    } else if (strcmp(name, "readfile") == 0) {
        sb_append_str(&sb, "readfile");
        append_arg(&sb, args, "path", false);
        append_arg(&sb, args, "regex", true);
    } else if (strcmp(name, "docs_toc") == 0) {
        sb_append_str(&sb, "docs_toc");
    } else if (strcmp(name, "docs_get") == 0) {
        sb_append_str(&sb, "docs_get");
        append_arg(&sb, args, "path", false);
    } else if (strcmp(name, "docs_search") == 0) {
        sb_append_str(&sb, "docs_search");
        append_arg(&sb, args, "keyword", false);
    } else if (strcmp(name, "docs_recent_notes") == 0) {
        sb_append_str(&sb, "docs_recent_notes");
    } else if (strcmp(name, "docs_vector_search") == 0) {
        sb_append_str(&sb, "docs_vector_search");
        append_arg(&sb, args, "query", false);
    } else if (strcmp(name, "plan_write") == 0) {
        char *body = jx_get_string(args, "body");
        sb_append_str(&sb, "plan_write\n");
        sb_append_str(&sb, body ? body : "");
        free(body);
    } else if (strcmp(name, "assumptions") == 0) {
        char *items = jx_get_string(args, "items");
        sb_append_str(&sb, "assumptions\n");
        sb_append_str(&sb, items ? items : "");
        free(items);
    } else if (strcmp(name, "spike_write") == 0) {
        char *body = jx_get_string(args, "body");
        sb_append_str(&sb, "spike_write\n");
        sb_append_str(&sb, body ? body : "");
        free(body);
    } else if (strcmp(name, "study_write") == 0) {
        char *body = jx_get_string(args, "body");
        sb_append_str(&sb, "study_write\n");
        sb_append_str(&sb, body ? body : "");
        free(body);
    } else if (strcmp(name, "study_run") == 0) {
        sb_append_str(&sb, "study_run");
        append_arg(&sb, args, "slug", false);
    } else if (strcmp(name, "plan_verify") == 0) {
        sb_append_str(&sb, "plan_verify");
        append_arg(&sb, args, "id", false);
    } else {
        sb_free(&sb);
        return NULL;   /* unknown tool */
    }

    return sb_to_str(&sb);
}
