#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include "util.h"
#include "globals.h"
#include "scaffold.h"

/* ── Scaffold templates ──────────────────────────────────────────── */

typedef struct {
    char       *root;
    const char *label;
} TemplateRoot;

static int discover_template_roots(TemplateRoot roots[2]) {
    int n = 0;
    struct stat st;

    if (stat("./.basi/templates", &st) == 0 && S_ISDIR(st.st_mode)) {
        roots[n].root = strdup("./.basi/templates");
        roots[n].label = "project";
        n++;
    }
    const char *home = getenv("HOME");
    if (home) {
        char user[1024];
        snprintf(user, sizeof(user), "%s/.config/basi-cli/templates", home);
        if (stat(user, &st) == 0 && S_ISDIR(st.st_mode)) {
            roots[n].root = strdup(user);
            roots[n].label = "user";
            n++;
        }
    }
    return n;
}

static void free_template_roots(TemplateRoot roots[2], int n) {
    for (int i = 0; i < n; i++) free(roots[i].root);
}

/* Read 'description: ...' or 'post_message: ...' from <dir>/_meta. */
static char *read_meta_field(const char *template_dir, const char *field) {
    char meta_path[2048];
    snprintf(meta_path, sizeof(meta_path), "%s/_meta", template_dir);
    FILE *f = fopen(meta_path, "r");
    if (!f) return strdup("");

    size_t flen = strlen(field);
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, field, flen) == 0 && line[flen] == ':') {
            const char *v = line + flen + 1;
            while (*v == ' ' || *v == '\t') v++;
            size_t len = strlen(v);
            while (len > 0 && (v[len-1] == '\n' || v[len-1] == '\r' || v[len-1] == ' ')) len--;
            char *r = malloc(len + 1);
            memcpy(r, v, len);
            r[len] = '\0';
            fclose(f);
            return r;
        }
    }
    fclose(f);
    return strdup("");
}

/* Build a string like "  - name1 — desc1\n  - name2 — desc2\n" across all roots.
 * Returns malloc'd string. Project-scope templates shadow user-scope by name. */
char *build_templates_index(void) {
    TemplateRoot roots[2];
    int nroots = discover_template_roots(roots);

    StringBuf sb;
    sb_init(&sb);
    char **seen = NULL;
    int n_seen = 0, cap_seen = 0;

    for (int i = 0; i < nroots; i++) {
        DIR *d = opendir(roots[i].root);
        if (!d) continue;
        struct dirent *e;
        while ((e = readdir(d))) {
            if (e->d_name[0] == '.') continue;

            bool dup = false;
            for (int j = 0; j < n_seen; j++)
                if (strcmp(seen[j], e->d_name) == 0) { dup = true; break; }
            if (dup) continue;

            char tdir[2048];
            snprintf(tdir, sizeof(tdir), "%s/%s", roots[i].root, e->d_name);
            struct stat st;
            if (stat(tdir, &st) != 0 || !S_ISDIR(st.st_mode)) continue;

            char *desc = read_meta_field(tdir, "description");
            char line[1024];
            snprintf(line, sizeof(line), "  - %s — %s\n",
                     e->d_name, *desc ? desc : "(no description)");
            sb_append_str(&sb, line);
            free(desc);

            if (n_seen >= cap_seen) {
                cap_seen = cap_seen ? cap_seen * 2 : 8;
                seen = realloc(seen, cap_seen * sizeof(char *));
            }
            seen[n_seen++] = strdup(e->d_name);
        }
        closedir(d);
    }

    for (int i = 0; i < n_seen; i++) free(seen[i]);
    free(seen);
    free_template_roots(roots, nroots);

    if (sb.len == 0) {
        sb_free(&sb);
        return strdup("  (none — to add one: mkdir -p ~/.config/basi-cli/templates/<name>, drop files inside, and add a '_meta' file with 'description: ...')\n");
    }
    return sb_to_str(&sb);
}

/* Find a template by name. Returns malloc'd absolute path or NULL. */
static char *find_template_dir(const char *name) {
    if (!name || !*name) return NULL;
    for (const char *p = name; *p; p++)
        if (*p == '/' || *p == '\\') return NULL;
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) return NULL;

    TemplateRoot roots[2];
    int nroots = discover_template_roots(roots);
    char *result = NULL;
    for (int i = 0; i < nroots; i++) {
        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", roots[i].root, name);
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
            result = strdup(path);
            break;
        }
    }
    free_template_roots(roots, nroots);
    return result;
}

/* Recursively copy template tree (skipping '_meta'). Returns NULL on success,
 * heap error string on failure. Appends each created file path to created. */
static char *copy_template_dir(const char *src, const char *dst, StringBuf *created) {
    if (mkdir_p(dst) != 0) {
        char *e = malloc(512);
        snprintf(e, 512, "scaffold: cannot create '%s' (%s)", dst, strerror(errno));
        return e;
    }

    DIR *d = opendir(src);
    if (!d) {
        char *e = malloc(512);
        snprintf(e, 512, "scaffold: cannot open template '%s'", src);
        return e;
    }

    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        if (strcmp(ent->d_name, "_meta") == 0) continue;

        char src_path[2048], dst_path[2048];
        snprintf(src_path, sizeof(src_path), "%s/%s", src, ent->d_name);
        snprintf(dst_path, sizeof(dst_path), "%s/%s", dst, ent->d_name);

        struct stat st;
        if (stat(src_path, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            char *e = copy_template_dir(src_path, dst_path, created);
            if (e) { closedir(d); return e; }
        } else if (S_ISREG(st.st_mode)) {
            struct stat dst_st;
            if (stat(dst_path, &dst_st) == 0) {
                char *e = malloc(512);
                snprintf(e, 512, "scaffold: '%.300s' already exists; refusing to overwrite", dst_path);
                closedir(d);
                return e;
            }
            FILE *fs = fopen(src_path, "rb");
            FILE *fd = fopen(dst_path, "wb");
            if (!fs || !fd) {
                if (fs) fclose(fs);
                if (fd) fclose(fd);
                char *e = malloc(512);
                snprintf(e, 512, "scaffold: cannot copy '%.180s' -> '%.180s'", src_path, dst_path);
                closedir(d);
                return e;
            }
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), fs)) > 0)
                fwrite(buf, 1, n, fd);
            fclose(fs);
            fclose(fd);
            sb_append_str(created, dst_path);
            sb_append_char(created, '\n');
        }
    }
    closedir(d);
    return NULL;
}

char *execute_scaffold(const char *args) {
    while (*args == ' ' || *args == '\t' || *args == '\n') args++;
    if (!*args) {
        return strdup("Error: scaffold requires a template name. Usage: <tool>scaffold <name> [<dest>]</tool> or <tool>scaffold list</tool>");
    }

    /* scaffold list */
    if (strncmp(args, "list", 4) == 0 &&
        (args[4] == '\0' || args[4] == ' ' || args[4] == '\t' || args[4] == '\n')) {
        char *idx = build_templates_index();
        size_t need = strlen(idx) + 32;
        char *result = malloc(need);
        snprintf(result, need, "Available templates:\n%s", idx);
        free(idx);
        return result;
    }

    /* parse name */
    const char *name_end = args;
    while (*name_end && *name_end != ' ' && *name_end != '\t' && *name_end != '\n') name_end++;
    size_t name_len = name_end - args;
    char *name = malloc(name_len + 1);
    memcpy(name, args, name_len);
    name[name_len] = '\0';

    /* parse dest (default ".") */
    const char *dest_start = name_end;
    while (*dest_start == ' ' || *dest_start == '\t' || *dest_start == '\n') dest_start++;
    char *dest;
    if (*dest_start) {
        const char *dest_end = dest_start;
        while (*dest_end && *dest_end != ' ' && *dest_end != '\t' && *dest_end != '\n') dest_end++;
        size_t dest_len = dest_end - dest_start;
        dest = malloc(dest_len + 1);
        memcpy(dest, dest_start, dest_len);
        dest[dest_len] = '\0';
    } else {
        dest = strdup(".");
    }

    char *template_dir = find_template_dir(name);
    if (!template_dir) {
        char *e = malloc(512 + name_len);
        snprintf(e, 512 + name_len,
            "scaffold: template '%s' not found. Use <tool>scaffold list</tool> to see available templates.", name);
        free(name);
        free(dest);
        return e;
    }

    bool sc_auto = (permission_mode == PERM_BYPASS)
                || (permission_mode == PERM_ACCEPT_EDITS)
                || scaffold_always_allowed;
    if (!sc_auto) {
        char prompt_line[1024];
        snprintf(prompt_line, sizeof(prompt_line), "%s -> %s", name, dest);
        int decision = request_approval("scaffold", prompt_line);
        if (decision == 0) {
            free(name); free(dest); free(template_dir);
            return strdup("User denied execution.");
        }
        if (decision == 2) scaffold_always_allowed = true;
    }

    StringBuf created;
    sb_init(&created);
    char *err = copy_template_dir(template_dir, dest, &created);
    char *post = read_meta_field(template_dir, "post_message");

    StringBuf result;
    sb_init(&result);
    if (err) {
        sb_append_str(&result, err);
        free(err);
    } else {
        sb_append_str(&result, "Created files:\n");
        sb_append_str(&result, created.len ? created.data : "(none)\n");
        if (*post) {
            sb_append_str(&result, "\nNote: ");
            sb_append_str(&result, post);
            sb_append_char(&result, '\n');
        }
    }

    sb_free(&created);
    free(post);
    free(name);
    free(dest);
    free(template_dir);
    return sb_to_str(&result);
}
