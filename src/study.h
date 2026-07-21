#ifndef BASI_STUDY_H
#define BASI_STUDY_H

#include <stdbool.h>
#include <stddef.h>

/* ── The `study` artifact: a falsifiable experiment, run unattended ────
 *
 * A study is BASI's unit of discovery. It pairs a hypothesis with a COMMAND
 * that measures it, so the verdict is computed from a number the model does
 * not control. The model proposes and interprets; everything between — run
 * the command n times, extract the metric, apply the decision rule, decide
 * SUPPORTED/REFUTED — is deterministic C, on purpose. A verdict must not be
 * talkable-into-existence by the thing under test.
 *
 * Artifact lives at .basi/studies/<slug>.md and is validated the way plan.c
 * validates plans: all failures accumulated, addressed to the model, always
 * saying what to do next.
 */

#define KB_STUDIES_DIR ".basi/studies"

#define STUDY_MAX_ARMS        8
#define STUDY_MAX_RUNS        50
#define STUDY_MIN_RUNS         5   /* measured noise floor demands it; see below */
#define STUDY_DEFAULT_TIMEOUT 600
#define STUDY_MAX_OUTPUT (1u << 20)

/* How a single execution of an arm's command ended. Kept separate from the
 * metric value because censored/failed runs silently biased earlier work:
 * a capped run reports its ceiling and looks like a real measurement. */
typedef enum {
    RUN_OK = 0,        /* exit 0 and the metric regex matched */
    RUN_FAILED,        /* non-zero exit */
    RUN_TIMEOUT,       /* killed at the deadline */
    RUN_NO_METRIC      /* exit 0 but nothing matched the extract regex */
} StudyRunStatus;

typedef struct {
    double         value;
    StudyRunStatus status;
    int            exit_code;
    double         seconds;
    /* A short excerpt of what the command actually printed, kept ONLY for runs
     * that produced no usable number. Without it the loop debugs blind: it is
     * told "no-metric rc=0" and cannot tell a wrong flag from a wrong regex from
     * a command that simply prints nothing. Measured: three unattended rounds
     * were spent guessing at a CLI whose error message was right there. */
    char          *sample;
} StudyRun;

typedef struct {
    char     *name;                    /* arm label, e.g. "A" */
    char     *command;                 /* shell command to execute */
    StudyRun  runs[STUDY_MAX_RUNS];
    int       nruns;                   /* runs attempted */
    /* Fingerprint of the working tree taken just before this arm's first run.
     * Arms execute sequentially, so an arm that edits the code under test
     * silently changes what every LATER arm measures. Comparing fingerprints
     * across arms catches that; see study_execute(). */
    unsigned long tree_hash;

    /* Worst ratio between the largest and smallest DISTINCT numbers the extract
     * regex matched within a SINGLE run. 1.0 means it matched one quantity. A
     * large value means the regex is not identifying the metric — measured on
     * `zstd -b`, which prints compression and decompression throughput on one
     * line, so one arm scored ~2580 (decompression) against another's ~405
     * (compression) and the loop called it SUPPORTED at p=1e-10. */
    double metric_spread;
    double metric_lo, metric_hi;  /* the conflicting values, for the error message */
} StudyArm;

typedef struct {
    char    *slug;
    char    *metric;                   /* human name of the measured quantity */
    char    *extract;                  /* POSIX ERE, capture group 1 = the number */
    char    *decision_rule;            /* see grammar in study.c */
    int      runs;                     /* executions per arm */
    int      timeout_s;
    StudyArm arms[STUDY_MAX_ARMS];
    int      narms;
} Study;

/* Aggregate statistics over an arm's OK runs only. */
typedef struct {
    int    n;
    double mean, median, sd, min, max;
} StudyStats;

typedef enum {
    VERDICT_SUPPORTED = 0,
    VERDICT_REFUTED,
    VERDICT_INCONCLUSIVE   /* not enough valid runs to decide honestly */
} StudyVerdict;

const char *study_verdict_name(StudyVerdict v);

/* Returns NULL if the artifact is valid, otherwise a malloc'd, human-readable
 * description of ALL validation failures (one "- ..." per line). */
char *validate_study_artifact(const char *content, size_t content_len);

/* Parse a validated artifact into `out`. Returns NULL on success, else a
 * malloc'd error string. Caller must study_free(out) on success. */
char *study_parse(const char *content, size_t content_len, Study *out);
void  study_free(Study *s);

/* Statistics over an arm's OK runs. */
StudyStats study_arm_stats(const StudyArm *arm);

/* Two-sided Mann-Whitney U p-value for two independent samples. Exact when
 * both n <= 20 and there are no ties; normal approximation with tie
 * correction otherwise. Returns -1.0 if either sample has n < 2. */
double study_mannwhitney_p(const double *a, int na, const double *b, int nb);

/* Execute every arm `s->runs` times, filling in s->arms[].runs. Blocking.
 * `progress` (may be NULL) is called after each individual run. */
typedef void (*StudyProgressFn)(const char *arm, int run, int of, void *ud);
void study_execute(Study *s, StudyProgressFn progress, void *ud);

/* Apply the decision rule to executed arms. Returns the verdict and, if
 * `detail` is non-NULL, stores a malloc'd explanation of how it was computed
 * (the substituted values, the p-value, and why). */
StudyVerdict study_decide(const Study *s, char **detail);

/* Run a study end to end: read .basi/studies/<slug>.md, execute, decide, and
 * write the Results and Verdict sections back into the file. Returns a
 * malloc'd human-readable report; never NULL. */
char *study_run_slug(const char *slug, StudyProgressFn progress, void *ud);

/* Tool entry point: validate + write .basi/studies/<slug>.md. */
char *execute_study_write(const char *body);

/* ── The outer loop ────────────────────────────────────────────────────
 * hypothesis -> experiment -> verdict -> next hypothesis, unattended and
 * bounded. Round N's artifact is written by the model from round N-1's
 * COMPUTED verdict; running it and judging it stay deterministic.
 *
 * The loop executes model-authored shell commands with no human watching, so
 * new arms are constrained by `allow_commands:` in the seed's frontmatter (a
 * comma-separated list of permitted command prefixes) and may not contain
 * shell chaining metacharacters. `unsafe` lifts both checks.
 */
typedef struct {
    int  max_rounds;
    int  port;         /* llama-server port for hypothesis generation */
    bool unsafe;       /* skip the arm-command allowlist */

    /* Start from a QUESTION rather than a hand-written seed artifact. This is
     * what makes the loop general-purpose: point it at any question in any
     * domain and it writes the first study itself — the metric, the extract
     * regex, the arms and the decision rule — instead of a human pre-encoding
     * the experiment. `allow` then comes from the CLI rather than the artifact,
     * because a model that writes its own allowlist has no allowlist. */
    const char *question;   /* NULL = seed from an existing artifact */
    const char *allow;      /* comma-separated command prefixes; required with question */

    /* Independent explorations of the SAME question, each told what the earlier
     * ones covered and pushed elsewhere. ROBIN's breadth mechanism: it generates
     * N *distinct* ideas and ranks them rather than iterating one chain. A single
     * trajectory stops at its first success — measured, one found flash attention
     * worth +1.5% and halted without touching batch size, threads or offload. */
    int trajectories;       /* <=1 = single chain */

    /* Dispatch the trajectories CONCURRENTLY (fork one child per trajectory, in
     * waves) instead of one after another, so their /v1/chat/completions
     * requests are in flight together and the server batches them across its
     * parallel KV slots. Requires the launch script to carry `-np N
     * --kv-unified` or the requests serialise onto fewer slots. Only sound when
     * the arms are write-isolated by construction — concurrent arms in the same
     * working directory can corrupt each other's measurement. See
     * study_loop_concurrent() for the full set of trade-offs. Ignored unless
     * trajectories > 1. */
    bool concurrent;
} StudyLoopOpts;

/* seed_slug names an existing artifact, or is the slug to CREATE when
 * opts->question is set. */
char *study_loop(const char *seed_slug, const StudyLoopOpts *opts);

/* CLI: basi-cli study <run|list|show> ... */
int cmd_study(int argc, char **argv);

#endif /* BASI_STUDY_H */
