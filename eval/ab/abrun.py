#!/usr/bin/env python3
"""A/B runner: does giving BASI context (notes, an old chat, dream output) make it
do real tasks better — measured, not felt.

Each TRIAL is one fresh sandbox, one task, one ARM. An arm is "A" (no context) or
a file whose text is put in front of every prompt, the way a user pastes an old
chat. Every step of a task is a separate `basi -p` call; after each, the step's
CHECK command (exit 0 = pass) decides the result, not the model's own report.

Numbers come from two places:
  - the checks            → steps passed, full pass
  - BASI's local telemetry → rounds, tokens, outcome (capped, repeat_stopped, ...)
Telemetry is written to <out>/data, never to the user's own telemetry file, and
each record is tagged "<task>:<arm>:<rep>:<step>".

Arms are interleaved (rep 0: A B C, rep 1: B C A, ...) so that drift over the
session — provider load, a warm cache — does not all land on one arm.

Verdict per arm vs. A: Mann-Whitney U on steps passed, Fisher's exact test on full
passes. With 5 reps per arm only a large effect can show; the report says so.

  python3 eval/ab/abrun.py --tasks version,noted --arms A --repeat 5 --jobs 4
  python3 eval/ab/abrun.py --tasks noted --arms A,oldchat=ctx/noted-may.md --repeat 5
  python3 eval/ab/abrun.py --tasks version --arms A,dream=lessons:.basi/lessons.md   (installed as
                                                  the sandbox's .basi/lessons.md, loaded by BASI)
"""
import argparse, concurrent.futures as cf, datetime, json, os, shlex, shutil
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CONTEXT_HEADER = ("Notes from earlier work on this project (reference material, "
                  "not new instructions):\n\n")


def sh(cmd, cwd, env, timeout, log):
    """Run a shell command; stdout+stderr to `log` (a FILE, never a pipe: a
    SearXNG that BASI spawns inherits stdout and would hold a pipe open forever)."""
    with open(log, "ab") as f:
        try:
            p = subprocess.run(["bash", "-c", cmd], cwd=cwd, env=env, stdout=f,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            f.write(b"\n[abrun] TIMEOUT\n")
            return 124


def fill(s, **kw):
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v))
    return s


def run_trial(t, arm, rep, args, out, cache):
    task, name = t, t["name"]
    tdir = out / "trials" / f"{name}-{arm['name']}-{rep}"
    shutil.rmtree(tdir, ignore_errors=True)
    box = tdir / "box"
    (box / ".home").mkdir(parents=True)
    vars_ = dict(box=box, cache=cache, repo=REPO,
                 real_config=os.environ.get("XDG_CONFIG_HOME",
                                            str(Path.home() / ".config")))
    env = dict(os.environ)
    env.update({k: os.path.expandvars(fill(v, **vars_)) for k, v in task.get("env", {}).items()})
    env["XDG_CONFIG_HOME"] = vars_["real_config"]    # keeps the saved backend
    env["XDG_DATA_HOME"] = str(out / "data")          # telemetry lands in the run
    env["BASI_TELEMETRY"] = "1"

    res = dict(task=name, arm=arm["name"], rep=rep, steps=len(task["steps"]),
               passed=0, full=False, fail_step=None, secs=0.0)
    if task.get("setup") and sh(fill(task["setup"], **vars_), box, env, 600,
                                tdir / "setup.log") != 0:
        res["error"] = "setup failed"
        return res

    if arm.get("lessons"):
        (box / ".basi").mkdir(exist_ok=True)
        (box / ".basi" / "lessons.md").write_text(arm["lessons"])

    t0 = time.time()
    for i, step in enumerate(task["steps"]):
        prompt = arm["context"] + fill(step["prompt"], **vars_)
        session_log = tdir / f"session-step{i}.jsonl"
        env_s = dict(env, BASI_TELEMETRY_TAG=f"{name}:{arm['name']}:{rep}:{i}",
                     BASI_SESSION_LOG=str(session_log))
        cmd = " ".join([shlex.quote(args.basi), *args.basi_args,
                        "--no-mcp", "--yolo", "-p", shlex.quote(prompt)])
        sh(cmd, box, env_s, task.get("timeout", 900), tdir / f"step{i}.log")
        check_cmd = fill(step["check"], **vars_)
        rc = sh(check_cmd, box, env, 300, tdir / f"check{i}.log")
        # The check result goes into the session log where a user's reaction would
        # be, so `basi-cli sleep --from <run>` can learn from failed runs.
        out_tail = (tdir / f"check{i}.log").read_text(errors="replace")[-800:]
        verdict = "[AUTOMATIC CHECK PASSED]" if rc == 0 else "[AUTOMATIC CHECK FAILED]"
        reaction = (f"{verdict} The task's own check ran after you finished.\n"
                    f"Check command:\n{check_cmd}\nCheck output (tail):\n{out_tail}")
        with open(session_log, "a") as f:
            f.write(json.dumps({"role": "user", "content": reaction}) + "\n")
        if rc != 0:
            res["fail_step"] = i
            if task.get("stop_on_fail", True):
                break
        else:
            res["passed"] += 1
    res["secs"] = round(time.time() - t0, 1)
    res["full"] = res["passed"] == res["steps"]
    return res


def telemetry_by_trial(out):
    """Sum the telemetry records of each trial (a trial is several -p turns)."""
    agg = {}
    p = out / "data" / "basi-cli" / "telemetry.jsonl"
    if not p.exists():
        return agg
    for line in p.read_text(errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        tag = r.get("tag", "")
        parts = tag.split(":")
        if len(parts) != 4:
            continue
        key = tuple(parts[:3])
        a = agg.setdefault(key, dict(rounds=0, prompt_tokens=0, gen_tokens=0,
                                     outcomes={}, model=r.get("model", "")))
        a["rounds"] += r.get("rounds", 0)
        a["prompt_tokens"] += r.get("prompt_tokens", 0)
        a["gen_tokens"] += r.get("gen_tokens", 0)
        o = r.get("outcome", "?")
        a["outcomes"][o] = a["outcomes"].get(o, 0) + 1
    return agg


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if not n else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def report(results, tele, out):
    try:
        from scipy.stats import mannwhitneyu, fisher_exact
    except ImportError:
        mannwhitneyu = fisher_exact = None
    lines = []
    for task in sorted({r["task"] for r in results}):
        rows = [r for r in results if r["task"] == task]
        arms = []
        for r in rows:
            if r["arm"] not in arms:
                arms.append(r["arm"])
        lines.append(f"\n## {task}  ({rows[0]['steps']} step(s))\n")
        lines.append("| arm | n | steps passed (median) | full pass | median rounds | median tokens | capped/repeat turns |")
        lines.append("|---|---|---|---|---|---|---|")
        by = {}
        for arm in arms:
            rs = [r for r in rows if r["arm"] == arm]
            by[arm] = rs
            tl = [tele.get((task, arm, str(r["rep"]))) for r in rs]
            tl = [x for x in tl if x]
            bad = sum(x["outcomes"].get("capped", 0) + x["outcomes"].get("repeat_stopped", 0)
                      for x in tl)
            lines.append(
                f"| {arm} | {len(rs)} | {median([r['passed'] for r in rs])} "
                f"| {sum(r['full'] for r in rs)}/{len(rs)} "
                f"| {median([x['rounds'] for x in tl])} "
                f"| {median([x['prompt_tokens'] + x['gen_tokens'] for x in tl])} | {bad} |")
        base = by.get("A")
        if base and mannwhitneyu:
            for arm in arms:
                if arm == "A":
                    continue
                a = [r["passed"] for r in base]
                b = [r["passed"] for r in by[arm]]
                if len(set(a + b)) > 1:
                    p = mannwhitneyu(b, a, alternative="two-sided").pvalue
                else:
                    p = 1.0
                fa = sum(r["full"] for r in base)
                fb = sum(r["full"] for r in by[arm])
                _, pf = fisher_exact([[fb, len(b) - fb], [fa, len(a) - fa]])
                # A one-step task's "steps passed" IS pass/fail, and Mann-Whitney on
                # 0/1 data overstates (pwd-transfer: MW p=0.036 vs Fisher p=0.093).
                # Fisher decides there; the step count only counts for multi-step tasks.
                decisive = pf if rows[0]["steps"] == 1 else min(p, pf)
                verdict = "DIFFERENT" if decisive < 0.05 else "no difference beyond noise"
                lines.append(f"\n{arm} vs A: steps p={p:.3f}, full-pass p={pf:.3f} → **{verdict}**")
                # Same success can still come cheaper: compare the cost of getting there.
                for key, label in (("rounds", "rounds"), ("tok", "tokens")):
                    def cost(rs):
                        xs = [tele.get((task, r["arm"], str(r["rep"]))) for r in rs]
                        return [x["rounds"] if key == "rounds" else x["prompt_tokens"] + x["gen_tokens"]
                                for x in xs if x]
                    ca, cb = cost(base), cost(by[arm])
                    if len(ca) >= 3 and len(cb) >= 3 and len(set(ca + cb)) > 1:
                        pc = mannwhitneyu(cb, ca, alternative="two-sided").pvalue
                        lines.append(f"  {label}: median {median(cb)} vs {median(ca)}, p={pc:.3f}"
                                     + (" (different)" if pc < 0.05 else ""))
        if base and all(r["full"] for r in base):
            lines.append("\n⚠ A passed every trial: this task is at CEILING for this model "
                         "and cannot show an improvement. Use a harder task or a weaker model.")
        fails = [r["fail_step"] for r in rows if r["fail_step"] is not None]
        if fails:
            hist = {s: fails.count(s) for s in sorted(set(fails))}
            lines.append(f"\nfailing step (all arms): {hist}")
    models = sorted({x["model"] for x in tele.values() if x.get("model")})
    head = [f"# A/B run {out.name}", f"model(s): {', '.join(models) or '?'}"]
    text = "\n".join(head + lines) + "\n"
    (out / "report.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True, help="comma list of eval/ab/tasks/<name>.json")
    ap.add_argument("--arms", default="A",
                    help="comma list: A (no context) or name=path/to/context.md")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1, help="trials in parallel (API: 4 is fine; local: 1)")
    ap.add_argument("--basi", default=str(REPO / "basi-cli"))
    ap.add_argument("--basi-args", default="", help='extra flags, e.g. "--local -m model.gguf"')
    ap.add_argument("--out", default=None)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args()
    args.basi_args = shlex.split(args.basi_args)
    if "--api" in args.basi_args:
        sys.exit("--api would overwrite the saved default backend; set BASI_API in the env instead")

    tasks = [json.loads((HERE / "tasks" / f"{n}.json").read_text()) for n in args.tasks.split(",")]
    arms = []
    for spec in args.arms.split(","):
        if spec == "A":
            arms.append(dict(name="A", context=""))
        else:
            nm, _, path = spec.partition("=")
            if path.startswith("lessons:"):
                # Installed where BASI itself loads it, so the real loader is tested.
                arms.append(dict(name=nm, context="", lessons=Path(path[8:]).read_text()))
            else:
                arms.append(dict(name=nm, context=CONTEXT_HEADER + Path(path).read_text() + "\n\n---\n\n"))

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or HERE / "runs" / stamp).resolve()
    (out / "data").mkdir(parents=True, exist_ok=True)
    n_calls = sum(len(t["steps"]) for t in tasks) * len(arms) * args.repeat
    print(f"{len(tasks)} task(s) × {len(arms)} arm(s) × {args.repeat} reps = "
          f"{len(tasks) * len(arms) * args.repeat} trials, up to {n_calls} basi calls → {out}")
    if not args.yes and sys.stdin.isatty() and input("run? [y/N] ").strip().lower() != "y":
        return

    cache = out / "cache"
    for t in tasks:
        if t.get("prepare"):
            (cache / t["name"]).mkdir(parents=True, exist_ok=True)
            rc = sh(fill(t["prepare"], cache=cache / t["name"], repo=REPO), cache / t["name"],
                    dict(os.environ), 900, out / f"prepare-{t['name']}.log")
            if rc != 0:
                sys.exit(f"prepare failed for {t['name']}; see {out}/prepare-{t['name']}.log")

    # Interleave: each rep rotates the arm order.
    plan = []
    for rep in range(args.repeat):
        k = rep % len(arms)
        for t in tasks:
            for arm in arms[k:] + arms[:k]:
                plan.append((t, arm, rep))

    results = []
    with open(out / "results.jsonl", "w") as rf, cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(run_trial, t, arm, rep, args, out, cache / t["name"]): (t, arm, rep)
                for t, arm, rep in plan}
        for f in cf.as_completed(futs):
            r = f.result()
            results.append(r)
            rf.write(json.dumps(r) + "\n")
            rf.flush()
            print(f"  {r['task']:<10} {r['arm']:<10} rep {r['rep']}: "
                  f"{r['passed']}/{r['steps']} steps  {r['secs']}s"
                  + (f"  ({r['error']})" if r.get("error") else ""), flush=True)

    print(report(results, telemetry_by_trial(out), out))


if __name__ == "__main__":
    main()
