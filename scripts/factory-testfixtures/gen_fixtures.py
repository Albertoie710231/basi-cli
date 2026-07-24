#!/usr/bin/env python3
"""Generate throwaway fixtures for exercising `basi-cli factory` end-to-end.

Usage: python3 gen_fixtures.py [OUTDIR]   (default: ./_fix)

Creates four directories under OUTDIR.

CAPABILITY fixtures (does the factory find a real win?):
  small/  speed.c  — a ~15-line O(N^2) busy loop; --file injected verbatim (<48KB).
  big/    big.c    — the same hot loop buried in 2600 filler fns (~170KB) so the
                     --file path exceeds the 48KB inject cap and takes the RAG
                     retrieval fallback (mem_add/mem_retrieve).
Each gets measure.sh (prints the elapsed ms) and verify.sh (correctness oracle:
the program must still compute s == N*N == 400000000; a cheat like "reduce N" fails).

GATE fixtures (do the REJECT and NO-WIN paths actually fire?):
  gate_reject/ — measure.sh emits a token derived from the SOURCE, so ANY edit flips it.
  gate_noise/  — measure.sh times a pristine ref.c and is IMMUNE to the edit.
These exist because waiting for the model to happen to propose a suitable theory is not
a test: a run where nothing is rejected proves nothing about the reject path. Here the
outcome is forced from the MEASURE side, so it holds whatever the model proposes.

Then, with the 35B up on :8181 (see project-factory-command memory), run e.g.:
  cd OUTDIR/big && basi-cli factory \
    --question "make the hot_loop computation in big.c faster; the metric is the \
milliseconds ./measure.sh prints (lower is better)" \
    --measure ./measure.sh --file big.c --expect 's=400000000' \
    --minimize --theories 3 --timeout 120
(Prefer --expect over --verify ./verify.sh: one build+run emits metric + s= token.)

The gate fixtures print their own ready-to-run command at the end of this script.
"""
import os, sys, stat

OUT = sys.argv[1] if len(sys.argv) > 1 else "_fix"

# Compile noise suppressed; the binary's own stderr (correctness token s=...) is kept so
# `factory --expect` can fold verify into this single run.
MEASURE = ("#!/usr/bin/env bash\n"
           "cc -O2 -o /tmp/factory_speed_bin {src} 2>/dev/null || {{ echo 999999; exit 0; }}\n"
           "/tmp/factory_speed_bin\n")

# correctness oracle: s must stay N*N = 20000*20000 = 400000000
VERIFY = ("#!/usr/bin/env bash\n"
          "cc -O2 -o /tmp/factory_verify_bin {src} 2>/dev/null || exit 1\n"
          "/tmp/factory_verify_bin >/dev/null 2>/tmp/factory_verify_err || exit 1\n"
          "grep -q 's=400000000' /tmp/factory_verify_err && exit 0 || exit 1\n")

def hot_main(marker=True):
    L = []
    L.append("int main(void){")
    L.append("    struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);")
    L.append("    volatile long s=0;")
    if marker:
        L.append("    /* hot_loop: the timed inner computation — this is the region to optimize */")
    L.append("    for(int i=0;i<N;i++)")
    L.append("        for(int j=0;j<N;j++)")
    L.append("            s+=1;")
    L.append("    clock_gettime(CLOCK_MONOTONIC,&b);")
    L.append("    double ms=(b.tv_sec-a.tv_sec)*1e3+(b.tv_nsec-a.tv_nsec)/1e6;")
    return L

def write_dir(d, src_name, lines):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, src_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    for name, tmpl in (("measure.sh", MEASURE), ("verify.sh", VERIFY)):
        p = os.path.join(d, name)
        with open(p, "w") as f:
            f.write(tmpl.format(src=src_name))
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

# --- small ---
small = ["#include <stdio.h>", "#include <time.h>", "#define N 20000"] + hot_main()
small += ['    fprintf(stderr,"s=%ld\\n", s);', '    printf("%.3f\\n", ms);', "    return 0;", "}"]
write_dir(os.path.join(OUT, "small"), "speed.c", small)

# --- big (pad past 48KB) ---
big = ["#include <stdio.h>", "#include <time.h>", "#define N 20000"]
for i in range(2600):
    big.append("static long filler_%04d(long x){ return (x ^ %d) + (x %% %d); }" % (i, i*7+1, (i%97)+3))
big.append("")
big += hot_main()
big += ['    fprintf(stderr,"s=%ld filler=%ld\\n", s, filler_0001(s));',
        '    printf("%.3f\\n", ms);', "    return 0;", "}"]
write_dir(os.path.join(OUT, "big"), "big.c", big)

sb = os.path.getsize(os.path.join(OUT, "big", "big.c"))
print("wrote %s/small/speed.c and %s/big/big.c (%d bytes; >48KB => retrieval fallback)" % (OUT, OUT, sb))

# ── GATE fixtures ────────────────────────────────────────────────────────────────
# Each ships speed.c (what the model edits) plus ref.c (a pristine copy the harness
# keeps). measure.sh uses ref.c to force a specific verdict regardless of the theory.

# --expect 'src=pristine' can only hold while speed.c is byte-identical to ref.c, so
# EVERY applied theory must be rejected. cmp, not a hash, so the token stays stable
# across regenerations and there is nothing to paste.
GATE_REJECT_MEASURE = (
    "#!/usr/bin/env bash\n"
    "# Proves the --expect REJECT path. The correctness token comes from the SOURCE, not\n"
    "# the program output, so any edit at all flips it to src=EDITED and the trial is\n"
    "# rejected -- whatever the model proposed.\n"
    "cc -O2 -o /tmp/factory_gate_bin speed.c 2>/dev/null || {{ echo 999999; exit 0; }}\n"
    "cmp -s speed.c ref.c && echo 'src=pristine' >&2 || echo 'src=EDITED' >&2\n"
    "/tmp/factory_gate_bin\n")

# Times the pristine copy, so no edit can move the metric: every theory must land inside
# the baseline's noise band and the run must conclude NO WIN. Needs --repeat >1, since a
# band needs more than one sample.
GATE_NOISE_MEASURE = (
    "#!/usr/bin/env bash\n"
    "# Proves the noise gate. Times ref.c, NOT the file the model was told to edit, so the\n"
    "# metric is immune to every theory; any apparent delta is pure run-to-run wobble and\n"
    "# must be reported as INSIDE NOISE / NO WIN.\n"
    "cc -O2 -o /tmp/factory_noise_bin ref.c 2>/dev/null || {{ echo 999999; exit 0; }}\n"
    "/tmp/factory_noise_bin\n")

def write_gate_dir(d, measure):
    os.makedirs(d, exist_ok=True)
    body = "\n".join(small) + "\n"
    for name in ("speed.c", "ref.c"):            # ref.c is the pristine reference copy
        with open(os.path.join(d, name), "w") as f:
            f.write(body)
    p = os.path.join(d, "measure.sh")
    with open(p, "w") as f:
        f.write(measure.format(src="speed.c"))
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

write_gate_dir(os.path.join(OUT, "gate_reject"), GATE_REJECT_MEASURE)
write_gate_dir(os.path.join(OUT, "gate_noise"),  GATE_NOISE_MEASURE)

Q = ("make the computation in speed.c faster; the metric is the milliseconds "
     "./measure.sh prints (lower is better)")
print("""
wrote %s/gate_reject and %s/gate_noise (gate regression tests)

  # must print REJECTED for every theory:
  cd %s/gate_reject && basi-cli factory --question "%s" --measure ./measure.sh --file speed.c --expect 'src=pristine' --minimize --theories 2 --timeout 120

  # must print [INSIDE NOISE] and end with "No theory beat the noise band":
  cd %s/gate_noise && basi-cli factory --question "%s" --measure ./measure.sh --file speed.c --expect 's=400000000' --minimize --theories 2 --repeat 5 --timeout 120
""" % (OUT, OUT, OUT, Q, OUT, Q))
