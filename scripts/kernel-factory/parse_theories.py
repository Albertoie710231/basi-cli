#!/usr/bin/env python3
# Parse the theory phase's output into per-theory {file, search, replace, desc}.
# Format (robust, unambiguous markers unlikely to appear in C):
#   @@THEORY@@
#   desc: <one line>
#   file: <relative path>
#   @@SEARCH@@
#   <exact current lines>
#   @@REPLACE@@
#   <exact new lines>
#   @@END@@
import sys, os, re
src = open(sys.argv[1], encoding="utf-8", errors="replace").read()
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)
blocks = re.findall(r'@@THEORY@@(.*?)@@END@@', src, re.DOTALL)
n = 0
for b in blocks:
    fm = re.search(r'^file:\s*(.+?)\s*$', b, re.MULTILINE)
    dm = re.search(r'^desc:\s*(.+?)\s*$', b, re.MULTILINE)
    sm = re.search(r'@@SEARCH@@\r?\n(.*?)\r?\n@@REPLACE@@', b, re.DOTALL)
    rm = re.search(r'@@REPLACE@@\r?\n(.*)$', b, re.DOTALL)
    if not (fm and sm and rm):
        continue
    search = sm.group(1)
    replace = rm.group(1)
    # trim a single trailing newline the format may add before @@END@@ (already stripped by @@END@@ split)
    n += 1
    base = os.path.join(outdir, f"theory-{n}")
    open(base + ".file", "w").write(fm.group(1).strip())
    open(base + ".desc", "w").write(dm.group(1).strip() if dm else f"theory {n}")
    open(base + ".search", "w").write(search)
    open(base + ".replace", "w").write(replace)
print(n)
