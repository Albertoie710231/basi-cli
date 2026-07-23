#!/usr/bin/env python3
# Apply a theory's SEARCH->REPLACE to a file. Exact match first, then a
# per-line-whitespace-tolerant match (models often get indentation slightly off).
# Prints APPLIED / APPLIED_FUZZY / SEARCH_NOT_FOUND; exit 1 on not found.
import sys
f, sf, rf = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(f, encoding="utf-8", errors="replace").read()
search = open(sf).read().strip("\n")
replace = open(rf).read().strip("\n")
if not search:
    print("EMPTY_SEARCH"); sys.exit(1)

if search in txt:
    open(f, "w").write(txt.replace(search, replace, 1))
    print("APPLIED"); sys.exit(0)

# fuzzy: match a window of lines whose stripped content equals the stripped search
def strip_lines(s):
    return [ln.strip() for ln in s.splitlines()]
want = strip_lines(search)
lines = txt.splitlines(keepends=True)
raw = txt.splitlines()
for i in range(len(raw) - len(want) + 1):
    if [ln.strip() for ln in raw[i:i+len(want)]] == want:
        # preserve the indentation of the first matched line for the replacement
        indent = raw[i][:len(raw[i]) - len(raw[i].lstrip())]
        new_block = "\n".join((indent + ln.lstrip()) if ln.strip() else ln
                              for ln in replace.splitlines())
        out = "".join(lines[:i]) + new_block + "\n" + "".join(lines[i+len(want):])
        open(f, "w").write(out)
        print("APPLIED_FUZZY"); sys.exit(0)
print("SEARCH_NOT_FOUND"); sys.exit(1)
