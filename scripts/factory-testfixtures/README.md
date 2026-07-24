# factory test fixtures

Throwaway fixtures for exercising `basi-cli factory` end-to-end (see the
`project-factory-command` memory for the full design + open items).

```
python3 gen_fixtures.py _fix     # writes _fix/{small,big,gate_reject,gate_noise}
```

## Capability fixtures — can the factory find a real win?

- `small/speed.c` — ~15-line O(N²) busy loop, <48KB → `--file` is injected verbatim.
- `big/big.c` — same hot loop among 2600 filler fns, ~170KB → `--file` exceeds the
  48KB inject cap and takes the RAG retrieval fallback (`mem_add`/`mem_retrieve`).
- each dir: `measure.sh` (prints elapsed ms) + `verify.sh` (correctness oracle:
  `s == N*N == 400000000`; a cheat like "reduce N" fails it).

## Gate fixtures — do the REJECT and NO-WIN paths actually fire?

A run in which nothing happens to be rejected proves nothing about the reject path,
and you cannot make the model propose a broken theory on demand. So these force the
verdict from the **measure** side, which works whatever the model proposes:

- `gate_reject/` — `measure.sh` emits `src=pristine` only while `speed.c` is byte-identical
  to the `ref.c` copy beside it. Run with `--expect 'src=pristine'` and **every** applied
  theory must be REJECTED — even one that keeps `s` correct, because the token tracks the
  source, not the output.
- `gate_noise/` — `measure.sh` times `ref.c`, *not* the file the model is told to edit, so
  the metric is immune to every theory. Run with `--repeat 5` (a noise band needs more than
  one sample) and every theory must be flagged `[INSIDE NOISE]`, ending in
  *"No theory beat the noise band."*

`gen_fixtures.py` prints both ready-to-run commands. The shell logic alone can be checked
without a server: `cd _fix/gate_reject && ./measure.sh 2>&1 >/dev/null` → `src=pristine`,
then touch `speed.c` and it must say `src=EDITED`.

Prereqs: the 35B chat server on `:8181` (the embedder auto-spawns on `:8183` from
`~/.cache/huggingface/hub` and is freed after retrieval). Then:

```
cd _fix/big
basi-cli factory \
  --question "make the hot_loop computation in big.c faster; the metric is the milliseconds ./measure.sh prints (lower is better)" \
  --measure ./measure.sh --file big.c --expect 's=400000000' \
  --minimize --theories 3 --timeout 120
```

`--expect` folds correctness into the single measure run (the binary already prints
`s=...` on stderr while timing). Prefer this over `--verify ./verify.sh` (legacy
two-run path). Expected: retrieval lands windows on the loop; behavior-preserving
theories are measured; a theory that changes `s` is REJECTED.
