# factory test fixtures

Throwaway fixtures for exercising `basi-cli factory` end-to-end (see the
`project-factory-command` memory for the full design + open items).

```
python3 gen_fixtures.py _fix          # writes _fix/small and _fix/big
```

- `small/speed.c` — ~15-line O(N²) busy loop, <48KB → `--file` is injected verbatim.
- `big/big.c` — same hot loop among 2600 filler fns, ~170KB → `--file` exceeds the
  48KB inject cap and takes the RAG retrieval fallback (`mem_add`/`mem_retrieve`).
- each dir: `measure.sh` (prints elapsed ms) + `verify.sh` (correctness oracle:
  `s == N*N == 400000000`; a cheat like "reduce N" fails it).

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
