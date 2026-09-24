# Lessons

## Code map
- Slash commands are registered in two places: the autocomplete table in src/slashmenu.c (rows like { "/cost", "show session token usage", false }) and the dispatch chain plus the /help printf listing in src/main.c (the /cost strcmp branch sits alongside the /help strcmp).  <!-- eval -->
- Plan-tooling phase state is a global: the PlanPhase enum (PHASE_NONE, PHASE_DRAFTING, ...) lives in src/globals.h and the current value is the plan_phase variable plus current_plan_slug in src/main.c; phase transitions and the plan file logic are in src/plan.c. plan_write only succeeds while plan_phase is drafting/premortem, reached via the /plan command.  <!-- eval -->

## Habits
- When a planning/phase-gated tool refuses ('only callable during drafting or premortem phase'), enter the required phase with the documented entry command (e.g. /plan <slug>) instead of hand-writing the artifact and asserting it is equivalent — and never claim a file was written without a confirming tool call in the transcript.  <!-- eval -->
- Before editing an existing file, open it with the read tool (not just sed/cat via bash); otherwise the edit is rejected as unread and you burn a loop.  <!-- eval -->
- Don't claim verification needs a model load and skip it — CLI slash commands can be exercised by piping stdin into the binary (HOME=$(mktemp -d) BASI_TELEMETRY=0 ./basi-cli --no-mcp) and grepping the output; run that instead of asserting it's impossible.  <!-- eval -->
- Verify CLI behaviour by actually driving the binary end-to-end (e.g. pipe stdin into ./basi-cli --no-mcp) instead of assuming it needs a model download and settling for greps of source; the automated check ran exactly that way and passed, so the shortcut was unnecessary.  <!-- eval -->
- Recon has a budget: once the edit sites are located (dispatch branch, help listing, autocomplete table), stop reading and make the edits in the same turn — do not keep grepping/reading until the tool budget is exhausted, leaving the change unapplied.  <!-- eval -->
- Do not stop at a plan: run the freshly built binary with piped input (e.g. `printf '/version\n/help\n' | ./basi-cli`) and actually read the captured output to confirm the new line prints, rather than reporting it as unobserved/expected.  <!-- eval -->
- When the user says "plan it before implementing", treat the plan as a step, not the deliverable: after planning, actually make the edits, build, and run the change end-to-end before finishing — never end the turn by asking permission to implement.  <!-- eval -->
- Do not assert a runtime check is impossible without trying it: run the built binary with the simplest flags to smoke-test the feature end-to-end, rather than settling for 'source parity' with a similar command.  <!-- eval -->
- Create scratch directories with mkdir -p on a fresh unique path (or mktemp -d) instead of a fixed name like /tmp/vtest; a leftover entry with that name makes mkdir fail and the follow-up cd fail, wasting a loop.  <!-- eval -->
- When a tool call is refused with the exact steps needed to unlock it, do not repeat the same call verbatim — either perform the unlock step or drop that tool and say so. Here plan_write was rejected twice with the same message, wasting two full loops on an identical, already-answered request.  <!-- eval -->
