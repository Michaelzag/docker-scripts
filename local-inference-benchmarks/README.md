# Local inference benchmarks

This directory contains the shared benchmark harness, controlled experiment
overrides, and raw result artifacts for the local generation fleet. It is not
owned by any one model deployment.

The harness measures code-review prompts, sustained tool loops, same-prefix
reuse, concurrency, near-window retrieval, and reproducible quality tasks.
The existing 60-second marker is reported for usability comparisons; it is
**not** a new quality gate in the campaign challenge suite.

Model deployment files remain in their model-specific directories. Retained
fleet configuration and the human-readable benchmark record live in
`../../ubuntu-admin`.

## Layout

- `agent_workload_benchmark.py`: long-context, tool-loop, concurrency, and
  context-window workloads.
- `quality_regression_suite.py`: the historical regression gate, reported
  separately (24 fixed cases plus two default long-context cases = 26).
- `intelligence_challenge_suite.py`: exploratory 30-case campaign task set.
- `challenge_cases.py`: seeded fixtures, independent gold, strict JSON grading.
- `code_sandbox.py`: generated Python execution only in constrained Docker.
- `test_campaign_benchmarks.py`: focused standard-library invariant tests.
- `published_knowledge_benchmark.py`: fixed local 32-question published-gold slice.
- `test_published_knowledge_benchmark.py`: offline knowledge/parser/retention tests.
- `datasets/mmlu-pro-32-20261107/`: immutable prompts, separate gold and provenance.
- `experiments/`: reversible candidate-only runtime/configuration inputs.
- `results/`: raw JSON evidence and generated campaign summaries.
- `summarize_optimization_campaign.py`: regenerates the campaign index and
  evidence summary from `results/`.

By default the older entrypoints still omit full generated responses. Pass
`--artifacts-dir PATH` to retain every HTTP attempt, including `/tokenize`
calibration. The new challenge entrypoint requires that option.

## Campaign challenge protocol

`campaign-challenges-v1` has six cases in each of five categories:

- **reasoning:** assignment optimization, directed shortest paths, dependency
  scheduling, modular constraints, capacity optimization, and stock ledgers;
- **code:** actual Python synthesis/repair graded on hidden functional inputs
  (interval merging, stable deduplication, tenant boundaries, run lengths,
  revision selection, and topological ordering);
- **tools:** multi-turn depot allocation against mutable state, reserves,
  locks and capacities. The goal and final state determine success, not a
  prescribed tool sequence. Tool results expose real fixture state, not hints;
- **structured:** nested selection/aggregation, quoted instruction resistance,
  escaping, missing/null/false distinctions, and exact multiline constraints;
- **retrieval:** early/middle/late placement, numeric revision selection,
  tenant collisions, inactive records, and genuinely unanswerable queries
  among 220 seeded distractor records per case.

Prompts, gold and hidden code fixtures are written to a scoring manifest before
inference. Hidden fixtures are not included in model prompts; expected code
outputs are never sent into the candidate's execution container. Same seed and
case selection mean identical prompts/gold across candidates. Use multiple
`--seed` values and separate run directories for repetitions. `--nonce` labels
evidence without changing prompts. There is no LLM judge or model-specific
answer repair. Code must be source, not markdown.

Defaults are **8192 maximum output tokens, 600 seconds HTTP timeout**, temperature
zero, and at most ten assistant turns / twenty tool operations for tool tasks.
The twentieth operation is allowed and may be followed by the final answer;
an attempted twenty-first operation is rejected without changing depot state.
The token ceiling is not a required answer length. Errors, timeouts, malformed
responses, `length` finishes, wrong answers and blocked code execution count
as quality failures; no failures are dropped from category denominators.
Quality failure is a measured result (exit 0); invalid setup exits 2 before
inference. Reports checkpoint after each case and identify incomplete runs.
Before inference, exclusively created report/manifest placeholders and
`<output-stem>.run-owner.json` reserve every exact final destination. Cross-role
collisions (one report named as another run's manifest) are rejected too.
Partial reservation failures remove only that attempt's newly created files.
Ownership evidence remains after success, failure or interruption; use a fresh
output path rather than overwrite a run.
Compare only reports with `complete=true` and matching manifest fixture hashes.
The regression suite also rejects truncated completions even when visible text
looks correct.

Per-category reports contain passed/failed/total counts, request errors,
timeouts, length finishes, inference elapsed total/median/nearest-rank p95,
total case elapsed (including grading), reported token sums and number of
requests that supplied usage. Case evidence includes every call, exact finish
reason, sandbox failed check indexes, tool state transitions and artifact paths.
Missing usage is unknown, not evidence of zero compute. A long-context prompt's
character length is recorded; token usage is model-specific. Regression
calibration token counts and full `/tokenize` inputs/results are retained.

### Container prerequisite

Parent/operator prepares a known Python image and resolves its immutable digest:

```sh
docker pull python:3.12-slim
docker image inspect --format '{{index .RepoDigests 0}}' python:3.12-slim
```

Use the observed `python@sha256:...` (or full registry/repository digest) as
`SANDBOX_IMAGE`. On Armada use `sudo -n docker` for both preparation commands.
The harness accepts only the argv forms `docker` and `sudo -n docker`, parsed
without a shell, and never pulls an image automatically. A missing/mutable pin
or unavailable image fails before inference when code cases are selected.

Generated source runs only inside Docker with network `none`, read-only root,
UID/GID `65534`, all capabilities dropped, no-new-privileges, 256 MiB memory
and no extra swap, one CPU, 64 PIDs, bounded file descriptors/file sizes,
16 MiB noexec/nosuid/nodev `/tmp`, OOM score adjustment 800, and an eight-second
execution deadline (configurable only up to ten seconds). No host bind mounts,
Docker socket, privileges, devices, or GPU are exposed. Host collection bounds
stdout/stderr to 64 KiB and source to 128 KiB. The uniquely named container is
force-removed on completion/error/timeout: killing just the Docker CLI is not
treated as container cleanup. Failed cleanup is an explicit grading failure.
Docker is a containment boundary, not a claim of protection from kernel/runtime
vulnerabilities; use the operator-approved image and host policy.

### Run commands

Run from this directory on the endpoint's host. Set nonsecret variables
`MODEL`, `BASE_URL` (ending `/v1`), `SANDBOX_IMAGE`, `RUN_DIR`, and
`RUNTIME_MANIFEST` to the actual candidate and a new evidence path such as
`results/model-selection-20261107/<model>/<profile>/<run-id>`. Authentication
uses the existing key environment variable by **name**, never a key CLI value.
The manifest is an operator-authored JSON object containing explicit observed
runtime image/model hashes, configuration, host and GPU identities; no
environment dump or credential material. Without `--manifest`, identities
are honestly marked unknown, not inferred from labels.

```sh
python3 intelligence_challenge_suite.py \
  --base-url "$BASE_URL" --model "$MODEL" --api-key-env LOCAL_LLM_API_KEY \
  --output "$RUN_DIR/challenge.json" --artifacts-dir "$RUN_DIR/raw" \
  --sandbox-image "$SANDBOX_IMAGE" --docker-command docker \
  --manifest "$RUNTIME_MANIFEST" --label "$MODEL" --seed 20261107

python3 quality_regression_suite.py \
  --base-url "$BASE_URL" --model "$MODEL" --api-key-env LOCAL_LLM_API_KEY \
  --output "$RUN_DIR/regression.json" --artifacts-dir "$RUN_DIR/regression-raw" \
  --label "$MODEL"

python3 agent_workload_benchmark.py \
  --base-url "$BASE_URL" --model "$MODEL" --api-key-env LOCAL_LLM_API_KEY \
  --output "$RUN_DIR/agent.json" --artifacts-dir "$RUN_DIR/agent-raw" \
  --label "$MODEL"
```

On Armada change the challenge command to `--docker-command 'sudo -n docker'`.
For a small real-model smoke use
`--cases stock_ledger,stable_unique,depot_goal_1,nested_filter,retrieval_middle`;
for category screening use `--categories reasoning,structured,retrieval`.
Unknown filters and empty selections are errors. Use identical filters, seed
and budgets across candidates, and record any native profile differences.
`--extra-body-json` accepts explicit sampling keys `temperature`, `top_p`,
`top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repeat_penalty`
(llama.cpp), `repetition_penalty` (vLLM), and `seed` where the server supports
them, alongside `chat_template_kwargs` (including native thinking controls).
These values are sent and retained exactly; unsupported backend parameters
remain visible request failures rather than silently dropped settings. An
explicit nonzero temperature does not inherit the default greedy `top_k=1`.
The challenge CLI rejects all other top-level extra keys, including model,
messages, stream, token budgets, tools, tool_choice, stop sequences, sample
count and other request-shape overrides. For example:

```sh
--extra-body-json '{"temperature":1.0,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":true}}'
```

Older entrypoints retain their existing budget defaults; do not combine their
scores with challenge scores. Default challenge sampling remains temperature
zero unless the explicit native profile overrides it.

### Retention and metric limits

Each artifact uses an exclusive UUID-bearing filename, safe for concurrent
requests. It records complete JSON request payload/messages and wire-body text,
raw response text plus base64 bytes (including errors/malformed/partial bodies),
parsed response/reasoning/tool calls, HTTP status/error, usage/finish reasons,
UTC and America/New_York presentation timestamps, monotonic elapsed time,
request/messages/response hashes, model, case/turn, label, nonce and budgets.
No request headers, credential objects or environment values are serialized.
An echoed configured API key is redacted, including mixed Unicode-escaped JSON
representations. Valid wire JSON is reserialized only when redaction is needed;
UTF8 and base64 views derive from the same sanitized bytes. Hashes still describe
original bytes and `credential_redaction` identifies that exception. Complete
challenge reports/manifests are sanitized persistence copies; scoring retains
the original candidate response in memory. Incremental HTTP `read1` collection
retains bytes returned before a later socket timeout, including HTTP-error
bodies. Short declared Content-Length bodies are failures even when their
prefix is valid JSON; chunked framing uses HTTPResponse's unknown-length state
instead of incorrectly trusting a conflicting Content-Length header.
HTTPError response objects are explicitly closed. Malformed/partial wire JSON
uses the same escaped-key matcher, preserving unrelated bytes losslessly.
Keep payloads nonsecret; this is not a general arbitrary-secret scanner.

`nonstream_eof_seconds` is request-to-complete-body wall time, **not TTFT**.
`ttft_seconds` is null; streaming token timing is not implemented. The HTTP
timeout is the stdlib transport's timeout, not a new throughput/60s gate.
Prompt cache state is uncontrolled (no forced flush); record cold/warm profiles
externally and do not infer cache hits from repetition. Compare latency along
with quality counts, usage, configured budgets and runtime evidence. Six cases
per category make p95 effectively the maximum; repeated seeds are still not a
validated standardized intelligence benchmark or a general capability claim.
GPU/host-pressure capture remains an external operator wrapper.

### Focused checks (parent runs after all writers finish)

No third-party Python packages are needed:

```sh
python3 -m py_compile agent_workload_benchmark.py quality_regression_suite.py \
  challenge_cases.py code_sandbox.py intelligence_challenge_suite.py \
  test_campaign_benchmarks.py
python3 -m unittest -v test_campaign_benchmarks
```

These tests cover independent gold examples, strict JSON types/constraints,
seed/case invariants, failure-inclusive scoring, truncation, alternative legal
tool plans including the 20/21-operation boundary, atomic rejected operations,
real HTTP timeout prefix retention for success/error statuses, full/error/
partial/concurrent HTTP retention, premature EOF versus chunked framing,
escaped-key and complete-report redaction, same-path/cross-role output
ownership and rollback, calibration provenance, bounded deep sandbox JSON,
allowed Docker launch forms and sandbox security flags. They do not claim a real model or
Docker daemon was exercised; the operator must run the pinned-container and
actual-endpoint smoke before trusting campaign measurements.

## Small published-gold knowledge slice

`published-knowledge-mmlu-pro-32-v1` is **not full MMLU-Pro, an IQ test, or
evidence of unseen generalization**. It is one fixed answer-blind selection of
32 public test questions: four each from math, physics, chemistry, biology,
computer science, engineering, history and philosophy. There is no automatic
download, LLM judge, candidate voting, retry, answer-based selection, profile
grid or new client. It uses the same `ChatClient`, `request_body`, raw retention,
exclusive output reservation and sanitized atomic persistence as the frozen
utility harness. It does not require Docker or call tools.

### Predeclared final-answer grammar (v1)

This grammar is fixed before knowledge calls and is identical for every model.
Only final `message.content` with `finish_reason="stop"` and no tool calls can
be scored. `reasoning_content` is retained but **never parsed for an answer**.
`length` finishes fail even if their visible content appears correct.

Strip outer whitespace, then accept exactly one of:

1. One case-insensitive letter A–J, optionally wrapped in matching parentheses
   with whitespace inside, optionally followed by one period: `B`, `( b )`,
   `B.`, `(B).`.
2. One exact case-insensitive prefix `Answer:`, `The correct answer is`, or
   `Option`, followed by that same letter form. Whitespace is required after
   `is`/`Option`; after `Answer:` it is optional. For example `Answer: B`,
   `The correct answer is (B).`, `Option b`.
3. One JSON object with exactly one nonduplicate key `"answer"` and an exact
   single-letter string value A–J (case-insensitive), such as `{"answer":"b"}`.
   Normal JSON whitespace is permitted, not whitespace inside the value.

Exactly one full outer plain or `json` fence (case-insensitive language tag)
may wrap an otherwise accepted body: opening fence on its own line, body,
newline, closing fence. No nested fences or extra text. Thus a fenced JSON
answer is accepted, but `A or B`, `B because ...`, `I think B`, multiple letters,
duplicate JSON keys, extra keys, other fence languages and extracted matches
fail. A valid letter outside the question's actual option range fails.
Content is bounded to 65,536 characters; invalid/deep JSON (including
`RecursionError`) is an invalid answer, not an attempted repair. No answer is
extracted from reasoning or other prose. Reports retain raw answer, normalized
parsed choice, parser reason, finish reason and complete response artifacts.
The exact grammar is also persisted in the scoring manifest before inference.

### Dataset pin, provenance and limitations

The runner reads all three files as native bytes, verifies their fixed SHA256,
checks metadata-declared hashes, exact whitelist/schema, unique case/question
IDs, prompt/gold joins, four-per-category balance and answer/index consistency.
The authoritative pins are:

| File | SHA256 |
|---|---|
| `mmlu-pro-32-prompts.json` | `eb096d0411983db0a4c807ac02f03536fdde109816b14306e81364f6305cdebc` |
| `mmlu-pro-32-gold.json` | `d619f2b3c81cc9795b9ea0f8ee6b22179eb4384b7247315c1d384d7298f51053` |
| `mmlu-pro-32-metadata.json` | `6a931821cad7ece3fb9b942c4f98d30cad5ea53d1b5168fa344c1360cc5fc225` |

Source is official ungated `TIGER-Lab/MMLU-Pro`, MIT-tagged, test split.
Selection used `random.Random(20261107)`, four per category from the complete
12,032-row captured official public Rows view; the metadata records the full
answer-blind algorithm and selected IDs. The contemporaneous HF main SHA
`b189ec765aa7ed75c8acfea42df31fdae71f97be` was observed, but the Rows URL was
**unversioned** and does not prove per-page revision binding. Exact retained
subset bytes, not that SHA alone, pin this measurement.

The complete captured source is durably preserved at
`/mnt/models/inference-evals/vane-model-selection-20261107/source-v2/mmlu-pro-captured-rows.jsonl`,
SHA256 `808a00891b968caeec82d52bb2a4156f1a6ec33df918f7d678aa97187759e67a`.
This supersedes scratch-path notes inside the immutable metadata; the runner
does not read or claim to reverify that full capture.

Published test questions may occur in training data. Upstream dataset
construction used GPT-4 option augmentation plus expert review; this campaign
did not create labels using an LLM. Published gold is the scoring authority,
not candidate consensus. With 32 questions, one item is 3.125 percentage points;
with four per category, one item is 25 points. Uncertainty is wide and small
margins must not be presented as robust rankings or broad capability claims.
No private financial inputs or credentials belong in prompts or provenance.

Only the prompts file's question/options enter actual model messages. Its
case ID/question ID/category remain evidence metadata. Gold answers, indexes,
teacher rationale, source tags and raw gold rows never enter messages.
Gold is persisted separately in the scoring manifest; source identity is
retained in case evidence. The stable prompt asks for only a final option
letter while preserving the endpoint's explicitly configured native thinking.

### Run and compare

From this directory, use an existing endpoint and a fresh run directory for
each model/profile. `RUNTIME_MANIFEST` is the operator-authored runtime/model/
host/GPU evidence described above. No GPU/runtime changes are made by this
runner.

```sh
# First three fixed prompts for an actual endpoint smoke (not a benchmark score):
python3 published_knowledge_benchmark.py \
  --dataset-dir datasets/mmlu-pro-32-20261107 \
  --base-url "$BASE_URL" --model "$MODEL" --api-key-env LOCAL_LLM_API_KEY \
  --output "$RUN_DIR/knowledge-smoke.json" --artifacts-dir "$RUN_DIR/smoke-raw" \
  --manifest "$RUNTIME_MANIFEST" --label "$MODEL" --limit 3 \
  --max-tokens 8192 --measurement-timeout 600 \
  --extra-body-json '{"temperature":1.0,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":true}}'

# Full fixed 32-question slice; omit --limit and --cases:
python3 published_knowledge_benchmark.py \
  --dataset-dir datasets/mmlu-pro-32-20261107 \
  --base-url "$BASE_URL" --model "$MODEL" --api-key-env LOCAL_LLM_API_KEY \
  --output "$RUN_DIR/knowledge.json" --artifacts-dir "$RUN_DIR/raw" \
  --manifest "$RUNTIME_MANIFEST" --label "$MODEL" \
  --max-tokens 8192 --measurement-timeout 600 \
  --extra-body-json '{"temperature":1.0,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":true}}'
```

Use the candidate's qualified native profile (for example T1), not a new
cross-product. The JSON above is illustrative, not a universal profile
recommendation. Native keys are the exact same eleven-key allowlist documented
for the utility challenge, including repetition penalties, seed and native
`chat_template_kwargs`; overrides of model/messages/tools/stream/budgets are
rejected. Native settings and effective payloads are retained exactly.
Default temperature is zero only when no explicit profile changes it.
`--cases mmlu-pro-001,mmlu-pro-009,mmlu-pro-025` selects exact IDs in fixture
order; `--limit N` takes the first N selected prompts. Neither uses gold/results.

The initial checkpoint has zero attempted cases and `complete=false`; every
attempt updates it. `categories.all` and each category retain the **selected**
denominator, attempted/failed/unattempted counts and accuracy over selected
questions. Full runs have category denominators four each. Wrong, ambiguous,
abstaining, truncated and HTTP-failed responses are never dropped. A run is
complete only after all selected questions are attempted. Compare only
complete runs with identical selected IDs, dataset hashes, budgets and parser;
record qualified native-profile differences explicitly. A smoke/subset is not
a 32-question result. Latency comes from actual calls, not manufactured zeroes;
unattempted aggregates and TTFT are null. Usage sums include only reported
integer fields, accompanied by per-field reporting counts; missing usage does
not mean zero tokens. Quality failures exit 0 as measured evidence; invalid
setup exits 2 without inference.

Parent-only checks after all edits land:

```sh
python3 -m py_compile published_knowledge_benchmark.py test_published_knowledge_benchmark.py
python3 -m unittest -v test_campaign_benchmarks test_published_knowledge_benchmark
```

The new offline tests cover watermarked teacher-field exclusion from actual
serialized payloads, native profiles, raw/response/reasoning retention, the
finite parser, deep JSON, pinned native bytes/schema/joins, deterministic
subset selection, failure-inclusive checkpoints and shared namespace ownership.
They do not replace the operator's actual-model three-case smoke.

## Finite formatted-token capacity probe

`capacity_probe.py` measures **one already-loaded model/profile**, without
starting containers, changing GPUs, executing generated code or calling a judge.
It reuses the frozen retaining `ChatClient`, request builder and shared output
ownership/persistence helpers. It does not change the utility or knowledge
runner's request whitelist. Use the candidate's qualified **native profile**,
not a new profile grid; do not disable its native thinking. Its qualified
temperature is preserved, including when it is not T1.

The operator chooses `--parallel 4` or `6` and one explicit allocation mode:

* **fixed reserved:** `parallel` long inputs plus one extra short contender.
  With six slots, `--per-slot 131072 --pool 786432` describes six independent
  128K attention reservations (server configuration must actually match).
* **pooled mixed load:** two long + four short for six workers, or one long +
  three short for four workers, plus one extra short contender. The six-worker
  default is `--per-slot 131072 --pool 393216`. This is a shared 384K pool
  with a per-request ceiling, **not six reserved 128K windows**. An explicit
  `--long-workers 1` selects one long + five short (plus the extra short) for
  six workers, e.g. on an operator-selected 262144-token pool. Nothing
  auto-shrinks: the default two-long plan fails actual pool arithmetic there.
  Requested and actual request mix are retained.

Defaults are long input target 120000, short target 8000, fixed `max_tokens`
8192 and per-call measurement timeout 600 seconds. All `parallel+1` HTTP calls
are released at one client barrier, with every body/count prepared beforehand.
The extra short is a **queue contender**, not a guarantee about server admission
order; actual processing/deferred maxima must be observed, never inferred from
flags or client futures. Backend scheduling may defer a different request.

```sh
python3 capacity_probe.py \
  --base-url "$BASE_URL" --model "$MODEL" \
  --mode pooled --parallel 6 --per-slot 131072 --pool 393216 \
  --long-target 120000 --short-target 8000 --seed 20261107 \
  --measurement-timeout 600 \
  --extra-body-json '{"temperature":1.0,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":true}}' \
  --manifest "$RUNTIME_MANIFEST" --label "$MODEL" \
  --output "$RUN_DIR/capacity.json" --artifacts-dir "$RUN_DIR/capacity-raw"
```

The server is loaded separately by the parent operator. For fixed mode, change
only the explicitly selected mode/pool and corresponding server allocation;
for the smaller pooled variant explicitly add `--long-workers 1 --pool 262144`.
`--api-key-env` names an optional key variable (default `LOCAL_LLM_API_KEY`);
an unauthenticated private endpoint does not require a dummy credential.
The runtime manifest is mandatory, preserved with its native-byte SHA256,
and labelled operator-supplied rather than independently verified.
The example native settings are illustrative, not a universal model recipe.

### What is proved and what is unknown

Calibration calls `/v1/chat/completions/input_tokens` through the same client,
with the **same final body fields, messages, response format and native profile**
used for inference. `verbose=true`, `cache_prompt=false` and typed
`response_format={"type":"json_object"}` are intentional, entrypoint-specific
fields. Counts are actual formatted chat counts, never character/token guesses.
Every request must land within 512 tokens below its target; unsupported, invalid
or missing counts fail **before any inference**, with the failed raw calibration
attempt retained. All active output ceilings are included. Pool proof uses the
largest `parallel` measured inputs among *all* contenders, so it is safe for
every possible server admission order. It does not prove architectural fit,
GPU-only placement or successful hardware provisioning.

Each synthetic, nonfinancial archive has an early unique client nonce and three
distinct authoritative facts near 8%, 51% and 91% of equal-shape archive records.
The manifest retains expected values and approximate placement metadata outside
the requested answer; only the archive supplies the answer facts. Placement
fractions are not presented as token counts.

Separate per-request fields retain:

1. **actual input proof:** `timings.prompt_n + timings.cache_n` must exactly
   equal its calibrated count, `cache_n` must be zero and
   `__verbose.truncated` must explicitly be `false`. Missing evidence is
   `UNKNOWN`, never successful capacity proof.
2. **retrieval:** final `message.content` must itself parse as a JSON object
   with exactly the three correct key/value facts. No fence removal, answer
   extraction from reasoning, format repair or LLM judging.
3. **finish:** finish must be `stop` without tool calls. A length finish or
   unknown truncation never earns a usable retrieval pass.

The frozen client retains its **per-I/O socket timeout**, not hard absolute wall
cancellation. `wall_exceeds_configured_io_timeout` is diagnostic only: a correct,
completed request does **not** fail just because wall exceeds 600 seconds.
Actual HTTP timeout/error responses remain failures. The parent bounds the
overall job; this runner promises neither hard 600-second abort nor partial
retention on externally forced termination. There are no model HTTP retries.

Only during the finite group's live futures, a bounded sampler reads `/metrics`
at approximately 2 Hz; `/props` is retained before and after the group. It is
stopped and joined in `finally`, including worker exceptions. Metadata GETs use
the host's `curl` binary (required): a one-second **absolute transfer deadline**
and a 2 MiB body limit apply equally to `/props`, `/metrics` and HTTP error
bodies. An independent two-second subprocess timeout kills and reaps curl if
necessary, retaining its captured prefix. Thus slow-drip metadata cannot strand
sampler cleanup after model futures finish. Curl config is disabled and optional
authorization goes through stdin, not command-line arguments. This is metadata
only; all model calls still use the unchanged retaining `ChatClient` and its
per-I/O timeout semantics above. Raw status/body/stderr/exit/deadline evidence is
retained; timeout/error samples keep counters unknown even if partial text
contains plausible metrics.
`llamacpp:requests_processing` and `llamacpp:requests_deferred` maxima distinguish
observed requested concurrency/deferment, not observed, and unknown. A processing
maximum above the requested worker count is `EXCEEDS_REQUESTED`, not matching
capacity proof. Missing, ambiguous, nonfinite or unavailable metrics are **not
zero**. Short peaks can be
missed; saturation is not assumed. If the sampler's bounded window is exhausted,
the report explicitly says so. The separate parent-owned GPU sampler is not
started here.

Nonstream wall time ends at response EOF in the raw client artifact, **not
token TTFT**. Backend prefill/decode seconds are retained separately.
`residual_seconds_not_queue = wall - (prompt_ms + predicted_ms)/1000`
is un-clamped residual overhead, **not exact queue latency**; absent backend
timings produce null. Throughput uses the elapsed interval from barrier release
to the final client return, includes errors and is explicitly group end-to-end,
not a sum or average masquerading as backend aggregate throughput. Missing usage
is unknown and its reporting denominator is explicit.

The selected denominator is always all `parallel+1` requests, including HTTP
errors (including actual timeouts), wrong facts and unknown input evidence. The report
is checkpointed during calibration and after each completed future. Calibration
attempts are flushed to the adjacent `.calibration.jsonl`; every calibration
and inference HTTP attempt has the standard full request/wire response/EOF
artifact. The adjacent scoring manifest retains full final payloads, gold,
actual token budgets, native settings and runtime provenance. Props and bounded
metrics raw UTF-8/base64/status/error bodies are in the report. Keep all of these
together; a successful single 32K request or allocated KV cache is not evidence
for four/six independent long contexts.

Parent-only offline checks after all writers finish, followed by the operator's
real loaded-model run:

```sh
python3 -m py_compile capacity_probe.py test_capacity_probe.py
python3 -m unittest -v test_capacity_probe
```

Offline tests exercise actual budget boundaries, every supported load plan and
the explicit smaller-pool variant, formatted-count calibration with a synthetic
nonlinear tokenizer, unsupported-count startup failure/full raw retention,
strict input/truncation/cache evidence, wrong needles, finish/timeout/error
denominators, missing/deferred metrics, residual semantics, concurrent barrier
release and sampler cleanup. They do not establish real-model fit or speed.

The metadata deadline regressions start only a controlled loopback HTTP fixture:
it streams bytes every 50 ms for three seconds, exercising success and HTTP-error
bodies plus sampler cleanup while a metadata response remains in flight. They
require host `curl`, make no model calls, and are run by the parent with the
offline suite above.

## System One option-distribution adapter

`system_one_benchmark.py` benchmarks ZagDB's Jev-style `/v1/systemone` decision
contract: `--mode native` calls a native provider directly, while `direct` and
`thinking` adapt local chat/completion models. No mode implements Jev's retries
or measures chat quality as a substitute for distributions.
`datasets/system-one-synthetic-smoke/smoke.jsonl` contains four clearly labelled
**SYNTHETIC smoke items, NOT ZagDB gold**: noul, choice, score, and uncertainty.
They only establish harness behavior. Real owner-supplied fixtures can be passed
as one or more `--fixtures` JSONL paths; no LLM judge is involved.

Each row has `id`, `family`, `difficulty`, `state`, `question`, `gold`, `special`,
and optionally `rationale`. `state` is a native Structured string, object or array.
`question` has `id`, `type` (`noul`, `choice`, `score`), `text`, and an ordered
`options` array of `{id, text}`. `gold` has `option_ids` (an acceptable set) and
`defer_acceptable`; `special` maps `new`, `unknown`, `defer` to lists of option ids
(including multiple ids); an empty list means absent. A single id string and null
remain accepted at the JSON boundary. Internally markers are immutable sets:
NEW/unknown gold membership uses set intersection, and prediction uses membership.
Noul requires the ids `true` and `false`. Score option ids remain fixture identities;
the canonical benchmark legend uses their **original zero-based order**. A native
provider response retains the requested variant's legend order separately.
Gold and rationale never enter the model prompt. An explicitly gold option is
always correct, including when it belongs to the defer set. `defer_acceptable`
adds credit for any defer-mapped option even when it is not explicitly gold;
it never vetoes a gold answer. Defer confusion's expected label is gold/defer
intersection **or** `defer_acceptable`; NEW/unknown use gold-set intersection.

The adapter converts question text to native `instructions`; choice and noul
criteria are ordered id-to-text maps, score criteria are ordered descriptions.
The native request manifest uses `{model, state, questions: {question_id: Question}}`.
It checks the 32,768-byte state and 61,440-byte native request limits before HTTP.
It is not a production batching or 1 MiB response-cap test.

`system_one_prompt.txt` is the owned default prompt template. A custom
`--prompt-template PATH` must use exactly the `$state`, `$question`, `$options`
placeholders; Structured objects/arrays render as compact JSON. `--system-rule`
inserts a separate system message. The harness assigns A..Z prompt keys itself,
with original order plus seeded permutations (`--variants 2`, `--seed 0` defaults).
Each item requires 1..26 options and normally at least two distinct orders; an
excessive variant count is rejected. A singleton has two identical diagnostic
orders, margin/confidence 1 and entropy 0 (no meaningful order experiment).
The final writer fixture format, not the older explicit chat-variants proposal,
is supported.

### Probability contract

Direct mode calls `/apply-template` with `--direct-template-kwargs JSON`, appends
`--direct-prefill STRING` (empty by default), and tokenizes every key without
special tokens. When all keys are distinct single tokens, the primary path is
**one `/completion` forward pass for all keys**: a union key grammar,
`temperature=0`, `n_probs>=max(50,K+20)`, `cache_prompt=true`, and
**`post_sampling_probs=false`**. Scores come from that response's
`completion_probabilities[0].top_logprobs` plus the generated token's own
`.token` / `.logprob` (the two must agree when both report it).
This avoids assembling an apparently simultaneous distribution from numerically
different cached/concurrently batched forward passes.

A key can still fall outside top-k. **Only missing keys** get individual
forced-key fallback requests, using the generated token's own pre-constraint
logprob, never an invented zero. Such cases record `scoring_source="mixed"` and
`mixed_sources=true`; complete one-forward-pass cases record `"single"`.
Multi-token/shared-first-token keys retain exact forced-sequence scoring:
sum every generated token's own logprob, with cache reuse, and mark the source
mixed. All complete key scores are renormalized with log-sum-exp and mapped to
original option ids. Missing scores, malformed output and token coverage gaps
remain coverage failures. `sequence_logprobs` and `first_token_logprobs` retain
the unnormalized evidence.

`raw_key_mass` is the sum of raw key first-token probabilities, counting a shared
first token once. `leaked_mass=max(0,1-raw_key_mass)` describes only the first
step for multi-token keys, not sequence complement mass. Single-source sums
above one (beyond the existing 1e-5 tolerance) are errors. Mixed-source sums
may overshoot due to cross-request numeric variance; they are renormalized,
**not silently treated as single-source evidence**. Reports count single/mixed
sources and show raw mass count/mean/max, including measured sums on failed cases.
Sequence scores use the server tokenizer's canonical encoding, not a sum over
alternative tokenizations. Post-sampling probabilities are never used.

Thinking mode uses `/v1/chat/completions` with `--thinking-extra-body JSON`,
guarded by the existing native-profile allowlist: no request-shape/budget overrides.
`--max-tokens` applies only to thinking (8192 default). Only a stop-finished,
tool-free final content that strips to **exactly one case-sensitive key** is valid.
Explanations, alternatives, truncation and reasoning-only answers are failures.
Thinking reports accuracy/validity, **not** fabricated option distributions.

Direct-mode adapter answers are emitted as tagged `native_answer` plus `native_response`
with the exact model pin, ordered question-id answer map, and usage. Noul emits
`P(true)`; choice emits argmax option id, probabilities and normalized-pmax
confidence; score emits a `"0"`..`"K-1"` legend with exact original descriptions,
indexed probabilities, and **expected zero-based ordinal `sum(i*p_i)`**.
ZagDB permits the ordinal scale but does not prescribe expectation versus argmax.
Both conventions are recorded: score scalar metrics include expected ordinal,
argmax ordinal, nearest acceptable gold-level absolute error for each, and exact
argmax-level matches. Score scalars are operational in ZagDB gates, so do not
assume the two conventions are interchangeable.
Noul records its probability and correct/wrong/ambiguous counts at minima
0.6, 0.7 and 0.8: the middle band is `1-min < P(true) < min`; confidently
wrong answers are wrong, not ambiguous.
Usage sums the **real reported counts of completion requests producing draw 1**
(one primary pass plus missing-key fallbacks, or exact sequence requests), not
tokenizer/template calls or analysis-only repeat draws. Cached prompts can
reduce evaluated input counts. It reads `usage.prompt_tokens/completion_tokens`,
`tokens_evaluated/predicted`, or `timings.prompt_n/predicted_n`; missing usage leaves
`native_response=null`, never fake zero usage. `native_response_n` exposes coverage.

### Repeat-draw drift diagnostics

`--repeat-draws N` defaults to **1 (off)**. With N>1, each direct/native
item-variant repeats the same scoring prompt/request and order N times within
its worker at the configured concurrency (direct cache settings stay unchanged).
Template/tokenizer preparation happens once. A direct draw uses one forward pass
plus only necessary missing-key fallbacks; multi-token keys repeat their exact
sequence-scoring group. **Draw 1 alone determines accuracy, distribution,
native response and usage**, matching ZagDB's first accepted-response persistence
contract (C-3.5). Later draws are analysis, never voting or retries: an extra-draw
failure does not invalidate draw 1, and a later success never rescues failed draw 1.
Thinking mode remains accuracy-only and is not repeated.

Every draw retains its distribution, normalized pmax, argmax, source, raw mass
and error; HTTP artifacts include `/draw-N` in their case ids. Per-case `drift`
reports valid/failed draw counts, all unordered valid draw pairs' mean/max
absolute normalized-pmax difference, mean/max absolute per-option probability
difference, and argmax flip-pair rate. A floor crosses when
`min(pmax)<floor<=max(pmax)` for floors 0.5, 0.6, 0.7, 0.8 and 0.9.
Fewer than two valid draws yields null comparisons, not invented stability.
Aggregate drift weights means by comparison counts, retains maxima and flip
rates, and counts cases crossing each floor; sample counts remain explicit.
Latency/throughput and request-error diagnostics include all analysis work,
while primary accuracy/coverage denominators still count item-variants, not draws.

### Native endpoint mode

`--mode native` POSTs the native `DecisionRequest` to `/v1/systemone` on the supplied
base URL's origin (both an origin URL and an origin with `/v1` are accepted).
It uses the shared HTTP client's `Authorization: Bearer` header, loading the
credential from `--api-key-env`; headers and the credential are never persisted.
Echoes in report, manifest, and raw request/response artifacts use existing redaction.
No prompt/logprob adapter, template/tokenizer calls, or inference-profile fields
are sent to this endpoint. `both` still means only direct plus thinking.

Choice reads `probabilities` keyed by option id. Noul maps `noul` to
`{"true": noul, "false": 1-noul}`. Score reads probabilities keyed by the requested
legend indices `"0"`..`"K-1"` and maps them back to original fixture option ids.
The identical seeded variant generator orders choice criteria and score criteria,
so order-stability metrics compare the same semantic options across all modes.
The echoed model must equal `--model` exactly; mismatch or omission is an item
failure labelled `ModelDrift`. Missing/extra question or probability keys, wrong
answer types, booleans/non-numbers, nonfinite or out-of-range probabilities, and
mass not summing to one (absolute tolerance `1e-5`) are coverage failures.
They are never zero-filled or renormalized into apparently valid answers.

Native mode uses the same accuracy, special confusion, normalized-pmax,
entropy/margin, AUROC, Brier/log-loss, order-stability and latency summaries as
direct mode. HTTP errors, timeouts and >10-second requests remain counted.
`native_response` retains the provider's response, including real `usage` and
optional `request_id`; `native_answer` and scalar metrics express the canonical
original fixture order for cross-mode comparison. Native mode does not estimate
forced-token leaked mass, so those fields are null.

Every case records `state_bytes`, `state_estimated_tokens` (`state_bytes/4`), and
`state_token_warning_exceeded`. `--state-token-warning` defaults to 16384; only
strictly exceeding it sets the warning. Structured state uses compact UTF-8 JSON.
This is a **heuristic**, not server-tokenizer accounting: question/prompt overhead
is not included, so a false warning flag does not rule out silent truncation.
The existing 32768-byte state limit means the default warning cannot fire for
accepted fixtures under this heuristic; use a lower threshold for a conservative
diagnostic. Armada's runtime can silently truncate state to its bound; these
fields expose input size, not proof that the provider consumed every token.

```sh
python3 system_one_benchmark.py \
  --fixtures /path/to/owner-supplied-system-one.jsonl \
  --base-url http://10.0.3.130:5919 --model Cloudflare/clef-flash \
  --mode native --api-key-env SYSTEM_ONE_API_KEY \
  --state-token-warning 16384 --variants 2 --concurrency 2 \
  --measurement-timeout 600 --label armada-native-system-one \
  --output results/unique-native-system-one.json \
  --artifacts-dir results/unique-native-system-one-raw
```

The endpoint/model pin and credential variable name are operator inputs, not
autodiscovered runtime provenance. Offline tests make no calls to this endpoint.

### Running and interpreting

```sh
python3 system_one_benchmark.py \
  --fixtures datasets/system-one-synthetic-smoke/smoke.jsonl \
  --base-url http://127.0.0.1:8080/v1 --model exact-loaded-model-pin \
  --mode both --direct-template-kwargs '{"enable_thinking":false}' \
  --thinking-extra-body '{"chat_template_kwargs":{"enable_thinking":true}}' \
  --concurrency 2 --measurement-timeout 600 --max-tokens 8192 \
  --label SYNTHETIC-smoke-NOT-ZagDB-gold \
  --output results/unique-system-one-smoke.json \
  --artifacts-dir results/unique-system-one-smoke-raw
```

Supply the credential using `LOCAL_LLM_API_KEY` (or the name in `--api-key-env`);
do not put its value on the command line. Muse can use
`--direct-prefill '<|message|>'`; Qwen/Nemotron direct profiles can disable thinking.
Native settings are operator-supplied, not silently guessed. Optional `--manifest`
is runtime/model provenance input, not a replacement for the generated scoring manifest.
The shared output reservation rejects namespace collisions; use a fresh output.
All requests/responses, including template/tokenizer/error attempts, use the shared
full-wire retention/redaction. Keep report, adjacent manifest, run ownership file,
fixture bytes and raw artifacts together.

Each mode reports per family and overall. **`n` counts all selected item-variants**,
not just successful HTTP responses; `n_items` separately counts fixture items.
Failures and unattempted checkpoint rows remain in accuracy, validity, coverage
and order-agreement denominators. Distribution summaries have explicit
`distribution_n`; unavailable confidence is not imputed. Statistics include
entropy (nats), top-two margin, `(K*pmax-1)/(K-1)` normalized-pmax, special-option
TP/FP/FN/TN counts, correct/wrong entropy and margin means, margin-correctness
AUROC (null unless both classes exist), binary argmax-confidence Brier/log-loss
(against correctness, **not an invented gold distribution**; log floor 1e-15),
and mean/max first-token leaked mass.
Order stability uses all unordered variant pairs, original-id argmax agreement
(failed pairs count against agreement), and mean Jensen-Shannon **distance**
(`sqrt(JS divergence)`, base 2) on complete-distribution pairs, with pair counts.
Nonstream per-item-variant latency includes template/scoring calls; p50 and
nearest-rank p95, per-mode wall time and throughput include errors and checkpoint
overhead. `items_per_second` divides completed item-variants by variant count;
`item_variants_per_second` reports attempts directly at the configured concurrency.
Requests and item-variants exceeding the production provider's 10-second
timeout are counted, **not dropped**. The measurement timeout is the shared
client's per-I/O timeout, not an absolute request deadline; this runner has no retries.

Offline verification (only controlled loopback fake servers; no live models):

```sh
python3 -m unittest -v test_campaign_benchmarks test_published_knowledge_benchmark test_capacity_probe test_system_one_benchmark
```

## ZagDB generative benchmark

`generative_benchmark.py` evaluates the supplied ontology naming and hub summary
fixtures without a model judge. It verifies each rendered message against the
fixture hashes, makes one request per row with temperature and seed fixed at zero,
and retains request/response evidence. The hub summary prompt is an **eval-only
proposal**, not a production prompt. Invalid shapes and length finishes fail
without repair or retry. The hs-13 oversize trial is context-dependent:
an accepted full-input request that fits the endpoint is not an oversize test.
The refusal check accepts only HTTP 4xx with error type
`context_length_exceeded` or llama.cpp's `exceed_context_size_error`, and no
generated output. `diagnostics.zagdb_detector_compatible` is true only for
`context_length_exceeded`; it does **not** gate row success. Each variant reports
`zagdb_detector_compatible_count` and `zagdb_detector_checked_count` separately,
so a correct server refusal is not penalized for ZagDB's detector limitation.

Before generation the runner makes one GET to the endpoint origin's `/props`,
using `default_generation_settings.n_ctx`, then top-level `n_ctx`, for per-slot
context. If neither is available, `--context-tokens N` is required; otherwise the
run fails before generation. The report's `context` records `n_ctx` and its source.
For an accepted hs-13 response, integer `usage.prompt_tokens < n_ctx` with no
input-truncation signal yields `outcome: not_oversize_for_config` and
`excluded_from_scoring: true` (`passed: null`). Its ordinary content checks remain
under `diagnostics`, not in pass/fail totals. Prompt counts at or above context,
missing counts, or explicit input-truncation flags fail. Variant summaries expose
`scored_rows` and `not_oversize_for_config_count`; `pass_rate` uses only scored rows.
The output budget is not capped at 8192: `--max-tokens 65536` is supported end to
end, independently of the context comparison.

```sh
python3 generative_benchmark.py \
  --fixtures /path/to/generative --templates /path/to/generative/templates.json \
  --base-url http://127.0.0.1:8080/v1 --model MODEL \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --max-tokens 8192 --concurrency 1 --measurement-timeout 600 \
  --output /path/to/report.json --artifacts-dir /path/to/artifacts --label run
```

The API key is read from `LOCAL_LLM_API_KEY` by default; `--api-key-env`
selects another variable. Reports retain per-row check outcomes and family
statistics, including errors, latency, and terminal length/shape failures.
The shared client retains sanitized wire evidence in the artifacts directory.
Run the offline fake-server tests with `python3 -m unittest -v
test_generative_benchmark`.

The runner also evaluates all 14 hub rows using **candidate real prompt
(unmerged)** from the P6 integration work. Its default template path is
`<fixtures>/real_hub_prompt/hub_summary@1.tera`; override the location with
`--candidate-hub-template`. Its exact byte SHA256 is pinned to
`8ca5e76dd7fb1960cedc0c969e5f8955ec334f05bb79402d4679be2d386ba372`.
The template uses only bare Tera variable interpolation, without filters,
conditionals, loops, or autoescaping. Unsupported constructs are rejected.

Candidate variables are mapped from the supplied community id and level, the
complete member list serialized as canonical JSON, and retained member
assessments collected into a JSON object keyed by revision id. Inactive-unknown
members supply no assessment body. No absent values are invented: incomplete
rows fail with `reason: missing_variable` and a `missing_variables` list before
any HTTP call. The proposal and candidate use the same row validator but have
separate labelled report groups. Thus a complete run has 43 attempts (15 naming,
14 proposed hub, 14 candidate hub), less any missing-variable failures.

### Native JSON extraction and offline rescoring

Output extraction follows ZagDB's `json_candidate`
(`crates/zagdb-bridge/src/judges.rs:1528-1557`): take the first fenced body if a
complete fence exists, otherwise slice from the first `[`, `{`, or `"` to its
last corresponding closer. JSON parsing and the fixture's exact shape rules
then decide validity; extra keys still fail. This intentionally preserves the
native failure when a prose quote precedes an unfenced JSON object. It does not
search for a later valid object or repair a malformed candidate.

```sh
python3 generative_benchmark.py --rescore /path/to/finished-run \
  --fixtures /path/to/generative --templates /path/to/generative/templates.json
```

Rescoring reads `finished-run/report.json`, its retained raw artifacts, and the
same immutable fixture rules. It requires no API key, model, or endpoint and
makes zero HTTP calls. It writes `report.rescored.json` beside the original,
preserving both the old report and artifacts. Existing output is never
overwritten; use `--output /path/to/finished-run/report.rescored-again.json` for
another scoring revision. Relative artifact paths resolve from the run directory;
relocated captures can use `finished-run/raw/<original-artifact-filename>`.
The retained per-slot context is reused; older reports without it require
`--context-tokens N`. Latency and wall-time measurements remain the original
inference measurements, not offline parsing time.

