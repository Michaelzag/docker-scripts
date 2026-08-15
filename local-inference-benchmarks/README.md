# Local inference benchmarks

This directory contains the shared benchmark harness, controlled experiment
overrides, and raw result artifacts for the local generation fleet. It is not
owned by any one model deployment.

The harness measures realistic code-review prompts, sustained tool loops,
same-prefix reuse, concurrency where it is part of the service contract,
near-window context retrieval, and a source-controlled quality suite. A
request taking more than 60 seconds is recorded as a usability failure unless
the corresponding model record explicitly documents an accepted cold-ingress
exception.

Model deployment files remain in their model-specific directories. Retained
fleet configuration and the human-readable benchmark record live in
`../../ubuntu-admin`.

## Layout

- `agent_workload_benchmark.py`: long-context, tool-loop, concurrency, and
  context-window workloads.
- `quality_regression_suite.py`: deterministic 26-case quality gate.
- `experiments/`: reversible candidate-only runtime/configuration inputs.
- `results/`: raw JSON evidence and generated campaign summaries.
- `summarize_optimization_campaign.py`: regenerates the campaign index and
  evidence summary from `results/`.

Generated responses and API keys are not retained. Result files record prompt
hashes, token counts, timings, cache behavior, tool validity, finish reasons,
and bounded quality outputs.
