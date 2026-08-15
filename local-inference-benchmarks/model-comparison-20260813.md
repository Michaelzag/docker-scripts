# Gemma 4 slot model comparison — 2026-08-13

## Decision

Keep Gemma 4 in the RTX 5090 production slot.

- Gemma is the fastest option on both NVIDIA hosts and retains strong objective quality.
- Muse Glimmer is the quality winner by 1–2 cases, but is much slower on both GPUs and is operationally unusable on the M4 Pro at full 131K-per-request sizing.
- Nemotron 3.5 Lightning is usable and reasonably fast, but scored materially worse on the objective suite on all three hosts.

All configurations passed the OpenAI-compatible tool-call smoke test. Every context configuration below provides at least 131,072 tokens per request. Parallelism was reduced before context size.

## Usability rule

The benchmark now has a hard 60-second per-response cutoff. A configuration is marked DNF if any individual response exceeds that limit. Muse on the Mac was stopped after its completed 512-token sequential tier reached 84.1-second p50 and 91.3-second maximum latency.

## RTX 5090 32 GB

| Model | Runtime and context | Quality | 64 tok seq / aggregate | 256 tok seq / aggregate | 512 tok seq / aggregate |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B-A4B | vLLM 0.25.1-cu129, NVFP4, BF16 MTP, 196K request context | 23/26 | 173.2 / 513.9 tok/s | 170.4 / 560.8 tok/s | 174.1 / 568.9 tok/s |
| Muse Glimmer 30B | llama.cpp b10398, Dynamic Q4_K_XL, DFlash, 2 × 131K | **25/26** | 78.9 / 88.3 tok/s | 61.8 / 88.9 tok/s | 61.2 / 90.3 tok/s |
| Nemotron 3.5 Lightning 30B-A3B | vLLM 0.27.1, Marlin, FP8 KV, MTP, 131K | 20/26 | 147.0 / 229.3 tok/s | 136.6 / 486.8 tok/s | 135.0 / 500.5 tok/s |

Nemotron's official vLLM recipe reported that this build did not have native FP4 support for the RTX 5090 and used weight-only Marlin. This may leave performance available, but it does not explain or repair the lower objective score.

## Armada RTX 3090 24 GB

| Model | Runtime and context | Quality | 64 tok seq / aggregate | 256 tok seq / aggregate | 512 tok seq / aggregate |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B-A4B | llama.cpp b10398, Q4_0, MTP, 2 × 131K | 24/26 | **186.7 / 249.1 tok/s** | **196.3 / 263.5 tok/s** | **198.1 / 266.3 tok/s** |
| Muse Glimmer 30B | llama.cpp b10398, 17 GB Q4_K_M, DFlash, 1 × 131K | **25/26** | 71.2 / 72.8 tok/s | 53.5 / 54.3 tok/s | 51.9 / 52.6 tok/s |
| Nemotron 3.5 Lightning 30B-A3B | llama.cpp b10398, Q4_K_M, 48 GPU layers plus CPU spill, 1 × 131K | 20/26 | 94.3 / 97.5 tok/s | 109.9 / 110.0 tok/s | 115.0 / 114.8 tok/s |

The 25.4 GB Nemotron Q4_K_M cannot fit wholly in 24 GB VRAM together with a 131K KV cache. Its measured configuration used 22.7 GB VRAM and partial CPU offload. Gemma used about 21.5 GB and Muse about 21.2 GB. GPU1 was returned to idle after the run; GPU0 services were not touched.

## Mac mini M4 Pro 64 GB

| Model | Runtime and context | Quality | 64 tok seq / aggregate | 256 tok seq / aggregate | 512 tok seq / aggregate |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B-A4B | MLX OptiQ 4-bit target plus BF16 assistant, 1 × 131K | 24/26 | 43.8 / 44.0 tok/s | 41.7 / 41.5 tok/s | 40.2 / 40.3 tok/s |
| Muse Glimmer 30B | llama.cpp b10373 Metal, Dynamic KQuant plus DFlash, 4 × 131K | **25/26** | 8.7 / 10.1 tok/s | 6.4 / 10.4 tok/s | 6.5 / **DNF** |
| Nemotron 3.5 Lightning 30B-A3B | llama.cpp b10373 Metal, Q4_K_M, 4 × 131K | 19/26 | **51.5 / 71.4 tok/s** | **50.9 / 70.1 tok/s** | **49.1 / 70.9 tok/s** |

Muse's completed 512-token sequential tier took 941.6 seconds for 12 requests, with 84.1-second p50 and 91.3-second maximum latency. Its DFlash acceptance was commonly only 12–15%. It fit in memory and remained stable, but failed the inference-usability bar.

Nemotron stayed under the timer. Its four-way 512-token tier had 29.2-second p50 and maximum latency, but its objective quality score fell to 19/26.

## Objective quality suite

The identical 26-case suite contains eight reasoning cases, six code-review cases, six judge cases, four instruction-following cases, and two long-context cases. Thinking was explicitly disabled for fair comparison because Nemotron defaults to thinking on.

| Host | Gemma | Muse | Nemotron |
|---|---:|---:|---:|
| RTX 5090 | 23/26 | **25/26** | 20/26 |
| RTX 3090 | 24/26 | **25/26** | 20/26 |
| M4 Pro | 24/26 | **25/26** | 19/26 |

Muse failed the same scheduling problem on all three hosts. Nemotron consistently passed code review, judging, and long-context retrieval, but performed poorly on arithmetic/graph reasoning and one exact JSON transformation. Gemma was one or two cases behind Muse while being much faster.

## Runtime research

- Muse Glimmer: official `meta-models/Muse-Glimmer-30B-GGUF` Dynamic KQuant, DFlash, multimodal projector, and recent llama.cpp path.
- Nemotron: official NVIDIA vLLM 0.27.1 recipe on the RTX 5090; official ggml-org Q4_K_M with llama.cpp on Ampere and Metal.
- Gemma: existing production NVFP4 vLLM stack on the RTX 5090; official ggml-org Q4_0/MTP llama.cpp artifacts on Ampere; the existing OptiQ/MLX target and BF16 drafter on Apple Silicon.

## Final state

- RTX 5090: production `gemma4-26b-a4b` restored, healthy, zero restarts, tool call passed, and live client traffic observed.
- Armada RTX 3090 GPU1: benchmark containers removed; GPU returned to idle.
- Mac mini: benchmark server stopped.

