#!/usr/bin/env python3
"""
Chat completion throughput benchmark for litellm-vs-direct comparison.

Hits an OpenAI-compatible /v1/chat/completions endpoint at varying
output lengths and concurrency. Reports tokens/sec, time-to-first-token
proxies (wall - decode-est), and p50/p99 latency.

Usage:
    python3 chat_benchmark.py --base-url http://localhost:7910/v1 \
        --api-key $KEY --model mbai/google/gemma-4-26b-a4b-it
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REQUEST_TIMEOUT_SECONDS = 60.0

PROMPTS_SHORT = [
    "Write a 50-word story about a robot.",
    "Explain quantum entanglement in 3 sentences.",
    "List 5 reasons to learn Rust.",
    "What is the capital of Mongolia and why is it interesting?",
    "Describe sunset on Mars in one paragraph.",
    "Why do cats purr? Brief answer.",
    "Compare HTTP/2 and HTTP/3 in 4 sentences.",
    "Translate 'hello world' into 5 different languages.",
]

PROMPTS_MEDIUM = [
    "Write a 200-word story about a ship lost at sea.",
    "Explain the differences between RAFT and Paxos consensus algorithms in 200 words.",
    "Describe the architecture of a typical microservices system, including API gateway, service mesh, observability, and data persistence patterns. About 200 words.",
    "Write a 200-word essay on why functional programming has gained popularity in recent years.",
    "Explain how FIDO2/WebAuthn works for passwordless authentication, in about 200 words covering the registration and authentication ceremonies.",
    "Describe what happens when you type 'curl example.com' and press enter, in about 200 words covering DNS, TCP, TLS, and HTTP.",
]

PROMPTS_LONG = [
    "Write a detailed 800-word essay on distributed systems consensus algorithms. Cover Paxos, RAFT, and Byzantine fault tolerance. Discuss tradeoffs and real-world applications.",
    "Write a 800-word technical analysis of how modern GPUs handle parallel computation. Cover SIMT execution, memory hierarchies, occupancy, warp divergence, and tensor cores.",
    "Write a comprehensive 800-word guide to Rust's borrow checker, covering the rules, common patterns that work, common patterns that don't, and how lifetimes interact with closures and async.",
]


def post_chat(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
    except TimeoutError as exc:
        raise RuntimeError(
            f"request exceeded the {REQUEST_TIMEOUT_SECONDS:g}s usability cutoff"
        ) from exc
    elapsed = time.time() - t0
    return {
        "elapsed": elapsed,
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "total_tokens": data["usage"]["total_tokens"],
    }


def run_sequential(base_url, api_key, model, prompts, max_tokens, label):
    print(f"\n  Sequential: {label}, n={len(prompts)}, max_tokens={max_tokens}")
    t0 = time.time()
    completion = 0
    prompt = 0
    latencies = []
    for p in prompts:
        r = post_chat(base_url, api_key, model, p, max_tokens)
        latencies.append(r["elapsed"])
        completion += r["completion_tokens"]
        prompt += r["prompt_tokens"]
    wall = time.time() - t0
    print(f"    wall: {wall:.2f}s | prompt_tok: {prompt} | completion_tok: {completion}")
    print(f"    completion_tok/sec: {completion / wall:.1f}")
    print(f"    per-call latency: p50={statistics.median(latencies)*1000:.0f}ms  max={max(latencies)*1000:.0f}ms")
    return {"wall": wall, "completion_tokens": completion, "prompt_tokens": prompt,
            "completion_per_sec": completion / wall, "p50_ms": statistics.median(latencies) * 1000}


def run_concurrent(base_url, api_key, model, prompts, max_tokens, n_workers, label):
    print(f"\n  Concurrent: {label}, n_workers={n_workers}, n={len(prompts)}, max_tokens={max_tokens}")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(post_chat, base_url, api_key, model, p, max_tokens) for p in prompts]
        for f in as_completed(futures):
            results.append(f.result())
    wall = time.time() - t0
    completion = sum(r["completion_tokens"] for r in results)
    prompt = sum(r["prompt_tokens"] for r in results)
    latencies = [r["elapsed"] for r in results]
    print(f"    wall: {wall:.2f}s | prompt_tok: {prompt} | completion_tok: {completion}")
    print(f"    completion_tok/sec (aggregate): {completion / wall:.1f}")
    print(f"    per-call latency: p50={statistics.median(latencies)*1000:.0f}ms  max={max(latencies)*1000:.0f}ms")
    return {"wall": wall, "completion_tokens": completion, "prompt_tokens": prompt,
            "completion_per_sec": completion / wall, "p50_ms": statistics.median(latencies) * 1000}


def main():
    global REQUEST_TIMEOUT_SECONDS

    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mult", type=int, default=4, help="Multiply prompt list to grow workload")
    p.add_argument(
        "--request-timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help="Fail the run when any response exceeds this many seconds (default: 60)",
    )
    args = p.parse_args()
    REQUEST_TIMEOUT_SECONDS = args.request_timeout

    short = (PROMPTS_SHORT * args.mult)
    medium = (PROMPTS_MEDIUM * args.mult)
    long_ = (PROMPTS_LONG * args.mult)

    # Warmup
    print(f"=== WARMUP ===")
    post_chat(args.base_url, args.api_key, args.model, "Hi", 5)

    print(f"\n=== Model: {args.model} ===")

    print(f"\n--- SHORT (small prompt, 64 tok output) ---")
    run_sequential(args.base_url, args.api_key, args.model, short, 64, "single-stream")
    run_concurrent(args.base_url, args.api_key, args.model, short, 64, 4, "4-way")

    print(f"\n--- MEDIUM (medium prompt, 256 tok output) ---")
    run_sequential(args.base_url, args.api_key, args.model, medium, 256, "single-stream")
    run_concurrent(args.base_url, args.api_key, args.model, medium, 256, 4, "4-way")

    print(f"\n--- LONG (long prompt, 512 tok output) ---")
    run_sequential(args.base_url, args.api_key, args.model, long_, 512, "single-stream")
    run_concurrent(args.base_url, args.api_key, args.model, long_, 512, 4, "4-way")


if __name__ == "__main__":
    main()
