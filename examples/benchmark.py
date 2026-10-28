"""Benchmark: original AirLLM vs. VyanLLM.

Each configuration runs in its own process so memory numbers don't bleed between runs.

    git clone https://github.com/lyogavin/airllm airllm-orig     # the baseline
    python examples/benchmark.py --orig airllm-orig/air_llm --new . \
        --model Qwen/Qwen2.5-7B-Instruct --tokens 8
    # add --compression 4bit to benchmark the compressed path (needs bitsandbytes)
    # on macOS add --backend torch (the PyTorch path, on Apple's GPU)

Reports: seconds per generated token (decode only, prefill timed separately), peak VRAM,
peak host RSS, disk read per token (VyanLLM only) and whether the generated tokens match
the original exactly.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

PROMPT = "Explain in one paragraph why the sky is blue."


def worker(args):
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    import torch
    # The baseline is an AirLLM checkout (package "airllm"); the candidate is VyanLLM.
    pkg = "vyanllm" if os.path.isdir(os.path.join(repo, "vyanllm")) else "airllm"

    kwargs = json.loads(args.kwargs)
    if "gpu_cache_gb" in kwargs and not torch.cuda.is_available():
        kwargs.pop("gpu_cache_gb")  # VRAM tier is CUDA-only; on a Mac cache_gb covers it
    if args.compression:
        kwargs["compression"] = args.compression
    if args.backend == "torch":
        # The PyTorch streaming path (what the patch changes). On macOS AutoModel would route to
        # the separate MLX backend instead, so construct the base class directly; it picks
        # CUDA, then Apple MPS, then CPU.
        import importlib
        if pkg == "airllm":
            # Original AirLLM on macOS switches its shard format globally to MLX, which its own
            # PyTorch path can't read. Force safetensors (VyanLLM does this by itself).
            mp = importlib.import_module("airllm.persist.model_persister")
            st = importlib.import_module("airllm.persist.safetensor_model_persister")
            mp.model_persister = st.SafetensorModelPersister()
            Base = importlib.import_module("airllm.airllm_base").AirLLMBaseModel
        else:
            Base = importlib.import_module("vyanllm.vyanllm_base").VyanLLMBaseModel
        model = Base(args.model, **kwargs)
    else:
        AutoModel = __import__(pkg).AutoModel
        model = AutoModel.from_pretrained(args.model, **kwargs)
    ids = model.tokenizer([PROMPT], return_tensors="pt",
                          return_attention_mask=False)["input_ids"].to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Prefill + first token, then the remaining tokens: decode speed is what streaming governs.
    t0 = time.time()
    model.generate(ids, max_new_tokens=1, do_sample=False)
    prefill = time.time() - t0

    stats = getattr(model, "cache_stats", None)  # only the patched version tracks disk reads
    if stats is not None:
        for k in stats:
            stats[k] = 0
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=args.tokens, do_sample=False,
                         return_dict_in_generate=True)
    total = time.time() - t0

    result = {
        "prefill_s": prefill,
        "s_per_token": total / args.tokens,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0,
        # ru_maxrss is bytes on macOS, kilobytes on Linux
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                       / (2**20 if sys.platform == "darwin" else 1024),
        "tokens": out.sequences[0].tolist(),
        "disk_mb_per_token": stats["disk_bytes"] / args.tokens / 2**20 if stats is not None else None,
    }
    print("RESULT " + json.dumps(result))


def run(label, repo, args, kwargs):
    cmd = [sys.executable, __file__, "--worker", "--repo", repo, "--model", args.model,
           "--tokens", str(args.tokens), "--kwargs", json.dumps(kwargs), "--backend", args.backend]
    if args.compression:
        cmd += ["--compression", args.compression]
    print(f"\n>>> {label}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        print(proc.stdout[-2000:], proc.stderr[-4000:])
        return None
    r = json.loads(line[7:])
    disk = "" if r.get("disk_mb_per_token") is None else f" | disk {r['disk_mb_per_token']:.0f} MB/token"
    print(f"    {r['s_per_token']:.2f} s/token | prefill {r['prefill_s']:.1f}s | "
          f"VRAM {r['peak_vram_mb']:.0f} MB | RSS {r['peak_rss_mb']:.0f} MB{disk}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--repo")
    ap.add_argument("--orig", help="path to the original AirLLM checkout's air_llm/ dir")
    ap.add_argument("--new", help="path to the VyanLLM project dir (the one containing vyanllm/)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--compression", choices=["4bit", "8bit"])
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--cache-gb", default="auto",
                    help="budget for the layer-cache row: GB or 'auto' (default)")
    ap.add_argument("--backend", choices=["auto", "torch"], default="auto",
                    help="torch = force the PyTorch streaming path (use this on macOS)")
    args = ap.parse_args()

    if args.worker:
        return worker(args)

    extra = json.loads(args.kwargs)
    runs = [("original", args.orig, {}),
            ("VyanLLM (all on)", args.new, {}),
            ("VyanLLM, async_copy off", args.new, {"async_copy": False}),
            ("VyanLLM, per-layer cleanup (old behaviour)", args.new, {"memory_cleanup": "layer"}),
            ("VyanLLM + layer cache", args.new, {"cache_gb": args.cache_gb, "gpu_cache_gb": args.cache_gb})]
    if sys.platform == "darwin":
        runs.pop(2)  # async_copy is CUDA-only; on a Mac it would just repeat the previous row
    results = {label: run(label, repo, args, {**extra, **kw}) for label, repo, kw in runs}

    base = results.get("original")
    print("\n=== summary ===")
    for label, r in results.items():
        if r is None:
            print(f"{label:45s} FAILED")
            continue
        speedup = base["s_per_token"] / r["s_per_token"] if base else float("nan")
        match = "same tokens" if base and r["tokens"] == base["tokens"] else "TOKENS DIFFER"
        disk = "     n/a" if r.get("disk_mb_per_token") is None else f"{r['disk_mb_per_token']:8.0f}"
        print(f"{label:45s} {r['s_per_token']:7.2f} s/tok  x{speedup:5.2f}  "
              f"VRAM {r['peak_vram_mb']:7.0f} MB  RSS {r['peak_rss_mb']:7.0f} MB  "
              f"disk {disk} MB/tok  {match}")


if __name__ == "__main__":
    main()
