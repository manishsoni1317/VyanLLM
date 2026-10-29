# VyanLLM

Run large language models on small GPUs and Macs by streaming the model one layer at a time
from disk, so a model much bigger than your GPU memory can still run.

VyanLLM is derived from [AirLLM](https://github.com/lyogavin/airllm) (Apache-2.0) and adds
speed and memory improvements on top of it. See `NOTICE` for the full list of changes.

## Install

```bash
pip install -e .                 # from this folder
pip install -e ".[compression]"  # + 4-bit / 8-bit compression (NVIDIA)
pip install -e ".[mac]"          # + the macOS MLX backend
```

Python 3.10 or newer is required.

## Use

```python
from vyanllm import AutoModel

model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
ids = model.tokenizer(["What is the capital of India?"], return_tensors="pt")["input_ids"]
out = model.generate(ids.to(model.device), max_new_tokens=20)
print(model.tokenizer.decode(out[0]))
```

On a Mac, `AutoModel` uses the MLX backend. To use the PyTorch path (Apple GPU via MPS),
where VyanLLM's speedups apply, construct the base class directly:

```python
from vyanllm.vyanllm_base import VyanLLMBaseModel
model = VyanLLMBaseModel("Qwen/Qwen2.5-7B-Instruct", cache_gb=4)
```

## Options added by VyanLLM

| Option | Default | What it does |
|---|---|---|
| `cache_gb` | `None` | Keep layers in memory between tokens instead of re-reading them from disk. A number of GB, or `"auto"`. On a Mac layers stay on the Apple GPU; on NVIDIA this is a pinned RAM tier. |
| `gpu_cache_gb` | `None` | NVIDIA only: keep layers resident in VRAM (filled before `cache_gb`). |
| `async_copy` | on with CUDA | Copy the next layer to the GPU on a side stream while the current one computes. Uses one extra layer of VRAM. |
| `memory_cleanup` | `"forward"` | `"layer"` restores AirLLM's cleanup after every layer (slow); `"end"` cleans only after `generate()`. |
| `precast_shards` | `True` | Convert wider-dtype shards (e.g. fp32 run as bf16) once, so each token reads fewer bytes. |
| `prefetching` | `True` | Read the next layer in the background. Now also works with `compression`. |

`"auto"` leaves at least 4 GB or 15% of RAM free. If that leaves nothing for the cache, it
says so and turns the cache off. On machines with little RAM, pass a number instead.

## Benchmark

`examples/benchmark.py` compares an original AirLLM checkout against VyanLLM and checks the
generated tokens are identical:

```bash
git clone https://github.com/lyogavin/airllm airllm-orig
python examples/benchmark.py --orig airllm-orig/air_llm --new . \
    --model Qwen/Qwen2.5-0.5B-Instruct --tokens 8     # add --backend torch on a Mac
```

### Results

Measured with `examples/benchmark.py` on a MacBook Air (Apple M3, 16 GB, PyTorch MPS backend),
`Qwen/Qwen2.5-0.5B-Instruct`, 8 generated tokens. All runs produce identical tokens.

| Run | Time per token | Speedup | Peak RAM |
|---|---|---|---|
| Original AirLLM | 1.58 s | 1.0x | 1257 MB |
| VyanLLM (defaults) | 0.20 s | 8.0x | 630 MB |
| VyanLLM, per-layer cleanup (AirLLM's old behaviour) | 1.60 s | 1.0x | 632 MB |
| VyanLLM + layer cache | 0.20 s | 8.0x | 626 MB |

Same machine (M3, 16 GB), `Qwen/Qwen2.5-3B-Instruct`, 16 generated tokens, identical tokens in all runs:

| Run | Time per token | Speedup | Peak RAM |
|---|---|---|---|
| Original AirLLM | 12.53 s | 1.0x | 2345 MB |
| VyanLLM (defaults) | 10.30 s | 1.22x | 600 MB |
| VyanLLM, per-layer cleanup (AirLLM's old behaviour) | 12.23 s | 1.02x | 485 MB |
| VyanLLM + layer cache | 10.20 s | 1.23x | 616 MB |

Running cleanup once per forward pass instead of after every layer is the source of the
speedup. On the small model, where per-layer overhead dominates, it is 8x. On the 3B model each
token reads about 5.3 GB of layer data from disk, so disk reads dominate and the gain is 1.2x;
peak RAM is about 74% lower. The layer cache did not help here because the cache was smaller
than the model. Results on larger models and on NVIDIA GPUs will differ; the script lets you
reproduce and extend these numbers.

## Tests

```bash
python -m pytest tests -q
```

Some tests need a CUDA GPU or network access and are skipped or fail without them.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
