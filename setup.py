import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

long_description = ""
_readme = os.path.join(here, "README.md")
if os.path.exists(_readme):
    with open(_readme, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setuptools.setup(
    name="vyanllm",
    version="0.1.0",
    author="Manish Kumar Soni",
    description="Run large language models on small GPUs and Macs by streaming one layer at a "
                "time, with prefetching, async GPU copies and a layer cache. "
                "Derived from AirLLM (Apache-2.0).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=setuptools.find_packages(include=["vyanllm", "vyanllm.*"]),
    python_requires=">=3.10",
    install_requires=[
        'tqdm',
        'torch>=2.4',
        'transformers>=4.49,<6',
        'accelerate>=1.0',
        'safetensors',
        'huggingface-hub',
        'scipy',
        'sentencepiece',
        'psutil',  # sizing the layer cache with cache_gb="auto"
    ],
    extras_require={
        'compression': ['bitsandbytes'],   # compression='4bit' / '8bit' (NVIDIA)
        'mac': ['mlx'],                    # the macOS MLX backend used by AutoModel
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
