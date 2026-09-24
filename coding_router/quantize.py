"""Model download, caching, and quantization utilities for the classifier.

The adapted-arch-router-1.5B classifier is distributed as a pre-quantised
GGUF file on Hugging Face.  This module handles:

  1. Locating the local cache directory.
  2. Downloading the GGUF if it isn't cached yet.
  3. Providing the path to the cached model file.

The GGUF format is used by llama-cpp-python and runs on CPU without any
GPU requirements, making it ideal for a pip package.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

_CACHE_ENV = "CODING_ROUTER_CACHE_DIR"

# Pre-quantised GGUF hosted on Hugging Face.
# The Q4_K_M variant balances quality and size (~600-900 MB for a 1.5B model).
DEFAULT_GGUF_REPO = "katanemo/adapted-arch-router-1.5B-GGUF"
DEFAULT_GGUF_FILENAME = "adapted-arch-router-1.5B-Q4_K_M.gguf"


def cache_dir() -> Path:
    """Return (and create) the directory where downloaded models are cached."""
    if env := os.environ.get(_CACHE_ENV):
        path = Path(env)
    else:
        path = Path.home() / ".cache" / "coding-router"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Simple ASCII progress bar for download feedback."""
    if total <= 0:
        return f"{current / 1_048_576:.1f} MB"
    fraction = min(current / total, 1.0)
    filled = int(width * fraction)
    bar = "█" * filled + "░" * (width - filled)
    mb_current = current / 1_048_576
    mb_total = total / 1_048_576
    return f"|{bar}| {mb_current:.1f}/{mb_total:.1f} MB ({fraction:.0%})"


def download_gguf(
    repo_id: str = DEFAULT_GGUF_REPO,
    filename: str = DEFAULT_GGUF_FILENAME,
    *,
    force: bool = False,
) -> Path:
    """Download a GGUF model file from Hugging Face Hub.

    Uses ``huggingface_hub`` for robust, resumable downloads with proper
    cache management.  Returns the local path to the downloaded file.
    """
    local_path = cache_dir() / filename

    if local_path.exists() and not force:
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required to download the classifier model.  "
            "Install it with:  pip install huggingface-hub"
        ) from error

    print(
        f"Downloading classifier model: {repo_id}/{filename}\n"
        f"  → {local_path}\n"
        "  This is a one-time download (~600 MB)...",
        file=sys.stderr,
    )

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(cache_dir()),
        local_dir_use_symlinks=False,
    )

    # huggingface_hub may place the file in a subdirectory; move to our
    # flat cache if needed.
    downloaded_path = Path(downloaded)
    if downloaded_path != local_path:
        downloaded_path.rename(local_path)

    print(f"  ✓ Download complete: {local_path}", file=sys.stderr)
    return local_path


def ensure_model(
    repo_id: str = DEFAULT_GGUF_REPO,
    filename: str = DEFAULT_GGUF_FILENAME,
) -> Path:
    """Return the path to the cached GGUF, downloading it if needed."""
    return download_gguf(repo_id, filename)


# ---------------------------------------------------------------------------
# Platform-specific llama-cpp-python settings
# ---------------------------------------------------------------------------


def llama_cpp_kwargs() -> dict:
    """Return sensible default kwargs for ``Llama(...)`` on this platform.

    Optimisations:
      - macOS: Enable Metal GPU offloading when available.
      - Windows/Linux with CUDA: Enable GPU layers.
      - Everywhere: Use mmap for memory-efficient loading.
    """
    system = platform.system()
    kwargs: dict = {
        "use_mmap": True,        # memory-mapped I/O — fast cold start
        "use_mlock": False,      # don't pin in RAM on low-memory systems
        "verbose": False,        # suppress llama.cpp logging
    }

    if system == "Darwin":
        # macOS — Metal acceleration.  n_gpu_layers=-1 means "offload all".
        # llama-cpp-python with Metal is the default pip install on Apple Silicon.
        kwargs["n_gpu_layers"] = -1
    elif system == "Windows":
        # Windows — conservative CPU default.  If the user has a CUDA build
        # of llama-cpp-python, they can override via the constructor.
        kwargs["n_gpu_layers"] = 0
    else:
        # Linux — try GPU offloading; harmless if no GPU backend is compiled.
        kwargs["n_gpu_layers"] = -1

    return kwargs
