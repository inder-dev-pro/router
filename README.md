# coding-router

A lightweight LLM router that classifies coding requests and selects the best-fit model using semantic similarity and cost-aware scoring.

Based on the LLMRouter paper (Feng et al., 2026):

```
reward_m = α · perf_normalized(m) − β · cost_normalized(m)
```

## Features

- **Local classifier** — runs the `adapted-arch-router-1.5B` classifier locally via a quantised GGUF model (~600 MB, auto-downloaded). No external server needed.
- **Semantic routing** — embeds your query and every model description with `Qwen3-Embedding-0.6B`, then ranks by cosine similarity + cost penalty.
- **Cost-aware selection** — five built-in routing modes from quality-only to cost-dominant, plus arbitrary `(α, β)` overrides.
- **Cross-platform** — Metal acceleration on macOS, CPU/CUDA on Windows and Linux.
- **Customisable catalog** — edit a JSONC file to enable only the models you actually use.

## Install

```bash
pip install coding-router
```

### Platform notes

| Platform | What happens |
|---|---|
| **macOS (Apple Silicon)** | `llama-cpp-python` auto-uses Metal for GPU acceleration |
| **macOS (Intel)** | CPU-only inference, works fine for the 1.5B classifier |
| **Windows** | CPU by default; install `llama-cpp-python` with CUDA support for GPU |
| **Linux** | GPU offloading attempted by default; falls back to CPU |

## Quick start

### 1. Initialise the model catalog

```bash
coding-router init
```

This copies the bundled model catalog to `~/.config/coding-router/coding_llm_models.jsonc`.

### 2. Enable the models you use

Open `~/.config/coding-router/coding_llm_models.json` and change `"enabled": false` to `"enabled": true` for the models you have API keys for (cloud) or running locally (self-hosted):

```json
// Before (disabled):
"claude-sonnet-5": {
  "enabled": false,
  "size": "Undisclosed (mid-tier)",
  ...
},

// After (enabled):
"claude-sonnet-5": {
  "enabled": true,
  "size": "Undisclosed (mid-tier)",
  ...
},
```

### 3. Set API keys

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
# etc.
```

### 4. Route a query

```bash
# Route without calling a provider (dry run):
coding-router --route-only "Fix an intermittent race condition in our Python worker"

# Route and invoke the selected model:
coding-router "Design a caching layer for this API"
```

## Python API

```python
from coding_router import CodingRouter, RouterConfig

router = CodingRouter()
result = router.route("Fix this race condition", route_only=True)

print(result["selected_model"]["catalog_key"])  # e.g. "claude-sonnet-5"
print(result["classifier_category"])            # e.g. "bug_fixing"
print(result["selected_model"]["reward"])        # cost-adjusted score
```

### Custom configuration

```python
from pathlib import Path
from coding_router import CodingRouter, RouterConfig

router = CodingRouter(RouterConfig(
    routing_mode="cost_efficient",       # favour cheaper models
    classifier_backend="local",          # default: local GGUF
    classifier_n_threads=4,              # limit CPU threads
    target_max_tokens=2048,
))
```

## Routing modes

| Mode | α (quality) | β (cost) | Use it when |
|---|---:|---:|---|
| `skill_based` | 1.0 | 0.0 | Best semantic match regardless of price |
| `quality_leaning` | 0.8 | 0.2 | Slight cost awareness |
| `mixed` (default) | 0.6 | 0.4 | Balanced quality/cost decision |
| `cost_sensitive` | 0.4 | 0.6 | Cost matters more than quality |
| `cost_efficient` | 0.2 | 0.8 | Cost should dominate |

Override with `--alpha` and `--beta` for arbitrary sweep points.

## Add custom models

```bash
# Add a local Ollama model:
coding-router add-model \
  --key my-local-coder \
  --model my-coder-model \
  --endpoint http://localhost:11434/v1 \
  --feature "Fast local coding model for Python and TypeScript" \
  --size "14B" \
  --input-price 0 --output-price 0

# Add a cloud model:
coding-router add-model \
  --key team-cloud-coder \
  --model provider-coder-v1 \
  --endpoint https://provider.example/v1 \
  --feature "Advanced agentic model for refactors and migrations" \
  --size "Unknown" --tier advanced \
  --input-price 0.8 --output-price 3.2 \
  --api-key-env TEAM_CLOUD_CODER_API_KEY
```

Use `--pool user` to route only among your added models.

## Classifier backends

### Local GGUF (default)

The classifier model is automatically downloaded on first use (~600 MB) and cached at `~/.cache/coding-router/`. No separate server required.

```bash
# Override the default GGUF path:
coding-router --classifier-model-path /path/to/custom.gguf "your query"
```

### Remote vLLM (advanced)

If you already have a vLLM server running the classifier:

```bash
coding-router --classifier-backend vllm --classifier-base-url http://localhost:8000/v1 "your query"
```

## Environment variables

| Variable | Purpose |
|---|---|
| `CODING_ROUTER_CONFIG_DIR` | Override config directory (default: `~/.config/coding-router/`) |
| `CODING_ROUTER_CACHE_DIR` | Override model cache directory (default: `~/.cache/coding-router/`) |
| `OPENAI_API_KEY` | OpenAI models |
| `ANTHROPIC_API_KEY` | Anthropic models |
| `GOOGLE_API_KEY` | Google models |
| `XAI_API_KEY` | xAI models |
| `DEEPSEEK_API_KEY` | DeepSeek models |
| `DASHSCOPE_API_KEY` | Alibaba/Qwen models |
| `MISTRAL_API_KEY` | Mistral AI models |
| `ZAI_API_KEY` | Z.ai (Zhipu) models |
| `MOONSHOT_API_KEY` | Moonshot AI models |
| `MINIMAX_API_KEY` | MiniMax models |

## License

MIT
