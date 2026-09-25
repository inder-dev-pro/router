# coding-router

A lightweight LLM router that classifies coding requests and selects the best-fit model using semantic similarity and cost-aware scoring.

Based on the LLMRouter paper (Feng et al., 2026):

```
reward_m = α · perf_normalized(m) − β · cost_normalized(m)
```

## Features

- **Local classifier** — runs the ModelGate router classifier locally via a quantised GGUF model (auto-downloaded). No external server needed.
- **Semantic routing** — embeds your query and every model description with `Qwen3-Embedding-0.6B`, then ranks by cosine similarity + cost penalty.
- **Cost-aware selection** — five built-in routing modes from quality-only to cost-dominant, plus arbitrary `(α, β)` overrides.
- **Cross-platform** — Metal acceleration on macOS, CPU/CUDA on Windows and Linux.
- **Project-local config** — catalog lives in `./router/coding_llm.json`, safe to commit to git.

## Install

```bash
pip install coding-router
```

### Platform notes

| Platform | What happens |
|---|---|
| **macOS (Apple Silicon)** | `llama-cpp-python` auto-uses Metal for GPU acceleration |
| **macOS (Intel)** | CPU-only inference, works fine for the classifier |
| **Windows** | CPU by default; install `llama-cpp-python` with CUDA support for GPU |
| **Linux** | GPU offloading attempted by default; falls back to CPU |

## Quick start

### 1. Initialise the project

```bash
cd your-project/
coding-router init
```

This creates:
- `router/coding_llm.json` — model catalog (all models included)
- `.env.example` — lists the API key env vars you need to set

### 2. Edit the catalog

Open `router/coding_llm.json` and **delete any models you don't have access to**. Every model present in the file is available for routing.

### 3. Set API keys

```bash
# Copy the example and fill in your keys
cp .env.example .env

# Or export directly:
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

> **Note:** API keys are never stored in the catalog. They're always read from environment variables at runtime. `router/coding_llm.json` is safe to commit to git.

### 4. Validate your setup

```bash
coding-router validate
```

This checks:
- ✓ JSON structure is valid
- ✓ All required fields are present
- ✓ Cloud models have their API keys set
- ✓ Local model endpoints are reachable

### 5. Route a query

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
    catalog_path=Path("router/coding_llm.json"),
    routing_mode="cost_efficient",       # favour cheaper models
    classifier_backend="llama-server",   # default: auto-starts llama serve
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

## CLI commands

| Command | What it does |
|---|---|
| `coding-router init` | Create `router/` dir with catalog + `.env.example` |
| `coding-router validate` | Check catalog, API keys, and endpoints |
| `coding-router models` | List all models in the catalog |
| `coding-router add-model` | Add a custom model to the catalog |
| `coding-router "query"` | Route (and optionally invoke) a query |
| `coding-router --route-only "query"` | Select a model without invoking |

## Config resolution

The router looks for the model catalog in this order:

1. Explicit `--catalog` path passed on the CLI
2. `LLMROUTER_CONFIG` environment variable
3. `./router/coding_llm.json` (project-local)
4. Error telling you to run `coding-router init`

## Classifier backends

### llama-server (default, recommended)

Auto-starts `llama serve -hf AaryanK/ModelGate:Q8_0` as a background process. The model stays loaded across calls for fast subsequent requests.

Requires `llama.cpp` installed:
```bash
brew install llama.cpp  # macOS
```

### Local GGUF (fallback)

Loads the GGUF model in-process via `llama-cpp-python`. No external binary needed, but reloads on each invocation.

```bash
coding-router --classifier-backend local "your query"
```

### Remote server (advanced)

Connect to any OpenAI-compatible server (vLLM, Ollama, llama-server):

```bash
coding-router --classifier-backend vllm --classifier-base-url http://localhost:8000/v1 "your query"
```

## Environment variables

| Variable | Purpose |
|---|---|
| `LLMROUTER_CONFIG` | Override catalog path (default: `./router/coding_llm.json`) |
| `CODING_ROUTER_CACHE_DIR` | Override model cache directory (default: `~/.cache/coding-router/`) |
| `OPENAI_API_KEY` | OpenAI models |
| `ANTHROPIC_API_KEY` | Anthropic models |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Google models |
| `XAI_API_KEY` | xAI models |
| `DEEPSEEK_API_KEY` | DeepSeek models |
| `DASHSCOPE_API_KEY` | Alibaba/Qwen models |
| `MISTRAL_API_KEY` | Mistral AI models |
| `ZAI_API_KEY` | Z.ai (Zhipu) models |
| `MOONSHOT_API_KEY` | Moonshot AI models |
| `MINIMAX_API_KEY` | MiniMax models |

## License

MIT
