# LangGraph coding-model router

This project routes a coding request through the following graph:

```text
user query
  → Qwen/Qwen3-Embedding-0.6B query embedding
  → existing adapted-arch-router vLLM classifier (localhost:8000)
  → cosine similarity against embedded model catalog
  → standard or advanced model subgroup
  → selected model provider
```

The model descriptions in `data/coding_llm_models.json` are embedded once using
`Qwen/Qwen3-Embedding-0.6B`. The normalized vectors and catalog fingerprint are
saved to `data/model_embeddings_qwen3_0.6b.npz`. Later runs reuse that file;
changing the catalog or embedding model automatically rebuilds it.

Each document includes the capability description, deployment metadata, and token
prices. Qwen cosine similarity is the capability-fit proxy. The decision rule then
follows the paper's performance-minus-cost formulation:

```text
selected model = argmax_m [ wq * normalized_quality(m, query)
                            + wc * (1 - normalized_estimated_request_cost(m)) ]
```

This is equivalent to maximizing performance minus a scaled cost penalty after
normalizing the active candidate pool. Estimated request cost uses the input length
and `--max-output-tokens` with each model's per-million-token rates.

## Install

Use a Python environment with PyTorch appropriate for the local GPU/CPU, then:

```bash
cd /Users/indersharma/Developer/router
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first embedding build downloads the Qwen embedding weights from Hugging Face.
Keep the vLLM classification server running at `http://localhost:8000/v1`.
The encoder follows Qwen's documented left-padding and normalized last-token pooling recipe.

## Build the persisted index

```bash
python -m model_router --build-index
```

## Route without calling a provider

This is the recommended first check. It calls the local classifier and produces
the selected catalog record, but does not send the request to a cloud/local target.

```bash
python -m model_router --route-only "Fix an intermittent race condition in our Python worker"
python -m model_router --route-only --force-advanced "Plan a multi-service migration"
python -m model_router --route-only --mode cost_efficient "Generate a small Python utility"
```

## Routing modes

| Mode | Quality weight | Cost-efficiency weight | Use it when |
| --- | ---: | ---: | --- |
| `cost_efficient` | 0.15 | 0.85 | Cost should dominate, without restricting the route to open-source models. |
| `skill_based` | 1.00 | 0.00 | The most capable semantic match matters regardless of price. |
| `mixed` (default) | 0.65 | 0.35 | You want a balanced quality/cost decision. |

The graph runs an advanced subgroup for explicit complex-work cues. In
`skill_based` mode it may also enter that subgroup when the best semantic match is
an advanced model.

## Add models you use

Your own candidates live in `data/user_models.json` and can be mixed with the
supplied catalog or routed exclusively. Add a local vLLM/Ollama server:

```bash
python -m model_router add-model \
  --key my-local-coder \
  --model my-coder-model \
  --endpoint http://localhost:9000/v1 \
  --feature "Fast local coding model for Python, TypeScript, debugging, and code generation" \
  --size "14B" \
  --input-price 0 --output-price 0
```

Or add a cloud OpenAI-compatible provider with real prices and an optional key
environment-variable name:

```bash
python -m model_router add-model \
  --key team-cloud-coder \
  --model provider-coder-v1 \
  --endpoint https://provider.example/v1 \
  --feature "Advanced agentic model for repository refactors, security reviews, and migrations" \
  --size "Unknown" --tier advanced \
  --input-price 0.8 --output-price 3.2 \
  --api-key-env TEAM_CLOUD_CODER_API_KEY
```

Use only your added, available models with `--pool user`. Adding or updating a
model automatically invalidates and rebuilds the persisted embedding index on the
next route.

## Route and invoke

Omit `--route-only` to send the user query to the selected endpoint:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
python -m model_router "Design a caching layer for this API"
```

The router uses the appropriate key for the selected service. Supported names are
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`,
`DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `MISTRAL_API_KEY`, `ZAI_API_KEY`,
`MOONSHOT_API_KEY`, `MINIMAX_API_KEY`, and `LOCAL_LLM_API_KEY`. For a proxy or
model-specific credential, set `MODEL_ROUTER_<CATALOG_KEY>_API_KEY`, replacing
hyphens with underscores. Local Ollama/vLLM servers work with no key when they do
not require authentication.

## Advanced subgroup

The graph has separate `standard_model_group` and `advanced_model_group` nodes.
Complexity cues such as architecture, repository-wide changes, migrations,
security audits, and long input automatically select the advanced group. Use
`--force-advanced` to override it. The advanced list is explicit in
`model_router/catalog.py`, making the cost/quality policy easy to audit.

Use `--show-ranking` to inspect every semantic score and category tie-breaker.
