"""Qwen embedding model and an on-disk cosine-similarity index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .catalog import ModelProfile


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCTION = (
    "Given a coding request, retrieve the most suitable model profile for the task."
)


class QwenEmbeddingEncoder:
    """Qwen's documented AutoModel + normalized last-token pooling encoder."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "Missing embedding dependencies. Install with: pip install -r requirements.txt"
                ) from error

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )
            self._model = AutoModel.from_pretrained(self.model_name)
            # Use the fastest locally available hardware, without requiring
            # accelerate or a particular vLLM deployment for embedding builds.
            # Qwen publishes BF16 weights. CUDA supports those directly; CPU is
            # the reliable fallback (many MPS installations do not support BF16).
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(device).eval()
        return self._tokenizer, self._model

    def _encode(self, texts: Sequence[str]):
        """Implement Qwen's official left-padded, last-token pooling recipe."""
        import torch
        import torch.nn.functional as functional

        tokenizer, model = self._load()
        batch = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        batch = {name: value.to(model.device) for name, value in batch.items()}
        with torch.inference_mode():
            hidden_states = model(**batch).last_hidden_state
        attention_mask = batch["attention_mask"]
        if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
            embeddings = hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            row_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            embeddings = hidden_states[row_indices, sequence_lengths]
        return functional.normalize(embeddings.float(), p=2, dim=1).cpu().numpy()

    def encode_documents(self, documents: Sequence[str]):
        """Embed catalog entries once; vectors are L2-normalized for dot-product cosine."""
        return self._encode(documents)

    def encode_query(self, query: str):
        """Embed a query with Qwen's retrieval instruction, matching its recommended use."""
        prompted_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
        return self._encode([prompted_query])[0]


@dataclass
class EmbeddingIndex:
    ids: list[str]
    vectors: object
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> "EmbeddingIndex | None":
        if not path.exists():
            return None
        try:
            import numpy as np

            with np.load(path, allow_pickle=False) as data:
                return cls(
                    ids=data["ids"].astype(str).tolist(),
                    vectors=data["vectors"].astype("float32"),
                    fingerprint=str(data["fingerprint"].item()),
                )
        except (KeyError, OSError, ValueError):
            return None

    @classmethod
    def ensure(
        cls,
        *,
        path: Path,
        profiles: Sequence[ModelProfile],
        fingerprint: str,
        encoder: QwenEmbeddingEncoder,
    ) -> tuple["EmbeddingIndex", bool]:
        """Load a current index or embed every catalog profile and persist it atomically."""
        expected_ids = [profile.catalog_key for profile in profiles]
        existing = cls.load(path)
        if existing and existing.fingerprint == fingerprint and existing.ids == expected_ids:
            return existing, False

        import numpy as np

        vectors = np.asarray(encoder.encode_documents([p.document() for p in profiles]), dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(profiles):
            raise RuntimeError("Embedding model returned an unexpected vector shape.")
        index = cls(ids=expected_ids, vectors=vectors, fingerprint=fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        np.savez_compressed(
            temporary_path,
            ids=np.asarray(index.ids),
            vectors=index.vectors,
            fingerprint=np.asarray(index.fingerprint),
        )
        temporary_path.replace(path)
        return index, True

    def cosine_scores(self, query_vector) -> list[float]:
        """Vectors and query are normalized, so their dot product is cosine similarity."""
        import numpy as np

        query = np.asarray(query_vector, dtype="float32")
        if query.ndim != 1 or self.vectors.shape[1] != query.shape[0]:
            raise ValueError("Query embedding dimension does not match the stored index.")
        return (self.vectors @ query).astype(float).tolist()
