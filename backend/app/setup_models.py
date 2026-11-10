"""Provision model artifacts before starting an offline deployment."""
from pathlib import Path
from huggingface_hub import try_to_load_from_cache

MODEL_NAMES = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


def models_ready():
    def cached(model, filename):
        value = try_to_load_from_cache(model, filename)
        return isinstance(value, str) and Path(value).is_file()
    return all(
        cached(model, "config.json")
        and any(cached(model, weight) for weight in ("model.safetensors", "pytorch_model.bin"))
        and any(cached(model, tokenizer) for tokenizer in ("tokenizer.json", "vocab.txt"))
        for model in MODEL_NAMES
    )


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SentenceTransformer(MODEL_NAMES[0], local_files_only=False)
    CrossEncoder(MODEL_NAMES[1], local_files_only=False)
    print("Embedding and reranking models are ready.")
