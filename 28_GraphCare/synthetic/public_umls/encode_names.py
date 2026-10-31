# Encode concept and relation names with the local GTE model.

import gc
import hashlib
import importlib.util
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def encode(directory, graph, output, batch_size=4):
    directory, output = Path(directory).resolve(), Path(output)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_MODULES_CACHE"] = str(directory.parent / "hf_modules")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    texts = [item["name"] for item in graph["nodes"]] + [
        relation.replace("_", " ") for relation in graph["relations"]
    ]
    print("Loading checked local GTE weights on CPU...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=True
    )
    # AutoModel rejects an optional GPU import, so import the local source directly.
    specification = importlib.util.spec_from_file_location(
        "verified_local_gte_qwen", directory / "modeling_qwen.py"
    )
    implementation = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(implementation)
    model = implementation.Qwen2Model.from_pretrained(
        str(directory),
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    vectors = []
    for offset in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[offset : offset + batch_size],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(
                **encoded, use_cache=False, is_causal=False
            ).last_hidden_state
            # Use the last token, as the GTE model card specifies.
            if bool(
                (
                    encoded["attention_mask"][:, -1].sum()
                    == encoded["input_ids"].shape[0]
                ).item()
            ):
                pooled = hidden[:, -1]
            else:
                last = encoded["attention_mask"].sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0]), last]
            vectors.append(F.normalize(pooled, p=2, dim=1).cpu())
        print(
            "Encoded %d/%d public text names"
            % (min(offset + batch_size, len(texts)), len(texts)),
            flush=True,
        )
    embeddings = torch.cat(vectors)
    assert embeddings.shape == (len(texts), 1536) and torch.isfinite(embeddings).all()
    count = len(graph["nodes"])
    node_embedding, relation_embedding = (
        embeddings[:count].clone(),
        embeddings[count:].clone(),
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "public_name_vectors.npz",
        node=node_embedding.numpy(),
        relation=relation_embedding.numpy(),
    )
    report = {
        "encoder": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "revision": "a9af15a6372d7d6b25e9fb07c2ccb9e1fe645644",
        "device": "cpu",
        "dtype": "float32",
        "attention": "eager, bidirectional",
        "batch_size": batch_size,
        "max_tokens": 128,
        "pooling": "last unpadded token, followed by L2 normalization",
        "input_texts": len(texts),
        "concepts": count,
        "relations": len(graph["relations"]),
        "output_width": 1536,
        "minimum_norm": float(embeddings.norm(dim=1).min()),
        "maximum_norm": float(embeddings.norm(dim=1).max()),
        "vector_bytes_sha256": hashlib.sha256(embeddings.numpy().tobytes()).hexdigest(),
        "inference_api_calls": 0,
        "input_text_origin": "Actual concept and relation names from author-public UMLS files.",
    }
    report["loading_method"] = (
        "Direct import of hash-checked local Qwen2Model class, retaining its conditional optional FlashAttention import and selecting its eager CPU implementation."
    )
    del model, tokenizer, hidden, vectors, embeddings
    gc.collect()
    return node_embedding, relation_embedding, report
