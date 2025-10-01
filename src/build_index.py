"""
Build a persistent Chroma index using LangChain (GPU-aware).

- Reads the preprocessed table: data/processed/products.parquet
- (Optional) merges image captions from data/captions/captions.parquet
- Creates an augmented text field (title + brand + category + description + tags + color + material + caption)
- Embeds with HuggingFaceEmbeddings (normalized) via LangChain, using CUDA/MPS if available
- Writes vectors + metadata + ids into a persistent Chroma collection

Run:
    python src/build_index.py
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm
import torch

# langchain (vector store + embeddings)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# local config
from src.config import PROC_DIR, CAPTIONS_DIR, CHROMA_DIR

# optional config entries with safe fallbacks
try:
    from src.config import LC_EMBED_MODEL  # preferred when using LangChain
except Exception:
    LC_EMBED_MODEL = None

try:
    from src.config import EMBED_MODEL  # fallback if LC_EMBED_MODEL not set
except Exception:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------
# device helpers (CUDA / MPS)
# ---------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
# helpful on MPS when an op has no kernel; falls back to CPU for that op
if DEVICE == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
print(f"[INFO] Using device: {DEVICE}")

# ---------------------------
# tunables
# ---------------------------
COLLECTION_NAME = "products"
# how many rows we push per Chroma .add_texts() call (not the embedder batch)
BATCH_SIZE = 10_000
# embedding micro-batch (inside the encoder); larger on GPU, smaller on CPU
EMBED_BATCH = 256 if DEVICE != "cpu" else 64
NORMALIZE_EMBEDDINGS = True
CLEAR_EXISTING_INDEX = False   # set True to wipe index/chroma before rebuilding


def load_products() -> pd.DataFrame:
    """Load preprocessed products and (optionally) captions; build augmented_text."""
    parquet_path = PROC_DIR / "products.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}. Run preprocess_products.py first.")

    products = pd.read_parquet(parquet_path)

    # Optional captions
    cap_path = CAPTIONS_DIR / "captions.parquet"
    if cap_path.exists():
        caps = pd.read_parquet(cap_path)
        products = products.merge(caps, on="product_id", how="left")
        products["caption"] = products["caption"].fillna("")
    else:
        products["caption"] = ""

    # ensure fields exist
    for col in ["title", "brand", "category", "description", "tags", "color", "material", "caption"]:
        if col not in products.columns:
            products[col] = ""

    # augmented text
    products["augmented_text"] = (
        products["title"].fillna("") + ". " +
        products["brand"].fillna("") + ". " +
        products["category"].fillna("") + ". " +
        products["description"].fillna("") + ". " +
        products["tags"].fillna("") + ". " +
        products["color"].fillna("") + ". " +
        products["material"].fillna("") + ". " +
        products["caption"].fillna("")
    ).str.strip()

    products = products.dropna(subset=["product_id"]).drop_duplicates("product_id")

    print(f"[INFO] Loaded products: {len(products)} rows")
    missing_aug = (products["augmented_text"].str.len() == 0).sum()
    if missing_aug:
        print(f"[WARN] {missing_aug} rows have empty augmented_text (will still be indexed).")

    return products


def prepare_payload(products: pd.DataFrame) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Prepare ids, documents, and metadatas lists."""
    ids = [str(x) for x in products["product_id"].tolist()]
    docs = products["augmented_text"].astype(str).tolist()

    metas: List[Dict[str, Any]] = []
    for r in products.itertuples(index=False):
        md = {
            "product_id": getattr(r, "product_id", ""),
            "title": getattr(r, "title", ""),
            "brand": getattr(r, "brand", ""),
            "category": getattr(r, "category", ""),
            "url": getattr(r, "url", ""),
            "image_url": getattr(r, "image_url", ""),
            "color": getattr(r, "color", ""),
            "material": getattr(r, "material", ""),
            "country": getattr(r, "country", ""),
        }
        metas.append(md)

    return ids, docs, metas


def get_embeddings():
    """Construct a LangChain HF embedding function with device + normalization."""
    model_name = LC_EMBED_MODEL or EMBED_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
    print(
        f"[INFO] Using embeddings: {model_name} "
        f"(normalize={NORMALIZE_EMBEDDINGS}, device={DEVICE}, emb_batch={EMBED_BATCH})"
    )
    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        # sentencetransformers -> .to(device)
        model_kwargs={"device": DEVICE},
        # Do NOT include show_progress_bar here; LC will pass it itself.
        encode_kwargs={
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "batch_size": EMBED_BATCH,
        },
    )
    return emb


def build_index():
    """Create (or update) a persistent Chroma collection with LangChain."""
    if CLEAR_EXISTING_INDEX and CHROMA_DIR.exists():
        print(f"[INFO] Clearing existing index at {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    products = load_products()
    if products.empty:
        print("[WARN] No products to index. Exiting.")
        return

    ids, docs, metas = prepare_payload(products)

    # Debug preview
    print("\n[DEBUG] Example payload going into Chroma:")
    print("ID:", ids[0])
    print("Doc (augmented_text):", docs[0][:500], "..." if len(docs[0]) > 500 else "")
    print("Metadata:", metas[0])
    print("----------------------------------------------------\n")

    emb = get_embeddings()

    chroma_path = str(CHROMA_DIR)
    print(f"[INFO] Persist directory: {chroma_path}")
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=chroma_path,
        embedding_function=emb,
    )

    n = len(docs)
    for start in tqdm(range(0, n, BATCH_SIZE), desc="Indexing", unit="batch"):
        end = min(start + BATCH_SIZE, n)
        vectordb.add_texts(
            texts=docs[start:end],
            metadatas=metas[start:end],
            ids=ids[start:end],
        )

    vectordb.persist()
    print(f"[OK] Indexed rows: {len(ids)} into collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    build_index()
