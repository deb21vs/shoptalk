"""Create a Chroma vector index from processed products (+ optional captions)."""
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from config import PROC_DIR, CAPTIONS_DIR, CHROMA_DIR, EMBED_MODEL

def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    products = pd.read_parquet(PROC_DIR / "products.parquet")
    if (CAPTIONS_DIR / "captions.parquet").exists():
        caps = pd.read_parquet(CAPTIONS_DIR / "captions.parquet")
        products = products.merge(caps, on="product_id", how="left")
    else:
        products["caption"] = ""

    products["augmented_text"] = (
        products["title"].fillna("") + ". " +
        products["brand"].fillna("") + ". " +
        products["category"].fillna("") + ". " +
        products["description"].fillna("") + ". " +
        products["tags"].fillna("") + ". " +
        products["caption"].fillna("")
    ).str.strip()

    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection("products")

    # Clear existing to allow repeatable runs
    try:
        ids_existing = col.get()["ids"]
        if ids_existing:
            col.delete(ids=ids_existing)
    except Exception:
        pass

    BATCH = 256
    for i in tqdm(range(0, len(products), BATCH)):
        chunk = products.iloc[i:i+BATCH]
        texts = chunk["augmented_text"].tolist()
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        col.add(
            ids=[str(x) for x in chunk["product_id"].tolist()],
            documents=texts,
            embeddings=vecs.tolist(),
            metadatas=[{
                "title": r.title,
                "brand": r.brand,
                "category": r.category,
                "url": r.url,
                "image_url": r.image_url
            } for r in chunk.itertuples(index=False)]
        )

    print("[OK] Indexed rows:", len(products))

if __name__ == "__main__":
    main()
