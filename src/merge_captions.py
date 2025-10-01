"""
Merge caption outputs back into products.parquet to create products_enriched.parquet.

Supports:
- --source products  -> uses data/captions/captions_products.parquet (1 caption per product)
- --source images    -> uses data/captions/captions_images.parquet (many captions per product);
                       prefers main image caption, else longest; can also join top-K captions.

Adds:
- caption_main        (best caption per product)
- captions_joined     (optional, top-K joined; images source only)
- search_text         (title + description + caption(s) + tags + brand + category)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

try:
    # normal module run: python -m src.merge_captions
    from src.config import PROC_DIR, CAPTIONS_DIR
except Exception:
    # fall back if executed from src/
    from config import PROC_DIR, CAPTIONS_DIR


def build_search_text(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    return df[fields].fillna("").agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()


def merge_from_products() -> pd.DataFrame:
    prods = pd.read_parquet(PROC_DIR / "products.parquet")
    caps_path = CAPTIONS_DIR / "captions_products.parquet"
    if not caps_path.exists():
        raise FileNotFoundError(f"Missing captions file: {caps_path}")
    caps = pd.read_parquet(caps_path)
    caps = caps.dropna(subset=["caption"]).drop_duplicates("product_id")

    out = prods.merge(caps[["product_id", "caption"]], on="product_id", how="left")
    out = out.rename(columns={"caption": "caption_main"})
    out["search_text"] = build_search_text(
        out, ["title", "description", "caption_main", "tags", "brand", "category"]
    )
    return out


def merge_from_images(topk: int = 3) -> pd.DataFrame:
    prods = pd.read_parquet(PROC_DIR / "products.parquet")

    imgs_path = PROC_DIR / "product_images.parquet"
    caps_path = CAPTIONS_DIR / "captions_images.parquet"
    if not imgs_path.exists():
        raise FileNotFoundError(f"Missing image mapping file: {imgs_path}")
    if not caps_path.exists():
        raise FileNotFoundError(f"Missing captions file: {caps_path}")

    imgs = pd.read_parquet(imgs_path)               # product_id, image_id, image_path, is_main, width, height...
    caps = pd.read_parquet(caps_path)               # product_id, image_id, image_path, caption, is_main (maybe)

    # Ensure we have is_main in df used for ranking
    df = imgs.merge(caps[["image_id", "caption"]], on="image_id", how="left")
    df["is_main"] = df.get("is_main", False).fillna(False)
    df["cap_len"] = df["caption"].fillna("").str.len()

    # Best caption = prefer main image, else longest caption
    best = (
        df.sort_values(["product_id", "is_main", "cap_len"], ascending=[True, False, False])
          .drop_duplicates("product_id")[["product_id", "caption"]]
          .rename(columns={"caption": "caption_main"})
    )

    # Top-K joined captions for richer search text
    topk_joined = (
        df.sort_values(["product_id", "is_main", "cap_len"], ascending=[True, False, False])
          .groupby("product_id")["caption"]
          .apply(lambda s: "; ".join([c for c in s.head(topk).dropna() if c]))
          .rename("captions_joined")
          .reset_index()
    )

    out = prods.merge(best, on="product_id", how="left").merge(topk_joined, on="product_id", how="left")
    out["search_text"] = build_search_text(
        out, ["title", "description", "caption_main", "captions_joined", "tags", "brand", "category"]
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["products", "images"], default="products",
                    help="Which captions parquet to merge from.")
    ap.add_argument("--topk", type=int, default=3, help="Top-K captions to join (only for --source images).")
    ap.add_argument("--out", type=str, default=None, help="Override output path.")
    args = ap.parse_args()

    if args.source == "products":
        enriched = merge_from_products()
    else:
        enriched = merge_from_images(topk=args.topk)

    out_path = Path(args.out) if args.out else (PROC_DIR / "products_enriched.parquet")
    enriched.to_parquet(out_path, index=False)

    cov = (enriched["caption_main"].notna()).mean() * 100
    print(f"[OK] Saved {out_path} | rows={len(enriched)} | caption_coverage={cov:.1f}%")
    print("[OK] Columns:", list(enriched.columns))

if __name__ == "__main__":
    main()
