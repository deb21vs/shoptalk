# src/caption_images.py
"""
Batch image captioning to enrich product text.

Outputs (under data/captions):
- captions_products.parquet  (product_id, caption)                          # --source products
- captions_images.parquet    (product_id, image_id, image_path, caption, is_main)  # --source images
"""
from __future__ import annotations

import argparse, io, os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
import requests

try:
    # when run as: python -m src.caption_images
    from src.config import PROC_DIR, CAPTIONS_DIR, CAPTION_MODEL
except Exception:
    # when run directly from src/
    from config import PROC_DIR, CAPTIONS_DIR, CAPTION_MODEL

CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Device selection (prefers Apple MPS on Mac, else CUDA, else CPU) ----------------
def pick_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def autocast_ctx(device: torch.device):
    """
    Returns an autocast context for cuda/mps if supported; otherwise a nullcontext.
    """
    try:
        if device.type in ("cuda", "mps"):
            return torch.autocast(device_type=device.type, dtype=torch.float16)
    except Exception:
        pass
    from contextlib import nullcontext
    return nullcontext()


# ---------------- I/O helpers ----------------
def _is_http(path: str) -> bool:
    return isinstance(path, str) and (path.startswith("http://") or path.startswith("https://"))


def open_image_any(path: str) -> Image.Image | None:
    """Open image from local path or HTTP URL."""
    try:
        if _is_http(path):
            r = requests.get(path, timeout=10)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        if not os.path.exists(path):
            return None
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def load_inputs(source: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns needed for captioning.
    - products:  product_id, image_path
    - images:    product_id, image_id, image_path, is_main
    """
    if source == "products":
        df = pd.read_parquet(PROC_DIR / "products.parquet")
        col = "image_url" if "image_url" in df.columns else "image_path"
        df = df[df[col].fillna("").ne("")]
        df = df.rename(columns={col: "image_path"})[["product_id", "image_path"]]
        return df

    # source == "images"
    df = pd.read_parquet(PROC_DIR / "product_images.parquet")
    df = df[df["image_path"].fillna("").ne("")]
    if "is_main" not in df.columns:
        df["is_main"] = False
    return df[["product_id", "image_id", "image_path", "is_main"]]


# ---------------- Core batching ----------------
def run_batch(batch_rows: List[Tuple[dict, str]], processor, model, device: torch.device, source: str) -> list[dict]:
    """
    batch_rows: list of (row_dict, image_path)
    Returns list of dicts with caption outputs.
    """
    images, keep = [], []
    for r, p in batch_rows:
        img = open_image_any(p)
        if img is not None:
            images.append(img)
            keep.append((r, p))
    if not images:
        return []

    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.inference_mode(), autocast_ctx(device):
        out = model.generate(**inputs, max_new_tokens=30)

    texts = processor.batch_decode(out, skip_special_tokens=True)

    rows_out = []
    for (r, _), cap in zip(keep, texts):
        if source == "products":
            rows_out.append({"product_id": r["product_id"], "caption": cap})
        else:
            rows_out.append({
                "product_id": r["product_id"],
                "image_id": r["image_id"],
                "image_path": r["image_path"],
                "caption": cap,
                "is_main": bool(r.get("is_main", False)),
            })
    return rows_out


def checkpoint_append(rows: list[dict], out_path: Path, key_cols: list[str]):
    """Append and de-duplicate on key_cols."""
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all = df_all.drop_duplicates(key_cols, keep="first")
    df_all.to_parquet(out_path, index=False)
    print(f"[CHECKPOINT] saved {out_path} rows={len(df_all)}")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["products", "images"], default="products",
                    help="Caption one representative image per product, or all images.")
    ap.add_argument("--batch-size", type=int, default=12, help="Try 12–16 on Apple Silicon (M1/M2/M3).")
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true", help="Skip rows already captioned in the output parquet.")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    ap.add_argument("--no-shuffle", action="store_true", help="Disable default shuffle.")
    args = ap.parse_args()

    df = load_inputs(args.source)

    # Shuffle by default for better sampling/throughput on early runs
    if not args.no_shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if args.limit:
        df = df.head(args.limit)

    # Output + resume
    if args.source == "products":
        out_path = CAPTIONS_DIR / "captions_products.parquet"
        key_cols, id_col = ["product_id"], "product_id"
    else:
        out_path = CAPTIONS_DIR / "captions_images.parquet"
        key_cols, id_col = ["image_id"], "image_id"

    if args.resume and out_path.exists():
        done = set(pd.read_parquet(out_path)[id_col].astype(str).tolist())
        before = len(df)
        df = df[~df[id_col].astype(str).isin(done)]
        print(f"[RESUME] Skipping {before-len(df)} already-captioned rows; remaining: {len(df)}")

    if df.empty:
        print("[INFO] Nothing to caption.")
        return

    device = pick_device()
    print(f"[INFO] Using device: {device}")

    processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).to(device)
    model.eval()

    rows_out: list[dict] = []
    batch: list[Tuple[dict, str]] = []

    print(f"[INFO] Captioning {len(df)} rows ...")
    for _, r in tqdm(df.iterrows(), total=len(df)):
        row = r.to_dict()
        path = row["image_path"]
        batch.append((row, path))

        if len(batch) >= args.batch_size:
            rows_out += run_batch(batch, processor, model, device, args.source)
            batch = []
            if args.checkpoint_every and (len(rows_out) >= args.checkpoint_every):
                checkpoint_append(rows_out, out_path, key_cols)
                rows_out = []

    if batch:
        rows_out += run_batch(batch, processor, model, device, args.source)

    checkpoint_append(rows_out, out_path, key_cols)
    print(f"[FINAL] Saved captions to {out_path}")

if __name__ == "__main__":
    main()
