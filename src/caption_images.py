# src/caption_images.py
"""
Batch image captioning to enrich product text.

Outputs (under data/captions):
- captions_products.parquet  (product_id, caption)                          # --source products
- captions_images.parquet    (product_id, image_id, image_path, caption, is_main)  # --source images
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate a few imperfect files

from tqdm import tqdm
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
import requests

# --- config import ---
try:
    # when run as: python -m src.caption_images
    from src.config import PROC_DIR, CAPTIONS_DIR, CAPTION_MODEL
except Exception:
    # when run directly from src/
    from config import PROC_DIR, CAPTIONS_DIR, CAPTION_MODEL

CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

# repo root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[1]

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
    """Returns an autocast context for cuda/mps if supported; otherwise a nullcontext."""
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

def normalize_local_path(p: str) -> Path:
    """
    Make a best-effort absolute filesystem path from whatever we got in the parquet.
    Handles: stray whitespace, backslashes, relative 'data/...', and ~.
    """
    p = str(p).strip().replace("\\", "/")
    p = os.path.expanduser(p)  # handle ~
    path = Path(p)
    if not path.is_absolute():
        # Typical in this repo: 'data/...'
        path = ROOT_DIR / p
    return path

def open_image_local(path_like: str) -> Image.Image | None:
    """
    Open a local image with PIL; return None on failure.
    We do NOT call img.verify() because it can be over-strict on valid JPEGs.
    """
    try:
        path = normalize_local_path(path_like)
        if not path.exists():
            return None
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None

def open_image_any(path: str) -> Image.Image | None:
    """Open image from local path or HTTP URL (http/https). Always return RGB if possible."""
    try:
        if _is_http(path):
            r = requests.get(path.strip(), timeout=10)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        # IMPORTANT: use the robust local loader
        return open_image_local(path)
    except Exception:
        return None

# ------------ bad-image logging ------------
_BAD_LOG = CAPTIONS_DIR / "bad_images.tsv"
def log_bad(product_id: str, image_path: str, reason: str):
    _BAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_BAD_LOG, "a", encoding="utf-8") as f:
        f.write(f"{product_id}\t{image_path}\t{reason}\n")

# ---------------- Input loading ----------------
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
    Returns list of dicts with caption outputs. Robust to individual failures.
    """
    rows_out: list[dict] = []

    # keep only images we can actually open and normalize to RGB
    images: list[Image.Image] = []
    keep: list[Tuple[dict, str]] = []
    for r, p in batch_rows:
        p_norm = normalize_local_path(p) if not _is_http(p) else p
        img = open_image_any(str(p_norm))
        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                try:
                    img = img.convert("RGB")
                except Exception:
                    img = None
            if img is not None:
                images.append(img)
                keep.append((r, str(p_norm)))
        else:
            # informative reason
            try:
                lp = normalize_local_path(p)
                reason = "not found" if (not _is_http(p) and not lp.exists()) else "PIL open failed"
            except Exception:
                reason = "path normalization/open failed"
            log_bad(str(r.get("product_id", "")), str(p), reason)

    if not images:
        return rows_out

    def decode_and_collect(out_tensors, kept_pairs):
        texts = processor.batch_decode(out_tensors, skip_special_tokens=True)
        out = []
        for (r, _), cap in zip(kept_pairs, texts):
            if source == "products":
                out.append({"product_id": r["product_id"], "caption": cap})
            else:
                out.append({
                    "product_id": r["product_id"],
                    "image_id": r["image_id"],
                    "image_path": r["image_path"],
                    "caption": cap,
                    "is_main": bool(r.get("is_main", False)),
                })
        return out

    # Try whole batch first (use the image_processor explicitly; return pixel_values only)
    try:
        enc = processor.image_processor(images=images, return_tensors="pt", padding=True)
        inputs = {"pixel_values": enc["pixel_values"].to(device)}
        with torch.inference_mode(), autocast_ctx(device):
            out = model.generate(**inputs, max_new_tokens=30)
        rows_out.extend(decode_and_collect(out, keep))
        return rows_out

    except Exception:
        # Fall back to per-image so only broken items are skipped
        for (r, p_str), img in zip(keep, images):
            try:
                enc1 = processor.image_processor(images=[img], return_tensors="pt", padding=True)
                inputs = {"pixel_values": enc1["pixel_values"].to(device)}
                with torch.inference_mode(), autocast_ctx(device):
                    out = model.generate(**inputs, max_new_tokens=30)
                rows_out.extend(decode_and_collect(out, [(r, p_str)]))
            except Exception:
                log_bad(str(r.get("product_id", "")), p_str, "processor/generate failed")
        return rows_out

def checkpoint_append(rows: list[dict], out_path: Path, key_cols: list[str]) -> bool:
    """Append and de-duplicate on key_cols. Returns True if we wrote anything."""
    if not rows:
        return False
    df_new = pd.DataFrame(rows)
    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all = df_all.drop_duplicates(key_cols, keep="first")
    df_all.to_parquet(out_path, index=False)
    print(f"[CHECKPOINT] saved {out_path} rows={len(df_all)}")
    return True

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["products", "images"], default="products",
                    help="Caption one representative image per product, or all images.")
    ap.add_argument("--batch-size", type=int, default=12, help="Batch size.")
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
        print(f"[RESUME] Skipping {before - len(df)} already-captioned rows; remaining: {len(df)}")

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
    wrote_any = False

    print(f"[INFO] Captioning {len(df)} rows ...")
    for _, r in tqdm(df.iterrows(), total=len(df)):
        row = r.to_dict()
        path = row["image_path"]
        batch.append((row, path))

        if len(batch) >= args.batch_size:
            rows_out += run_batch(batch, processor, model, device, args.source)
            batch = []
            if args.checkpoint_every and (len(rows_out) >= args.checkpoint_every):
                wrote_any |= checkpoint_append(rows_out, out_path, key_cols)
                rows_out = []

    if batch:
        rows_out += run_batch(batch, processor, model, device, args.source)

    wrote_any |= checkpoint_append(rows_out, out_path, key_cols)

    if wrote_any or out_path.exists():
        print(f"[FINAL] Saved captions to {out_path}")
    else:
        print("[FINAL] No captions written. Likely all images failed to open.")
        print(f"See {_BAD_LOG} for details.")

if __name__ == "__main__":
    main()
