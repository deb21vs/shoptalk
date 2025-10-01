"""
Transform raw ABO product records into tidy tables and link images.

Inputs (under data/raw):
- ABO listings: listings_*.json.gz (JSONL, one JSON per line)
- ABO image metadata: abo-images-small/images/metadata/images.csv.gz
  (with fields: image_id,height,width,path ; path is relative under images/small/)

Outputs (under data/processed):
- products.parquet (one row per product, includes:
    product_id, title, brand, category, description, tags,
    image_url, url, color, color_code, material, country,
    width_in, length_in, height_in, weight_lb, marketplace, domain_name,
    node_id, node_path, product_type, model_number, num_images)
- product_images.parquet (one row per image with:
    product_id, image_id, image_path, path, height, width, is_main)
"""
from __future__ import annotations

import gzip
import json
import re
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Expect RAW_DIR and PROC_DIR to be defined in src/config.py
from src.config import RAW_DIR, PROC_DIR

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROC_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_META = RAW_DIR / "abo-images-small" / "images" / "metadata" / "images.csv.gz"
IMAGES_ROOT = RAW_DIR / "abo-images-small" / "images" / "small"

from src.config import (
    RAW_DIR, PROC_DIR,
    LANG_PREF,
    USE_NORMALIZED_DIMS_FIRST, DIM_MIN_IN, DIM_MAX_IN,
    USE_NORMALIZED_WEIGHT_FIRST, WEIGHT_MIN_LB, WEIGHT_MAX_LB,
    ENABLE_TAG_TEXT_CLEAN,
    PREFER_RECORD_WITH_IMAGES,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _clean(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def _pick_lang_value(items: Any, lang: str = "en_US") -> str:
    """From a list of dicts with {language_tag, value}, pick preferred language, else first non-empty."""
    if not isinstance(items, list) or not items:
        return ""
    # exact language match
    for it in items:
        if it.get("language_tag") == lang and it.get("value"):
            return str(it["value"])
    # fallback: any value
    for it in items:
        if it.get("value"):
            return str(it["value"])
    return ""

def _join_lang_values(items: Any, lang: str = "en_US", sep: str = "; ") -> str:
    """Join multiple bullet points for a language; fallback to any language if none match."""
    if not isinstance(items, list) or not items:
        return ""
    vals = [it.get("value") for it in items if it.get("language_tag") == lang and it.get("value")]
    if not vals:
        vals = [it.get("value") for it in items if it.get("value")]
    return sep.join(_clean(v) for v in vals if isinstance(v, str))

def _get_dimension_in_inches(dims: dict, key: str) -> Optional[float]:
    """
    Pulls (normalized_value.value) if present, else (value).
    Returns float or None.
    """
    if not isinstance(dims, dict):
        return None
    node = dims.get(key) or {}
    nv = (node.get("normalized_value") or {}).get("value")
    if nv is not None:
        try:
            return float(nv)
        except Exception:
            pass
    v = node.get("value")
    if v is not None:
        try:
            return float(v)
        except Exception:
            return None
    return None

def _get_weight_lb(item_weight: Any) -> Optional[float]:
    """Prefer normalized_value.value (pounds) if present, else value."""
    if isinstance(item_weight, list) and item_weight:
        w0 = item_weight[0]
        nv = (w0.get("normalized_value") or {}).get("value")
        if nv is not None:
            try:
                return float(nv)
            except Exception:
                pass
        v = w0.get("value")
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None

def _extract_color(rec_color: Any) -> str:
    """
    Prefer standardized_values of en_US color entry, else value.
    Fallback to first entry.
    """
    if not isinstance(rec_color, list) or not rec_color:
        return ""
    # prefer en_US standardized_values/value
    for c in rec_color:
        if c.get("language_tag") == "en_US":
            std = c.get("standardized_values") or []
            if std:
                return _clean(std[0])
            return _clean(c.get("value", ""))
    # fallback: first record
    c0 = rec_color[0]
    std = c0.get("standardized_values") or []
    if std:
        return _clean(std[0])
    return _clean(c0.get("value", ""))

def _extract_color_code(codes: Any) -> str:
    """Join one or many hex codes into a single string (comma-separated)."""
    if isinstance(codes, list) and codes:
        return ", ".join(str(c) for c in codes if c)
    if isinstance(codes, str):
        return codes
    return ""

def _first_or_empty(items: Any, key: str = "value") -> str:
    """For fields like model_number: [{value: "..."}] -> string"""
    if isinstance(items, list) and items:
        return _clean(str(items[0].get(key, "")))
    return ""

# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------
def load_raw_records() -> List[Dict[str, Any]]:
    """
    Load all listings_*.json.gz (or *.json.gz as fallback) under RAW_DIR as JSONL.
    """
    files = sorted(glob.glob(str(RAW_DIR / "listings_*.json.gz"))) or sorted(glob.glob(str(RAW_DIR / "*.json.gz")))
    if not files:
        print(f"[WARN] No listings_*.json.gz files found in {RAW_DIR}.")
        return []
    rows: List[Dict[str, Any]] = []
    for fp in files:
        print(f"[INFO] Loading {fp}")
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows

# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------
def normalize_abo(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an ABO record into product-level fields.
    """
    product_id   = rec.get("item_id") or rec.get("asin") or rec.get("id")
    title        = _clean(_pick_lang_value(rec.get("item_name"), "en_US"))
    brand        = _clean(_pick_lang_value(rec.get("brand"), "en_US"))
    description  = _clean(_join_lang_values(rec.get("bullet_point"), "en_US"))

    node_list    = rec.get("node") or []
    node_id      = node_list[0].get("node_id") if (isinstance(node_list, list) and node_list) else None
    node_path    = _clean(node_list[0].get("node_name", "")) if (isinstance(node_list, list) and node_list) else ""

    category     = node_path  # keep the rubric's expected name

    color        = _extract_color(rec.get("color"))
    color_code   = _extract_color_code(rec.get("color_code"))
    material     = _clean(_pick_lang_value(rec.get("material"), "en_US"))
    country      = _clean(rec.get("country", ""))
    marketplace  = _clean(rec.get("marketplace", ""))
    domain_name  = _clean(rec.get("domain_name", ""))

    dims         = rec.get("item_dimensions") or {}
    width_in     = _get_dimension_in_inches(dims, "width")
    length_in    = _get_dimension_in_inches(dims, "length")
    height_in    = _get_dimension_in_inches(dims, "height")

    weight_lb    = _get_weight_lb(rec.get("item_weight"))

    main_image_id = rec.get("main_image_id")
    other_image_id = rec.get("other_image_id") or []
    image_ids = [i for i in ([main_image_id] if main_image_id else []) + (other_image_id if isinstance(other_image_id, list) else []) if i]

    product_type = _first_or_empty(rec.get("product_type"), "value")
    model_number = _first_or_empty(rec.get("model_number"), "value")

    # Build a simple tag field that helps baseline retrieval/search
    tags = ", ".join([t for t in [product_type, material, color, brand] if t])

    return {
        "product_id": product_id,
        "title": title,
        "brand": brand,
        "category": category,          # same as node_path for convenience
        "description": description,
        "tags": tags,
        "image_url": "",               # will be filled after image join (local path)
        "url": "",                     # ABO doesn’t ship canonical URLs
        "color": color,
        "color_code": color_code,
        "material": material,
        "country": country,
        "width_in": width_in,
        "length_in": length_in,
        "height_in": height_in,
        "weight_lb": weight_lb,
        "marketplace": marketplace,
        "domain_name": domain_name,
        "node_id": node_id,
        "node_path": node_path,
        "product_type": product_type,
        "model_number": model_number,
        # For image merging
        "main_image_id": main_image_id,
        "image_ids": image_ids,
    }

def normalize_router(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route to ABO normalizer if the record looks like ABO; otherwise store minimal generic.
    """
    if any(k in rec for k in ("item_id", "item_name", "node", "bullet_point")):
        return normalize_abo(rec)

    # Minimal generic fallback
    cats = rec.get("categories") or []
    cat_str = _clean(" > ".join(cats)) if isinstance(cats, list) else _clean(cats)
    desc = rec.get("description") or rec.get("bullet_points")
    kws = rec.get("keywords") or []
    tags = ", ".join(kws) if isinstance(kws, list) else (str(kws) if kws else "")

    return {
        "product_id": rec.get("asin") or rec.get("id"),
        "title": _clean(rec.get("title")),
        "brand": _clean(rec.get("brand")),
        "category": cat_str,
        "description": _clean(desc),
        "tags": _clean(tags),
        "image_url": rec.get("main_image") or rec.get("image_url") or "",
        "url": rec.get("url") or "",
        "color": _clean(rec.get("color", "")),
        "color_code": "",
        "material": _clean(rec.get("material", "")),
        "country": _clean(rec.get("country", "")),
        "width_in": None,
        "length_in": None,
        "height_in": None,
        "weight_lb": None,
        "marketplace": _clean(rec.get("marketplace", "")),
        "domain_name": _clean(rec.get("domain_name", "")),
        "node_id": None,
        "node_path": cat_str,
        "product_type": _clean(rec.get("product_type", "")),
        "model_number": _clean(rec.get("model_number", "")),
        "main_image_id": None,
        "image_ids": [],
    }

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    raw = load_raw_records()
    if not raw:
        print("[WARN] No raw records found; writing a tiny demo row.")
        df = pd.DataFrame([{
            "product_id": "DEMO-001",
            "title": "Demo Red Cotton T-Shirt",
            "brand": "DemoBrand",
            "category": "Clothing > Men > T-Shirts",
            "description": "A comfy red cotton tee suitable for daily wear.",
            "tags": "red, cotton, tee, men",
            "image_url": "",
            "url": "https://example.com/demo-red-tee",
            "color": "Red",
            "color_code": "#FF0000",
            "material": "Cotton",
            "country": "US",
            "width_in": None, "length_in": None, "height_in": None, "weight_lb": None,
            "marketplace": "Demo", "domain_name": "example.com",
            "node_id": None, "node_path": "Clothing/Men/T-Shirts",
            "product_type": "SHIRT", "model_number": "DEMO-TEE-1",
            "main_image_id": None, "image_ids": [],
        }])
    else:
        rows = [normalize_router(r) for r in tqdm(raw, desc="Normalizing")]
        df = pd.DataFrame(rows)

        # enforce valid product_id
        df = df.dropna(subset=["product_id"])
        # when duplicates exist, prefer the one that has image_ids/main_image_id populated
        df["_has_image_ref"] = df["image_ids"].apply(lambda x: bool(x))
        df = (
            df.sort_values(["_has_image_ref"], ascending=False)
              .drop_duplicates("product_id")
              .drop(columns=["_has_image_ref"])
        )

    # ---------- Build product↔image long table ----------
    product_images = pd.DataFrame(columns=["product_id", "image_id"])
    if "image_ids" in df.columns:
        tmp = df[["product_id", "image_ids"]].copy()
        tmp["image_ids"] = tmp["image_ids"].apply(lambda x: x if isinstance(x, list) else [])
        product_images = tmp.explode("image_ids").dropna().rename(columns={"image_ids": "image_id"})
        product_images = product_images[product_images["image_id"].astype(str) != ""]
    else:
        print("[INFO] No image_ids column found. Skipping image mapping.")

    # ---------- Join with images.csv.gz to get file paths ----------
    if IMAGES_META.exists():
        img_meta = pd.read_csv(IMAGES_META)
        # Standard fields in images.csv.gz: image_id, height, width, path
        # Build local absolute path to the small image
        img_meta["image_path"] = img_meta["path"].apply(lambda p: str(IMAGES_ROOT / p))
        img_meta = img_meta[["image_id", "image_path", "path", "height", "width"]]

        if not product_images.empty:
            product_images = product_images.merge(img_meta, on="image_id", how="inner")
        else:
            print("[INFO] No product image IDs to join; product_images remains empty.")
    else:
        print(f"[WARN] Image metadata not found at {IMAGES_META} — skipping image join.")

    # Add is_main flag and choose best per product for products.parquet
    if "main_image_id" in df.columns and not product_images.empty:
        main_map = df.set_index("product_id")["main_image_id"].to_dict()
        product_images["is_main"] = product_images.apply(
            lambda r: r["image_id"] == main_map.get(r["product_id"]), axis=1
        )
    else:
        product_images["is_main"] = False

    # choose representative image_url per product (prefer main if present)
    if not product_images.empty:
        best = (
            product_images.sort_values(["product_id", "is_main"], ascending=[True, False])
                         .drop_duplicates("product_id")
                         .loc[:, ["product_id", "image_path"]]
                         .rename(columns={"image_path": "image_url"})
        )
        df = df.drop(columns=["image_url"], errors="ignore").merge(best, on="product_id", how="left")
        # also store number of images per product
        counts = product_images.groupby("product_id", as_index=False).size().rename(columns={"size": "num_images"})
        df = df.merge(counts, on="product_id", how="left")
    else:
        df["num_images"] = 0

    # ---------- Final column order ----------
    col_order = [
        "product_id", "title", "brand", "category", "description", "tags",
        "image_url", "url", "color", "color_code", "material", "country",
        "width_in", "length_in", "height_in", "weight_lb",
        "marketplace", "domain_name", "node_id", "node_path",
        "product_type", "model_number", "num_images"
    ]
    # keep only expected + any extras (but order known ones first)
    final_cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order and c not in ("image_ids", "main_image_id")]
    df = df[final_cols]

    # ---------- Write outputs ----------
    products_out = PROC_DIR / "products.parquet"
    df.to_parquet(products_out, index=False)
    print("[OK] Saved:", products_out, "rows=", len(df))

    if not product_images.empty:
        images_out = PROC_DIR / "product_images.parquet"
        product_images.to_parquet(images_out, index=False)
        print("[OK] Saved:", images_out, "rows=", len(product_images))
    else:
        print("[INFO] No product_images to save.")

if __name__ == "__main__":
    main()
