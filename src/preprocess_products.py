# src/preprocess_products.py
"""
EC2/Kaggle-ready pipeline:
1) (Optional) EDA on raw ABO listings to justify preprocessing thresholds.
2) Transform raw ABO product records into tidy tables and link images.
   - Outputs:
     /processed/products.parquet       (one row per product)
     /processed/product_images.parquet (one row per image)

Fields produced in products.parquet:
    product_id, title, brand, category, description, tags,
    image_url, url, color, color_code, material, country,
    width_in, length_in, height_in, weight_lb, marketplace, domain_name,
    node_id, node_path, product_type, model_number, num_images
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 0) Config resolution (CLI > env vars > optional src.config > internal defaults)
# -----------------------------------------------------------------------------
def _try_import_repo_config():
    try:
        # Allow seamless use inside your repo; harmless if missing on EC2/Kaggle
        from src.config import (  # type: ignore
            ROOT, DATA_DIR, RAW_DIR, PROC_DIR,
            LANG_PREF,
            USE_NORMALIZED_DIMS_FIRST, DIM_MIN_IN, DIM_MAX_IN,
            USE_NORMALIZED_WEIGHT_FIRST, WEIGHT_MIN_LB, WEIGHT_MAX_LB,
            ENABLE_TAG_TEXT_CLEAN,
            PREFER_RECORD_WITH_IMAGES,
        )
        return dict(
            RAW_DIR=RAW_DIR, PROC_DIR=PROC_DIR,
            LANG_PREF=LANG_PREF,
            USE_NORMALIZED_DIMS_FIRST=USE_NORMALIZED_DIMS_FIRST,
            DIM_MIN_IN=DIM_MIN_IN, DIM_MAX_IN=DIM_MAX_IN,
            USE_NORMALIZED_WEIGHT_FIRST=USE_NORMALIZED_WEIGHT_FIRST,
            WEIGHT_MIN_LB=WEIGHT_MIN_LB, WEIGHT_MAX_LB=WEIGHT_MAX_LB,
            ENABLE_TAG_TEXT_CLEAN=ENABLE_TAG_TEXT_CLEAN,
            PREFER_RECORD_WITH_IMAGES=PREFER_RECORD_WITH_IMAGES,
        )
    except Exception:
        return {}

DEFAULTS = {
    # ---- paths (can be overridden with CLI or env) ----
    "RAW_DIR": Path(os.getenv("RAW_DIR", "data/raw")),
    "PROC_DIR": Path(os.getenv("PROC_DIR", "data/processed")),

    # ---- EDA-informed preprocessing knobs ----
    "LANG_PREF": ["en_US", "en_GB", "en_IN", "*"],  # "*" = any non-empty fallback
    "USE_NORMALIZED_DIMS_FIRST": True,
    "DIM_MIN_IN": 0.0,
    "DIM_MAX_IN": 200.0,
    "USE_NORMALIZED_WEIGHT_FIRST": True,
    "WEIGHT_MIN_LB": 0.0,
    "WEIGHT_MAX_LB": 200.0,
    "ENABLE_TAG_TEXT_CLEAN": True,
    "PREFER_RECORD_WITH_IMAGES": True,
}

REPO_CFG = _try_import_repo_config()
CFG = {**DEFAULTS, **REPO_CFG}  # repo config wins over defaults; CLI will win over this


# -----------------------------------------------------------------------------
# 1) CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ABO preprocess (+ optional EDA) — EC2/Kaggle ready"
    )
    p.add_argument("--raw-dir", type=Path, default=CFG["RAW_DIR"],
                   help="Source dir with ABO raw files (default: data/raw)")
    p.add_argument("--proc-dir", type=Path, default=CFG["PROC_DIR"],
                   help="Output dir for processed parquet (default: data/processed)")
    p.add_argument("--images-root", type=Path, default=None,
                   help="Override path to abo-images-small root; "
                        "default: <RAW_DIR>/abo-images-small")
    p.add_argument("--use-normalized-dims-first", action="store_true",
                   default=CFG["USE_NORMALIZED_DIMS_FIRST"])
    p.add_argument("--no-use-normalized-dims-first", action="store_false",
                   dest="use_normalized_dims_first")
    p.add_argument("--dim-min-in", type=float, default=CFG["DIM_MIN_IN"])
    p.add_argument("--dim-max-in", type=float, default=CFG["DIM_MAX_IN"])

    p.add_argument("--use-normalized-weight-first", action="store_true",
                   default=CFG["USE_NORMALIZED_WEIGHT_FIRST"])
    p.add_argument("--no-use-normalized-weight-first", action="store_false",
                   dest="use_normalized_weight_first")
    p.add_argument("--weight-min-lb", type=float, default=CFG["WEIGHT_MIN_LB"])
    p.add_argument("--weight-max-lb", type=float, default=CFG["WEIGHT_MAX_LB"])

    p.add_argument("--enable-tag-text-clean", action="store_true",
                   default=CFG["ENABLE_TAG_TEXT_CLEAN"])
    p.add_argument("--disable-tag-text-clean", action="store_false",
                   dest="enable_tag_text_clean")

    p.add_argument("--prefer-record-with-images", action="store_true",
                   default=CFG["PREFER_RECORD_WITH_IMAGES"])
    p.add_argument("--no-prefer-record-with-images", action="store_false",
                   dest="prefer_record_with_images")

    p.add_argument("--lang-pref", type=str, default=",".join(CFG["LANG_PREF"]),
                   help='Comma list of language prefs (e.g., "en_US,en_GB,*")')

    # EDA controls
    p.add_argument("--do-eda", action="store_true",
                   help="Run a lightweight EDA pass and save a JSON report.")
    p.add_argument("--eda-max-rows", type=int, default=300_000,
                   help="Rows to sample for EDA (streamed).")
    p.add_argument("--write-plots", action="store_true",
                   help="Also write histogram PNGs under <proc_dir>/eda/figs")
    p.add_argument("--row-limit", type=int, default=None,
                   help="Optional limit for normalization pass (dev/testing).")
    return p.parse_args()


# -----------------------------------------------------------------------------
# 2) Helpers (text cleanup, language picking, numeric guards)
# -----------------------------------------------------------------------------
def _clean(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def _pick_lang_value(items: Any, prefs: List[str]) -> str:
    """Pick first match in LANG_PREF; '*' means any non-empty fallback."""
    if not isinstance(items, list) or not items:
        return ""
    for pref in prefs:
        if pref == "*":
            break
        for it in items:
            if it.get("language_tag") == pref and it.get("value"):
                return str(it["value"])
    for it in items:
        if it.get("value"):
            return str(it["value"])
    return ""

def _join_lang_values(items: Any, prefs: List[str], sep: str = "; ") -> str:
    """Join bullets in preferred language; fallback to any language."""
    if not isinstance(items, list) or not items:
        return ""
    vals: List[str] = []
    for pref in prefs:
        if pref == "*":
            continue
        vs = [it.get("value") for it in items if it.get("language_tag") == pref and it.get("value")]
        if vs:
            vals = vs
            break
    if not vals:
        vals = [it.get("value") for it in items if it.get("value")]
    return sep.join(_clean(v) for v in vals if isinstance(v, str))

def _get_dimension_in_inches(dims: dict, key: str,
                             use_norm_first: bool,
                             vmin: float, vmax: float) -> Optional[float]:
    """
    Pull normalized_value.value vs value per policy; apply guards.
    Returns float or None.
    """
    if not isinstance(dims, dict):
        return None
    node = dims.get(key) or {}
    nval = (node.get("normalized_value") or {}).get("value")
    rval = node.get("value")
    candidates = (nval, rval) if use_norm_first else (rval, nval)
    for cand in candidates:
        if cand is None:
            continue
        try:
            x = float(cand)
        except Exception:
            continue
        if x <= vmin or x > vmax:
            return None
        return x
    return None

def _get_weight_lb(item_weight: Any,
                   use_norm_first: bool,
                   vmin: float, vmax: float) -> Optional[float]:
    """Prefer normalized_value.value vs value per policy; apply guards."""
    if isinstance(item_weight, list) and item_weight:
        w0 = item_weight[0]
        nval = (w0.get("normalized_value") or {}).get("value")
        rval = w0.get("value")
        candidates = (nval, rval) if use_norm_first else (rval, nval)
        for cand in candidates:
            if cand is None:
                continue
            try:
                x = float(cand)
            except Exception:
                continue
            if x <= vmin or x > vmax:
                return None
            return x
    return None

def _extract_color(rec_color: Any, prefs: List[str]) -> str:
    """
    Prefer standardized_values of preferred language entry, else value; fallback first record.
    """
    if not isinstance(rec_color, list) or not rec_color:
        return ""
    for pref in prefs:
        for c in rec_color:
            if c.get("language_tag") == pref:
                std = c.get("standardized_values") or []
                if std:
                    return _clean(std[0])
                return _clean(c.get("value", ""))
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
    """For fields like model_number: [{value: '...'}] -> '...' """
    if isinstance(items, list) and items:
        return _clean(str(items[0].get(key, "")))
    return ""

_TAG_RX = re.compile(r"[^a-z0-9 ]+", flags=re.IGNORECASE)

def _maybe_clean_tag(s: str, enable_clean: bool) -> str:
    if not isinstance(s, str):
        return ""
    s = _clean(s)
    if not enable_clean:
        return s
    s = s.lower()
    s = _TAG_RX.sub(" ", s)
    return " ".join(t for t in s.split() if len(t) > 2)


# -----------------------------------------------------------------------------
# 3) I/O: streaming raw JSONL
# -----------------------------------------------------------------------------
def listings_file_list(raw_dir: Path) -> List[str]:
    patterns = [
        str(raw_dir / "listings_*.json.gz"),
        str(raw_dir / "listings_*.json"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print(f"[WARN] No listings files found under {raw_dir}.")
    return files

def stream_json_lines(raw_dir: Path, max_rows: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Stream records from listings shards (.json or .json.gz) as dicts."""
    seen = 0
    for fp in listings_file_list(raw_dir):
        opener = gzip.open if fp.endswith(".gz") else open
        with opener(fp, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield rec
                seen += 1
                if max_rows is not None and seen >= max_rows:
                    return


# -----------------------------------------------------------------------------
# 4) EDA (lightweight; streamed)
# -----------------------------------------------------------------------------
def run_eda(raw_dir: Path, proc_dir: Path, prefs: List[str],
            write_plots: bool = False, max_rows: int = 300_000) -> Path:
    import numpy as np
    from collections import Counter, defaultdict

    eda_dir = proc_dir / "eda"
    fig_dir = eda_dir / "figs"
    eda_dir.mkdir(parents=True, exist_ok=True)
    if write_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)

    lang_counts = Counter()   # language used for title/brand
    bp_lang_counts = Counter()  # languages present in bullet points
    norm_hit = defaultdict(int)
    norm_total = defaultdict(int)

    rows = []
    for rec in tqdm(stream_json_lines(raw_dir, max_rows=max_rows), total=max_rows, desc="[EDA] streaming"):
        pid = rec.get("item_id") or rec.get("asin") or rec.get("id")
        title = _pick_lang_value(rec.get("item_name"), prefs)
        brand = _pick_lang_value(rec.get("brand"), prefs)

        if title: lang_counts["title"] += 1
        if brand: lang_counts["brand"] += 1

        bps = rec.get("bullet_point")
        if isinstance(bps, list):
            for it in bps:
                if it.get("language_tag"):
                    bp_lang_counts[it["language_tag"]] += 1

        dims = rec.get("item_dimensions") or {}
        def _get_pair(dkey):
            node = dims.get(dkey) or {}
            nv = (node.get("normalized_value") or {}).get("value")
            v  = node.get("value")
            return nv, v

        for k in ["width", "length", "height"]:
            nv, v = _get_pair(k)
            if nv is not None: norm_hit[k]+=1
            if nv is not None or v is not None: norm_total[k]+=1

        wt = _get_weight_lb(rec.get("item_weight"), use_norm_first=True, vmin=0.0, vmax=10_000.0)

        node = rec.get("node") or []
        node_id = node[0].get("node_id") if (isinstance(node, list) and node) else None
        node_name = node[0].get("node_name") if (isinstance(node, list) and node) else None

        main_image_id = rec.get("main_image_id")
        other_image_id = rec.get("other_image_id") if isinstance(rec.get("other_image_id"), list) else []
        num_imgs = int(bool(main_image_id)) + sum(1 for x in other_image_id if x)

        rows.append({
            "product_id": pid,
            "node_id": node_id,
            "node_path": node_name,
            "weight_lb": wt,
            "num_imgs": num_imgs,
        })

    eda_df = pd.DataFrame(rows)

    # Flags
    flags = {}
    for col in ["weight_lb", "num_imgs"]:
        v = pd.to_numeric(eda_df[col], errors="coerce")
        if col == "weight_lb":
            flags["weight_null_pct"] = float(v.isna().mean()*100)
            flags["weight_>200_pct"] = float((v>200).mean()*100)
            flags["weight_<=0_pct"] = float(((v<=0).fillna(False)).mean()*100)

    for k in ["width","length","height"]:
        tot = norm_total[k]
        hit = norm_hit[k]
        flags[f"normalized_{k}_coverage_pct"] = float((hit / tot * 100) if tot else 0.0)

    # Save JSON report
    report = {
        "counts": {
            "rows_sampled": len(eda_df),
            "title_brand_nonempty": dict(lang_counts),
            "bullet_point_languages_top": dict(bp_lang_counts.most_common(10)),
        },
        "flags": flags,
    }
    out_json = eda_dir / "eda_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Optional plots
    if write_plots:
        import matplotlib.pyplot as plt
        def hist(col, bins=60, rng=None, title=None):
            x = pd.to_numeric(eda_df[col], errors="coerce").dropna()
            if not len(x):
                return
            plt.figure()
            plt.hist(x if rng is None else x[(x>=rng[0]) & (x<=rng[1])], bins=bins)
            plt.title(title or col)
            plt.savefig(fig_dir / f"hist_{col}.png", bbox_inches="tight")
            plt.close()

        hist("weight_lb", bins=80, title="weight_lb (all)")
        hist("weight_lb", bins=80, rng=(0, 50), title="weight_lb (0-50)")
        hist("num_imgs", bins=20, title="num_imgs")

    print(f"[EDA] Wrote report -> {out_json}")
    return out_json


# -----------------------------------------------------------------------------
# 5) Normalization
# -----------------------------------------------------------------------------
def normalize_abo(rec: Dict[str, Any],
                  prefs: List[str],
                  use_norm_dims_first: bool,
                  dim_min: float, dim_max: float,
                  use_norm_wt_first: bool,
                  wt_min: float, wt_max: float,
                  enable_tag_clean: bool) -> Dict[str, Any]:
    """Normalize an ABO record into product-level fields."""

    product_id   = rec.get("item_id") or rec.get("asin") or rec.get("id")
    title        = _clean(_pick_lang_value(rec.get("item_name"), prefs))
    brand        = _clean(_pick_lang_value(rec.get("brand"), prefs))
    description  = _clean(_join_lang_values(rec.get("bullet_point"), prefs))

    node_list    = rec.get("node") or []
    node_id      = node_list[0].get("node_id") if (isinstance(node_list, list) and node_list) else None
    node_path    = _clean(node_list[0].get("node_name", "")) if (isinstance(node_list, list) and node_list) else ""
    category     = node_path

    color        = _extract_color(rec.get("color"), prefs)
    color_code   = _extract_color_code(rec.get("color_code"))
    material     = _clean(_pick_lang_value(rec.get("material"), prefs))
    country      = _clean(rec.get("country", ""))
    marketplace  = _clean(rec.get("marketplace", ""))
    domain_name  = _clean(rec.get("domain_name", ""))

    dims         = rec.get("item_dimensions") or {}
    width_in     = _get_dimension_in_inches(dims, "width",  use_norm_dims_first, dim_min, dim_max)
    length_in    = _get_dimension_in_inches(dims, "length", use_norm_dims_first, dim_min, dim_max)
    height_in    = _get_dimension_in_inches(dims, "height", use_norm_dims_first, dim_min, dim_max)

    weight_lb    = _get_weight_lb(rec.get("item_weight"), use_norm_wt_first, wt_min, wt_max)

    main_image_id = rec.get("main_image_id")
    other_image_id = rec.get("other_image_id") or []
    image_ids = [i for i in ([main_image_id] if main_image_id else []) +
                 (other_image_id if isinstance(other_image_id, list) else []) if i]

    product_type = _first_or_empty(rec.get("product_type"), "value")
    model_number = _first_or_empty(rec.get("model_number"), "value")

    # Short meta tags for retrieval/search
    tag_parts = [product_type, material, color, brand]
    tags = ", ".join(t for t in (_maybe_clean_tag(x, enable_tag_clean) for x in tag_parts) if t)

    return {
        "product_id": product_id,
        "title": title,
        "brand": brand,
        "category": category,
        "description": description,
        "tags": tags,
        "image_url": "",   # will be filled after image join (local path)
        "url": "",         # ABO doesn’t ship canonical URLs
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

def normalize_router(rec: Dict[str, Any], **kw) -> Dict[str, Any]:
    """Route to ABO normalizer if the record looks like ABO; else minimal generic."""
    if any(k in rec for k in ("item_id", "item_name", "node", "bullet_point")):
        return normalize_abo(rec, **kw)

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
        "width_in": None, "length_in": None, "height_in": None, "weight_lb": None,
        "marketplace": _clean(rec.get("marketplace", "")),
        "domain_name": _clean(rec.get("domain_name", "")),
        "node_id": None,
        "node_path": cat_str,
        "product_type": _clean(rec.get("product_type", "")),
        "model_number": _clean(rec.get("model_number", "")),
        "main_image_id": None,
        "image_ids": [],
    }


# -----------------------------------------------------------------------------
# 6) Main pipeline
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # Effective config after CLI overrides
    RAW_DIR: Path = args.raw_dir
    PROC_DIR: Path = args.proc_dir
    LANG_PREF: List[str] = [x.strip() for x in args.lang_pref.split(",") if x.strip()]
    USE_NORMALIZED_DIMS_FIRST = args.use_normalized_dims_first
    USE_NORMALIZED_WEIGHT_FIRST = args.use_normalized_weight_first
    DIM_MIN_IN, DIM_MAX_IN = args.dim_min_in, args.dim_max_in
    WEIGHT_MIN_LB, WEIGHT_MAX_LB = args.weight_min_lb, args.weight_max_lb
    ENABLE_TAG_TEXT_CLEAN = args.enable_tag_text_clean
    PREFER_RECORD_WITH_IMAGES = args.prefer_record_with_images

    # Image metadata default layout (can be overridden with --images-root)
    images_root = (args.images_root or (RAW_DIR / "abo-images-small"))
    IMAGES_META = images_root / "images" / "metadata" / "images.csv.gz"
    IMAGES_ROOT = images_root / "images" / "small"

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("=== CONFIG ===")
    print("RAW_DIR:", RAW_DIR)
    print("PROC_DIR:", PROC_DIR)
    print("IMAGES_META exists:", IMAGES_META.exists())
    print("USE_NORMALIZED_DIMS_FIRST:", USE_NORMALIZED_DIMS_FIRST,
          "DIM_MIN_IN:", DIM_MIN_IN, "DIM_MAX_IN:", DIM_MAX_IN)
    print("USE_NORMALIZED_WEIGHT_FIRST:", USE_NORMALIZED_WEIGHT_FIRST,
          "WEIGHT_MIN_LB:", WEIGHT_MIN_LB, "WEIGHT_MAX_LB:", WEIGHT_MAX_LB)
    print("LANG_PREF:", LANG_PREF)
    print("ENABLE_TAG_TEXT_CLEAN:", ENABLE_TAG_TEXT_CLEAN)
    print("PREFER_RECORD_WITH_IMAGES:", PREFER_RECORD_WITH_IMAGES)

    # ----- Optional EDA pass (writes eda/eda_report.json and optional plots) -----
    if args.do_eda:
        run_eda(RAW_DIR, PROC_DIR, prefs=LANG_PREF,
                write_plots=args.write_plots, max_rows=args.eda_max_rows)

    # ----- Normalize all records -----
    rows: List[Dict[str, Any]] = []
    n_stream = 0
    for rec in tqdm(stream_json_lines(RAW_DIR, max_rows=args.row_limit), desc="Normalizing"):
        rows.append(
            normalize_router(
                rec,
                prefs=LANG_PREF,
                use_norm_dims_first=USE_NORMALIZED_DIMS_FIRST,
                dim_min=DIM_MIN_IN, dim_max=DIM_MAX_IN,
                use_norm_wt_first=USE_NORMALIZED_WEIGHT_FIRST,
                wt_min=WEIGHT_MIN_LB, wt_max=WEIGHT_MAX_LB,
                enable_tag_clean=ENABLE_TAG_TEXT_CLEAN,
            )
        )
        n_stream += 1
    if not rows:
        print("[WARN] No raw records found; writing a tiny demo row.")
        rows = [{
            "product_id": "DEMO-001",
            "title": "Demo Red Cotton T-Shirt",
            "brand": "DemoBrand",
            "category": "Clothing > Men > T-Shirts",
            "description": "A comfy red cotton tee suitable for daily wear.",
            "tags": "red, cotton, tee, men",
            "image_url": "",
            "url": "",
            "color": "Red",
            "color_code": "#FF0000",
            "material": "Cotton",
            "country": "US",
            "width_in": None, "length_in": None, "height_in": None, "weight_lb": None,
            "marketplace": "Demo", "domain_name": "example.com",
            "node_id": None, "node_path": "Clothing/Men/T-Shirts",
            "product_type": "SHIRT", "model_number": "DEMO-TEE-1",
            "main_image_id": None, "image_ids": [],
        }]

    df = pd.DataFrame(rows)

    # ---- Product_id validity + dedupe policy ----
    before = len(df)
    df = df.dropna(subset=["product_id"])
    dropped_null_pid = before - len(df)

    if PREFER_RECORD_WITH_IMAGES and "image_ids" in df.columns:
        df["_has_image_ref"] = df["image_ids"].apply(lambda x: bool(x))
        df = (
            df.sort_values(["_has_image_ref"], ascending=False)
              .drop_duplicates("product_id")
              .drop(columns=["_has_image_ref"])
        )
    else:
        df = df.drop_duplicates("product_id")

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
    final_cols = [c for c in col_order if c in df.columns] + \
                 [c for c in df.columns if c not in col_order and c not in ("image_ids", "main_image_id")]
    df = df[final_cols]

    # ---------- Write outputs ----------
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    products_out = PROC_DIR / "products.parquet"
    df.to_parquet(products_out, index=False)

    if not product_images.empty:
        images_out = PROC_DIR / "product_images.parquet"
        product_images.to_parquet(images_out, index=False)
        print("[OK] Saved:", products_out, "rows=", len(df))
        print("[OK] Saved:", images_out, "rows=", len(product_images))
    else:
        print("[OK] Saved:", products_out, "rows=", len(df))
        print("[INFO] No product_images to save.")

    # Minimal processing report for your write-up
    report = {
        "raw_rows_streamed": n_stream,
        "dropped_null_product_id": int(dropped_null_pid),
        "unique_products": int(len(df)),
        "product_images_rows": int(len(product_images)),
    }
    rpt_path = PROC_DIR / "processing_report.json"
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[OK] Wrote processing report ->", rpt_path)


if __name__ == "__main__":
    main()
