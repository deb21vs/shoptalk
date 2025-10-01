python3.10 - <<'PY'
import pandas as pd, pathlib, numpy as np
root = pathlib.Path("/Users/deb/Documents/shoptalk")
prods = pd.read_parquet(root/"data/processed/products.parquet")
imgs  = pd.read_parquet(root/"data/processed/product_images.parquet")
print("[products] rows:", len(prods))
print("non-empty image_url %:", (prods["image_url"].fillna("").str.len()>0).mean()*100)
print(prods.sample(3, random_state=0)[["product_id","title","image_url","num_images"]])
print("\n[product_images] rows:", len(imgs))
print(imgs.sample(5, random_state=0)[["product_id","image_id","image_path","is_main","width","height"]])
PY
