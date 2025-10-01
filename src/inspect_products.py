import pandas as pd
import random
from src.config import PROC_DIR

def main():
    # Load products parquet
    path = PROC_DIR / "products.parquet"
    df = pd.read_parquet(path)

    print(f"[INFO] Loaded {len(df)} products from {path}")
    print(f"[INFO] Columns: {list(df.columns)}\n")

    # Show first 5 rows
    print("=== First 5 Products ===")
    print(df.head(1).to_markdown())

    # Show 5 random rows
    print("\n=== Random 5 Products ===")
    sample = df.sample(n=2, random_state=random.randint(0, 10000))
    print(sample.to_markdown())

    # Show some stats
    print("\n=== Quick Stats ===")
    print(df[["brand", "category", "color", "material"]].describe(include="all"))

if __name__ == "__main__":
    main()
