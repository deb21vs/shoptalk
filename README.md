# ShopTalk — AI Shopping Assistant (Starter)

This is a runnable scaffold for a RAG-based shopping assistant.

## Quickstart

```bash
conda create -n shoptalk python=3.10 -y
conda activate shoptalk
pip install -r requirements.txt

# 1) Preprocess (uses a tiny demo row if `data/raw/*.jsonl` is missing)
python src/preprocess_products.py

# 2) (Optional) Caption images
# python src/caption_images.py

# 3) Build vector index
python src/build_index.py

# 4) Start API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 5) Run UI (new terminal)
streamlit run src/ui_streamlit.py --server.port 8501
```

## Notes
- Replace the demo product with ABO subset by dropping JSONL files into `data/raw/`.
- For generation, swap `GEN_MODEL` in `src/config.py` to a model you can serve locally or via a hosted API.
- The API falls back to a template response if it cannot load a local LLM.

Generated at 2025-08-21T00:43:26.880413Z.
