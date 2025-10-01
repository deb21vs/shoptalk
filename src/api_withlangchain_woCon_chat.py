from __future__ import annotations
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

# LangChain vector store + embeddings
from langchain_community.vectorstores import Chroma
# Prefer the new package for embeddings to avoid deprecation warnings:
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    # fallback if you didn't install `langchain-huggingface`
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

# Optional: local HF LLM wrapped as a LangChain LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

import torch

from config import (
    CHROMA_DIR,
    TOP_K,
    MAX_GEN_TOKENS,
)

# Optional model names in config; provide safe fallbacks if not set
try:
    from config import LC_EMBED_MODEL as EMBED_NAME
except Exception:
    try:
        from config import EMBED_MODEL as EMBED_NAME  # old name
    except Exception:
        EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"

try:
    from config import GEN_MODEL as GEN_NAME
except Exception:
    GEN_NAME = None

app = FastAPI(title="ShopTalk API (LangChain)", version="0.2.0")

# -----------------------------
# Embeddings + VectorStore
# -----------------------------
emb = HuggingFaceEmbeddings(
    model_name=EMBED_NAME,
    encode_kwargs={"normalize_embeddings": True},
)
vectordb = Chroma(
    collection_name="products",
    persist_directory=str(CHROMA_DIR),
    embedding_function=emb,
)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

# -----------------------------
# Optional local generator
# -----------------------------
USE_LOCAL_LLM = False
llm = None
if GEN_NAME:
    try:
        tok = AutoTokenizer.from_pretrained(GEN_NAME)
        mdl = AutoModelForCausalLM.from_pretrained(
            GEN_NAME,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        text_gen = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            max_new_tokens=MAX_GEN_TOKENS,
            do_sample=False,
        )
        llm = HuggingFacePipeline(pipeline=text_gen)
        USE_LOCAL_LLM = True
        print(f"[INFO] Loaded local generator: {GEN_NAME}")
    except Exception as e:
        print(f"[WARN] Could not load local generator ({GEN_NAME}). Falling back. Err: {e}")
        USE_LOCAL_LLM = False
        llm = None

# -----------------------------
# Schemas
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

# -----------------------------
# Helpers
# -----------------------------
def _format_recommendations(items: List[Dict[str, Any]]) -> str:
    lines = []
    for it in items:
        title = it.get("title", "")
        brand = it.get("brand", "")
        link = it.get("url") or "no link"
        lines.append(f"- {title} (brand: {brand}) — {link}")
    return "\n".join(lines)

def _generate_answer(query: str, items: List[Dict[str, Any]]) -> str:
    """Use local LLM if available; otherwise return a concise templated answer."""
    topn = items[:5]
    rec_lines = _format_recommendations(topn)

    sys_prompt = (
        "You are a concise shopping assistant.\n"
        "Use the retrieved items to answer the user's request.\n"
        "Return a short paragraph followed by 3–5 bullet recommendations."
    )
    prompt = f"{sys_prompt}\n\nUser: {query}\n\nRecommendations:\n{rec_lines}\n\nAssistant:"

    if USE_LOCAL_LLM and llm is not None:
        try:
            text = llm.invoke(prompt)
            return str(text).strip()
        except Exception as e:
            print(f"[WARN] LLM generation failed, falling back. Err: {e}")

    # Templated fallback
    return f"Here are some options that match your query:\n{rec_lines}"

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/search")
def search(req: SearchRequest):
    q = (req.query or "").strip()
    if not q:
        return {"answer": "Please provide a query.", "items": []}

    k = req.top_k or TOP_K

    # Use LangChain vectorstore directly for similarity search + scores
    # (This returns (Document, score) pairs where score is typically cosine similarity-like)
    docs_with_scores = vectordb.similarity_search_with_relevance_scores(q, k=k)

    items: List[Dict[str, Any]] = []
    for doc, score in docs_with_scores:
        md = doc.metadata or {}
        items.append({
            "product_id": md.get("product_id"),
            "score": float(score) if score is not None else None,
            "title": md.get("title"),
            "brand": md.get("brand"),
            "category": md.get("category"),
            "url": md.get("url"),
            "image_url": md.get("image_url"),
            "snippet": (doc.page_content or "")[:240],
        })

    answer = _generate_answer(q, items)
    return {"answer": answer, "items": items}
