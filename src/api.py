from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Optional, List, Dict, Any, Tuple
from uuid import uuid4

# --- Config ---
from src.config import CHROMA_DIR, TOP_K, MAX_GEN_TOKENS
try:
    from src.config import LC_EMBED_MODEL as EMBED_NAME
except Exception:
    try:
        from src.config import EMBED_MODEL as EMBED_NAME
    except Exception:
        EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="ShopTalk API (Azure OpenAI, chain-free)", version="0.6.1")

# --- Embeddings + VectorStore ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

emb = HuggingFaceEmbeddings(
    model_name=EMBED_NAME,
    encode_kwargs={"normalize_embeddings": True},  # must match what you used when indexing
)

vectordb = Chroma(
    collection_name="products",                 # must match your build_index.py
    persist_directory=str(CHROMA_DIR),         # e.g., "index/chroma_bge_large"
    embedding_function=emb,
)

retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

# --- Azure OpenAI (optional) ---
from langchain_openai import AzureChatOpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")   # e.g. https://<name>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-04-09")
USE_AZURE = bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.2,
    max_tokens=MAX_GEN_TOKENS,
    timeout=30,
) if USE_AZURE else None

# --- In-memory chat history ---
CHAT_HISTORY: dict[str, List[Tuple[str, str]]] = {}

# --- Schemas ---
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None   # <= NEW (metadata filter)

class ChatStartResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    query: str
    top_k: Optional[int] = None

class ResetRequest(BaseModel):
    session_id: str

# --- Utils ---
def _docs_to_items(docs_with_scores) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc, score in docs_with_scores:
        md = doc.metadata or {}
        out.append({
            "product_id": md.get("product_id"),
            "score": float(score) if score is not None else None,
            "title": md.get("title"),
            "brand": md.get("brand"),
            "category": md.get("category"),
            "url": md.get("url"),
            "image_url": md.get("image_url"),
            "snippet": (doc.page_content or "")[:240],
        })
    return out

def _format_recs(items: List[Dict[str, Any]]) -> str:
    lines = []
    for it in items[:5]:
        title = it.get("title", "")
        brand = it.get("brand", "")
        link = it.get("url") or "no link"
        lines.append(f"- {title} (brand: {brand}) — {link}")
    return "\n".join(lines)

SYSTEM_PROMPT = (
    "You are a concise shopping assistant. "
    "Use the retrieved items to answer the user's request. "
    "Return a short helpful paragraph followed by 3–5 bullet recommendations. "
    "Do not invent products; rely on retrieved context."
)

def _chat_prompt(history: List[Tuple[str, str]], user_q: str, items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history[-8:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    context = _format_recs(items)
    messages.append({"role": "user", "content": f"{user_q}\n\nRecommendations:\n{context}"})
    return messages

def _one_shot_prompt(user_q: str, items: List[Dict[str, Any]]) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser: {user_q}\n\nRecommendations:\n{_format_recs(items)}"

# --- Routes ---
@app.get("/")
def root():
    return {"ok": True, "message": "See /docs. Endpoints: /health, /search, /chat/start, /chat, /chat/reset"}

@app.get("/health")
def health():
    try:
        # Chroma langchain wrapper doesn’t expose count directly; do a tiny check:
        test = retriever.search_kwargs.get("k", 1)
        return {
            "ok": True,
            "chroma_dir": str(CHROMA_DIR),
            "collection": "products",
            "embed_model": EMBED_NAME,
            "top_k_default": TOP_K,
            "azure_enabled": USE_AZURE,
            "test_k": test
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {e}")

@app.post("/search")
def search(req: SearchRequest):
    q = (req.query or "").strip()
    if not q:
        return {"answer": "Please provide a query.", "items": []}
    k = req.top_k or TOP_K
    # Optional metadata filter
    if req.filters:
        docs_with_scores = vectordb.similarity_search_with_relevance_scores(q, k=k, filter=req.filters)
    else:
        docs_with_scores = vectordb.similarity_search_with_relevance_scores(q, k=k)
    items = _docs_to_items(docs_with_scores)

    if llm is not None:
        try:
            answer = llm.invoke(_one_shot_prompt(q, items))
            return {"answer": str(answer).strip(), "items": items}
        except Exception as e:
            print(f"[WARN] Azure OpenAI call failed: {e}")

    return {"answer": "Here are some options that match your query:\n" + _format_recs(items), "items": items}

@app.post("/chat/start", response_model=ChatStartResponse)
def chat_start():
    sid = uuid4().hex
    CHAT_HISTORY[sid] = []
    return ChatStartResponse(session_id=sid)

@app.post("/chat")
def chat(req: ChatRequest):
    sid = (req.session_id or "").strip()
    if sid not in CHAT_HISTORY:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Call /chat/start first.")
    q = (req.query or "").strip()
    if not q:
        return {"answer": "Please provide a query.", "items": [], "session_id": sid}
    k = req.top_k or TOP_K

    docs_with_scores = vectordb.similarity_search_with_relevance_scores(q, k=k)
    items = _docs_to_items(docs_with_scores)

    if llm is not None:
        try:
            history = CHAT_HISTORY[sid]
            messages = _chat_prompt(history, q, items)
            answer = llm.invoke(messages)
            text = str(answer).strip()
            history.append((q, text))
            CHAT_HISTORY[sid] = history
            return {"answer": text, "items": items, "session_id": sid}
        except Exception as e:
            print(f"[WARN] Azure OpenAI call failed: {e}")

    text = "Here are some options that match your request:\n" + _format_recs(items)
    CHAT_HISTORY[sid].append((q, text))
    return {"answer": text, "items": items, "session_id": sid}

@app.post("/chat/reset")
def chat_reset(req: ResetRequest):
    sid = (req.session_id or "").strip()
    if sid in CHAT_HISTORY:
        CHAT_HISTORY[sid] = []
        return {"ok": True, "session_id": sid}
    raise HTTPException(status_code=400, detail="Invalid session_id.")
