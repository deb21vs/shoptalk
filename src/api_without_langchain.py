from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import CHROMA_DIR, EMBED_MODEL, GEN_MODEL, TOP_K, MAX_GEN_TOKENS

app = FastAPI(title="ShopTalk API", version="0.1.0")

# Load retriever
embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
col = client.get_collection("products")

# Load generator (optional; if not available, we fallback)
try:
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)
    gen_model.eval()
    USE_LOCAL_LLM = True
except Exception:
    tokenizer = None
    gen_model = None
    USE_LOCAL_LLM = False

class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/search")
def search(req: SearchRequest):
    q = req.query.strip()
    k = req.top_k or TOP_K
    qvec = embedder.encode([q], normalize_embeddings=True)
    res = col.query(query_embeddings=qvec.tolist(), n_results=k, include=["metadatas", "documents", "distances", "ids"])

    items = []
    for i in range(len(res["ids"][0])):
        items.append({
            "product_id": res["ids"][0][i],
            "score": float(1.0 - res["distances"][0][i]),
            **res["metadatas"][0][i],
            "snippet": res["documents"][0][i][:240]
        })

    # Build a concise answer using top items
    topn = items[:5]
    rec_lines = [f"- {it['title']} (brand: {it.get('brand','')}) — {it.get('url') or 'no link'}" for it in topn]
    sys_prompt = (
        "You are a concise shopping assistant.\n"
        "Use the retrieved items to answer the user's request.\n"
        "Return a short paragraph followed by 3–5 bullet recommendations.\n"
    )
    context = "\n".join(rec_lines)
    prompt = f"<system>\n{sys_prompt}\n</system>\nUser: {q}\nRetrieved:\n{context}\nAssistant:"

    if USE_LOCAL_LLM:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = gen_model.generate(ids, max_new_tokens=MAX_GEN_TOKENS, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    else:
        text = "Here are some options that match your query:\n" + "\n".join(rec_lines)

    return {"answer": text, "items": items}
