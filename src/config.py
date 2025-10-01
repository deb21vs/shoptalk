from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
CAPTIONS_DIR = DATA_DIR / "captions"
#CHROMA_DIR = ROOT / "index" / "chroma"
CHROMA_DIR = ROOT / "index" / "chroma_bge_large"

# Models (swap as needed)
EMBED_MODEL = "BAAI/bge-base-en-v1.5"         # strong general-purpose embeddings
RERANK_MODEL = "BAAI/bge-reranker-base"       # optional cross-encoder reranker
CAPTION_MODEL = "Salesforce/blip-image-captioning-large"  # optional offline captioning
GEN_MODEL = "meta-llama/Llama-3-8B-Instruct"  # replace with a locally/hosted available model

# Retrieval / Generation params
TOP_K = 20
MAX_GEN_TOKENS = 256

USE_LANGCHAIN = True

# LangChain bits (can reuse the same models)
#LC_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # or your BGE
LC_EMBED_MODEL = "BAAI/bge-large-en-v1.5" 
LC_LLM_NAME    = "gpt-4o-mini"  # or local HF via langchain (optional)
GEN_MODEL = None


# config.py
AZURE_OPENAI_ENDPOINT = "https://dmatchopenai.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4-turbo"   # must match your Azure deployment name
AZURE_OPENAI_API_VERSION = "2024-04-09"   # version you mentioned










