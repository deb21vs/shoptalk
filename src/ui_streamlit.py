import os
import time
import requests
import streamlit as st

# -----------------------------
# Config
# -----------------------------
DEFAULT_API_BASE = os.getenv("SHOP_TALK_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="ShopTalk", page_icon="🛒", layout="wide")

if "api_base" not in st.session_state:
    st.session_state.api_base = DEFAULT_API_BASE
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Helpers
# -----------------------------
def api_get(path: str):
    url = st.session_state.api_base.rstrip("/") + path
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def api_post(path: str, payload: dict):
    url = st.session_state.api_base.rstrip("/") + path
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def render_items(items):
    if not items:
        st.info("No results.")
        return
    for it in items:
        with st.container():
            cols = st.columns([7, 3, 2])
            title = it.get("title") or "(no title)"
            brand = it.get("brand") or ""
            cat = it.get("category") or ""
            score = it.get("score")
            url = it.get("url")
            img = it.get("image_url")

            with cols[0]:
                st.markdown(f"**{title}**")
                sub = f"{brand} · {cat}"
                if sub.strip(" ·"):
                    st.caption(sub)
                if url:
                    st.markdown(f"[Open product]({url})")
            with cols[1]:
                if score is not None:
                    st.metric("Relevance", f"{score:.3f}")
            with cols[2]:
                if img:
                    st.image(img, use_column_width=True)
            st.divider()

def ensure_chat_session():
    """Create a chat session if we don't have one."""
    if not st.session_state.chat_session_id:
        try:
            data = api_post("/chat/start", {})
            st.session_state.chat_session_id = data["session_id"]
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"Failed to start chat session: {e}")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    st.text_input("API Base URL", value=st.session_state.api_base, key="api_base")
    if st.button("Ping /health"):
        try:
            health = api_get("/health")
            st.success("Server OK")
            st.json(health)
        except Exception as e:
            st.error(f"Health check failed: {e}")

    st.markdown("---")
    st.caption("Filters apply to **Search**. Chat uses raw query + retrieval.")
    default_cat = st.text_input("Category contains (optional)", value="", key="flt_category")
    default_brand = st.text_input("Brand contains (optional)", value="", key="flt_brand")
    default_color = st.text_input("Color contains (optional)", value="", key="flt_color")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🔎 Search", "💬 Chat"])

with tab1:
    st.subheader("Semantic Product Search")
    q = st.text_input("Query", value="red men t-shirt", key="search_query")
    top_k = st.slider("Top K", 1, 50, 10, key="search_topk")

    # Build filters dict only with non-empty fields
    filters = {}
    if st.session_state.flt_category.strip():
        # Prefer server-side filter if you indexed a lowercased field (category_lc)
        # Otherwise, your API will handle post-filtering fallback
        # filters = {"category_lc": {"$contains": st.session_state.flt_category.lower()}}
        filters = {"category": {"$contains": st.session_state.flt_category}}  # works if your API passes through
    if st.session_state.flt_brand.strip():
        filters["brand"] = {"$contains": st.session_state.flt_brand}
    if st.session_state.flt_color.strip():
        # If you stored color in metadata (you did), this helps
        filters["color"] = {"$contains": st.session_state.flt_color}

    col_a, col_b = st.columns([1, 5])
    with col_a:
        run = st.button("Search")
    with col_b:
        if filters:
            st.caption(f"Filters: {filters}")

    if run:
        payload = {"query": q, "top_k": top_k, "filters": filters or None}
        try:
            resp = api_post("/search", payload)
            st.write(resp.get("answer", ""))
            render_items(resp.get("items", []))
        except Exception as e:
            st.error(f"Search failed: {e}")

with tab2:
    st.subheader("Conversational Shopping Assistant")
    ensure_chat_session()
    if st.session_state.chat_session_id:
        st.caption(f"Session: `{st.session_state.chat_session_id}`")

    # chat input at the bottom (like common chat UX)
    chat_container = st.container()
    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Type your message", value="Show me red men t-shirts", key="chat_input")
        ask = st.form_submit_button("Ask")

    if ask and user_msg.strip():
        try:
            payload = {
                "session_id": st.session_state.chat_session_id,
                "query": user_msg,
                "top_k": 10,
            }
            resp = api_post("/chat", payload)
            # update local visible history
            st.session_state.chat_history.append(("user", user_msg))
            st.session_state.chat_history.append(("assistant", resp.get("answer", "")))
            # show answer + cards
            with chat_container:
                for role, text in st.session_state.chat_history[-12:]:
                    if role == "user":
                        st.markdown(f"**You:** {text}")
                    else:
                        st.markdown(f"**Assistant:** {text}")
                st.markdown("---")
                render_items(resp.get("items", []))
        except Exception as e:
            st.error(f"Chat failed: {e}")

    cols = st.columns(3)
    with cols[0]:
        if st.button("Reset Chat"):
            try:
                if st.session_state.chat_session_id:
                    api_post("/chat/reset", {"session_id": st.session_state.chat_session_id})
                st.session_state.chat_session_id = None
                st.session_state.chat_history = []
                ensure_chat_session()
                st.success("Chat reset.")
            except Exception as e:
                st.error(f"Reset failed: {e}")
    with cols[1]:
        st.caption("Tip: filters are applied on the Search tab.")
    with cols[2]:
        st.caption("LLM answer depends on Azure OpenAI env vars.")
