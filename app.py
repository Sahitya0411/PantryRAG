"""PantryRAG chat UI (Streamlit).

    streamlit run app.py
Backend is chosen with PANTRYRAG_LLM (hf | anthropic | extractive | auto).
"""
import os

import streamlit as st

from pantryrag.corpus import PROCESSED
from pantryrag.pipeline import ChatSession, PantryRAG

st.set_page_config(page_title="PantryRAG", page_icon="🥘", layout="wide")


@st.cache_resource(show_spinner="Loading recipe index and models…")
def load_rag(backend: str) -> PantryRAG:
    return PantryRAG(llm=backend)


with st.sidebar:
    st.title("🥘 PantryRAG")
    st.caption("Pantry-based recipe chatbot · hybrid retrieval-augmented generation")
    backends = ["extractive", "hf", "anthropic"]
    default = os.environ.get("PANTRYRAG_LLM", "extractive")
    backend = st.selectbox("Generator", backends,
                           index=backends.index(default) if default in backends else 0,
                           help="extractive = no LLM (instant); hf = local Qwen2.5-1.5B; "
                                "anthropic = Claude (needs ANTHROPIC_API_KEY)")
    src = (PROCESSED / "SOURCE").read_text().strip() if (PROCESSED / "SOURCE").exists() else "?"
    st.caption(f"Corpus: **{src}**")
    if st.button("New conversation"):
        st.session_state.pop("chat", None)
        st.session_state.pop("msgs", None)
    st.markdown("""**Try:**
1. I have paneer, spinach, rice, onion and tomato; vegetarian, under 30 minutes
2. show me the second one
3. what can I use instead of curry powder?
4. make it vegan
5. chicken breast, capsicum, curd, jeera and haldi - indian, no cream

Also: “I also have peas”, “I don’t have spinach”, “no garlic”, “under 20 minutes”""")

rag = load_rag(backend)
if "chat" not in st.session_state or st.session_state.get("backend") != backend:
    st.session_state.chat = ChatSession(rag)
    st.session_state.msgs = []
    st.session_state.backend = backend
chat: ChatSession = st.session_state.chat

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("What's in your pantry?"):
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching recipes…"):
            reply, ans = chat.turn(prompt)
        st.markdown(reply)
    st.session_state.msgs.append({"role": "assistant", "content": reply})

q = chat.state.query
if q is not None:
    with st.sidebar:
        st.divider()
        st.subheader("Understood pantry")
        st.write(", ".join(q.ingredients) or "—")
        if q.unresolved:
            st.caption("Not recognised: " + ", ".join(q.unresolved))
        st.write("**Constraints:** " + q.constraints_text())
    if chat.state.hits:
        with st.expander("Retrieved recipes (evidence)", expanded=False):
            for i, h in enumerate(chat.state.hits[:5], 1):
                r = h.recipe
                ranks = ", ".join(f"{k} #{v}" for k, v in h.ranks.items())
                st.markdown(f"**{i}. {r.title}** `[R{r.id}]` — coverage {h.coverage:.0%}, "
                            f"missing {len(h.missing)} · {r.sub_region or r.region} · "
                            f"{int(r.total_minutes) if r.total_minutes else '?'} min · ranks: {ranks}")
                st.caption("Missing: " + (", ".join(h.missing) or "nothing"))
