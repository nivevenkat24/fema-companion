import os
from pathlib import Path
import streamlit as st
# from dotenv import load_dotenv
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# load_dotenv()

# DB_DIR = Path(__file__).resolve().parents[1] / "storage" / "chroma"
DB_DIR = Path(__file__).resolve().parents[1] / "storage" / "chroma_ollama"

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(name="seismic_docs", metadata={"hnsw:space": "cosine"})

@st.cache_resource
def get_embedder():
    # return OpenAIEmbeddings()
    return OllamaEmbeddings(model="nomic-embed-text")

def retrieve(query, k=6):
    coll = get_collection()
    embedder = get_embedder()
    qvec = embedder.embed_query(query)
    res = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    # Chroma returns lists-of-lists
    docs      = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    return list(zip(docs, metadatas, distances))

def format_citations(hits):
    # Deduplicate citations like (FEMA_P750.pdf p. 3)
    seen = set()
    cites = []
    for _, meta, _ in hits:
        tag = f"{meta['source']} p.{meta['page']}"
        if tag not in seen:
            cites.append(tag)
            seen.add(tag)
    return cites

def build_prompt(user_q, hits):
    context_blocks = []
    for i, (doc, meta, dist) in enumerate(hits, start=1):
        context_blocks.append(
            f"[{i}] Source: {meta['source']} (p.{meta['page']})\n{doc}\n"
        )
    context_text = "\n".join(context_blocks)

    system = (
        "You are a seismic design assistant for engineers. "
        "Answer concisely in bullet points when appropriate. "
        "Only use the supplied context. If unsure, say you don't know. "
        "Always include inline citations like [fema_2020-nehrp-provisions_part-1-and-part-2.pdf p.12] "
        "directly after the relevant sentence."
    )

    user = (
        f"User question:\n{user_q}\n\n"
        f"Context (excerpts from FEMA/NIST):\n{context_text}\n\n"
        "Write the best possible answer using only the context above."
    )

    return system, user

def answer(query):
    hits = retrieve(query, k=6)
    if not hits:
        return "I couldn't find anything relevant in the documents.", [], []

    system, user = build_prompt(query, hits)

    # llm = ChatOpenAI(temperature=0)  # default model, concise & deterministic
    llm = ChatOllama(model="llama3", temperature=0)
    resp = llm.invoke([{"role": "system", "content": system},
                       {"role": "user", "content": user}])

    citations = format_citations(hits)
    return resp.content, hits, citations

# ---- UI ----
st.set_page_config(page_title="Seismic Design Companion", page_icon="🌐", layout="wide")
st.title("Seismic Design Companion (FEMA P-750 + NIST GCR)")

st.write("Ask seismic design questions in plain English. Responses include citations to doc + page.")

default_examples = [
    "What are the criteria for Seismic Design Category D?",
    "List the response-modification factors for SMRFs.",
    "Summarize load combination guidance in FEMA P-750 Chapter 3.",
]

with st.sidebar:
    st.subheader("Examples")
    for ex in default_examples:
        if st.button(ex):
            st.session_state["q"] = ex
    st.markdown("---")
    st.caption("Docs indexed from: data/*.pdf")

q = st.text_input("Your question", value=st.session_state.get("q", ""))
if st.button("Ask") or (q and "q" not in st.session_state):
    st.session_state["q"] = q
    with st.spinner("Thinking..."):
        ans, hits, cites = answer(q)
    st.subheader("Answer")
    st.write(ans)

    st.subheader("Sources")
    for _, meta, dist in hits[:6]:
        st.markdown(f"- **{meta['source']}**, p. **{meta['page']}**")

    st.caption("Unique citations: " + "; ".join(cites) if cites else "No citations")
