import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import chromadb

# load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# DB_DIR   = Path(__file__).resolve().parents[1] / "storage" / "chroma"
DB_DIR   = Path(__file__).resolve().parents[1] / "storage" / "chroma_ollama"
DB_DIR.mkdir(parents=True, exist_ok=True)

def pdf_to_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        # Extract page text; strip to keep it clean
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "content": text,
                "page": i,
                "source": pdf_path.name,
            })
    return pages

def chunk_pages(pages, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for p in pages:
        for chunk in splitter.split_text(p["content"]):
            chunks.append({
                "content": chunk,
                "page": p["page"],
                "source": p["source"],
            })
    return chunks

def main():
    # 1) Gather all pages from all PDFs
    all_chunks = []
    for pdf in DATA_DIR.glob("*.pdf"):
        pages = pdf_to_pages(pdf)
        if not pages:
            print(f"[WARN] No text extracted from {pdf.name} (scanned/OCR needed?)")
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks to index. Exiting.")
        return

    # 2) Initialize Chroma
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name="seismic_docs",
        metadata={"hnsw:space": "cosine"},
    )

    # 3) Embed + add (in small batches)
    # embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    batch_size = 128
    docs = [c["content"] for c in all_chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in all_chunks]
    ids = [f"{m['source']}|{m['page']}|{i}" for i, m in enumerate(metadatas)]

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids  = ids[i:i+batch_size]
        # Compute embeddings once per batch
        vectors = embeddings.embed_documents(batch_docs)
        collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids, embeddings=vectors)
        print(f"Indexed {i + len(batch_docs)} / {len(docs)}")

    print("✅ Index built successfully at storage/chroma")

if __name__ == "__main__":
    main()
