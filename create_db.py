"""
create_db.py  –  Disaster RAG Bot  |  Knowledge Base Builder
Loads all PDFs from the 'data/' folder, chunks them, embeds them with
HuggingFace sentence-transformers, and saves a FAISS index locally.
Run this ONCE before launching the Streamlit app.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH   = "data"
INDEX_PATH  = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE  = 800
CHUNK_OVERLAP = 100

def build_knowledge_base():
    print("=" * 55)
    print("  Disaster RAG Bot  –  Knowledge Base Builder")
    print("=" * 55)

    # ── 1. Load PDFs ───────────────────────────────────────────
    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{DATA_PATH}/' directory.")
        return

    for pdf_file in pdf_files:
        path = os.path.join(DATA_PATH, pdf_file)
        print(f"  Loading: {pdf_file}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    print(f"\n✅ Total pages loaded : {len(documents)}")

    # ── 2. Split into chunks ───────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Total chunks created: {len(chunks)}")

    # ── 3. Create embeddings ───────────────────────────────────
    print(f"\n🔄 Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── 4. Build & save FAISS vectorstore ─────────────────────
    print("🔄 Building FAISS vector store …")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print(f"\n✅ FAISS knowledge base saved to '{INDEX_PATH}/'")
    print("   You can now launch the app with:  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    build_knowledge_base()