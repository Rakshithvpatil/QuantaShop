"""
rag/build_rag.py
─────────────────
Builds a local RAG (Retrieval-Augmented Generation) pipeline for:
  - Product Q&A (specs, availability, compatibility)
  - Return/shipping policy lookup
  - Wearable device FAQ

Tools:
  - LangChain for orchestration
  - ChromaDB for vector store (persisted locally)
  - Sentence-Transformers for embeddings (free, runs offline)
  - Ollama + Mistral-7B for LLM (free, runs locally)

Setup Ollama FIRST:
  1. Download from https://ollama.com (free)
  2. Run: ollama pull mistral
  3. Ollama runs as a local server on port 11434

Run this script once to build the index:
  python -m rag.build_rag

Then use query() for inference.
"""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

CHROMA_PATH  = "data/chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # ~90MB, free, runs fully offline


# ── Sample product knowledge base ─────────────────────────────
# In production: load from BigCommerce product catalog + policy PDFs
PRODUCT_DOCS = [
    Document(
        page_content="""
        SKU-001: Smart Glasses Gen1
        Price: $299.99
        Description: AI-powered smart glasses with built-in camera, microphone,
        accelerometer, ambient light sensor, and heart rate monitor.
        Battery: Up to 8 hours continuous use. Charges in 90 minutes.
        Connectivity: Bluetooth 5.2, WiFi 6. Pairs with iOS 15+ and Android 12+.
        Water resistance: IPX4 (splash proof — not waterproof).
        Weight: 42g. Available in matte black and silver.
        Firmware: Mentra OS v2.1. OTA updates supported.
        Warranty: 1 year limited. Extended warranty available for $49.99.
        """,
        metadata={"source": "product_catalog", "sku": "SKU-001", "category": "hardware"}
    ),
    Document(
        page_content="""
        SKU-002: Charging Case
        Price: $49.99
        Compatible with: SKU-001 Smart Glasses Gen1.
        Capacity: Provides 3 additional full charges. Total: 32 hours usage.
        Battery indicator: 4-LED charge level display.
        Charges via USB-C (cable included).
        Dimensions: 155mm x 60mm x 45mm. Weight: 180g.
        Material: Recycled polycarbonate shell with soft-touch coating.
        """,
        metadata={"source": "product_catalog", "sku": "SKU-002", "category": "accessory"}
    ),
    Document(
        page_content="""
        SKU-003: Replacement Lens Set
        Price: $29.99
        Compatible with: SKU-001 Smart Glasses Gen1.
        Includes: 2 clear lenses + 1 tinted lens.
        UV protection: UV400. Anti-scratch coating.
        Installation: Tool-free. Snaps into frame in under 60 seconds.
        Prescription inserts available via our optician partner program.
        """,
        metadata={"source": "product_catalog", "sku": "SKU-003", "category": "accessory"}
    ),
    Document(
        page_content="""
        Return Policy:
        - Unopened items: 30-day full refund, no questions asked.
        - Opened items: 15-day return window for defective products only.
        - Proof of purchase required for all returns.
        - Digital products (firmware licenses): non-refundable.
        - Return shipping: Customer pays unless item is defective.
        - Refund processing: 5-7 business days after receipt.
        - International returns: Customer responsible for duties/customs.
        - Contact: support@swiftpulse.com or 1-800-SWIFT-01.
        """,
        metadata={"source": "policy_docs", "type": "returns"}
    ),
    Document(
        page_content="""
        Shipping Policy:
        - Standard shipping (US): Free on orders over $75. $7.99 otherwise.
        - Express 2-day: $19.99. Overnight: $34.99.
        - Ships from Irving, TX warehouse.
        - International shipping: Available to 45+ countries via DHL.
        - Typical international transit: 7-14 business days.
        - Orders placed before 2pm CT ship same day (Mon-Fri).
        - Tracking provided via email for all shipments.
        """,
        metadata={"source": "policy_docs", "type": "shipping"}
    ),
    Document(
        page_content="""
        Smart Glasses FAQ:
        Q: Are the glasses compatible with prescription lenses?
        A: Yes, through our optician partner program. Submit your prescription
           at swiftpulse.com/rx and lenses ship within 10 business days.

        Q: Can I use the glasses while swimming?
        A: No. IPX4 rating means splash and light rain only. Do not submerge.

        Q: How does the heart rate sensor work?
        A: Optical PPG sensor in the nose bridge. Accuracy: ±2 BPM at rest.
           Accuracy decreases during high-intensity activity.

        Q: Does the AI assistant work offline?
        A: Core features (step counting, anomaly alerts) work offline.
           AI chat requires internet connection.

        Q: How do firmware updates work?
        A: Automatic OTA (over-the-air) via WiFi. Updates download overnight.
           Manual update available through the companion app.
        """,
        metadata={"source": "faq", "type": "product_faq"}
    ),
]


def build_index() -> Chroma:
    """
    Build or load ChromaDB vector index from product documents.
    Embeddings are computed locally using Sentence-Transformers.
    """
    print("🔧 Building RAG vector index...")
    print(f"   Embedding model: {EMBED_MODEL} (downloads ~90MB on first run)")

    # Chunking — split large documents into retrieval-friendly pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(PRODUCT_DOCS)
    print(f"   Split {len(PRODUCT_DOCS)} documents → {len(chunks)} chunks")

    # Local embeddings — no API key, no cost, runs on CPU in ~seconds
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Persist to disk — only rebuilds if collection doesn't exist
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="swiftpulse_kb",
    )
    print(f"✅ Index built. {vectorstore._collection.count()} vectors stored at {CHROMA_PATH}")
    return vectorstore


def load_index() -> Chroma:
    """Load existing ChromaDB index from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="swiftpulse_kb",
    )


def query(question: str, top_k: int = 3) -> dict:
    """
    Answer a question using RAG.
    Uses Ollama/Mistral if available, else returns retrieved context only.

    Args:
        question: Natural language question
        top_k: Number of context chunks to retrieve

    Returns:
        dict with 'answer', 'sources', and 'context_chunks'
    """
    # Load vector store
    db_path = Path(CHROMA_PATH)
    vectorstore = load_index() if db_path.exists() else build_index()

    # Retrieve relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(d.page_content for d in docs)
    sources  = list({d.metadata.get("source", "unknown") for d in docs})

    # Try Ollama (local LLM) — gracefully skip if not installed
    answer = None
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="mistral", temperature=0.1)

        prompt = f"""You are a helpful customer support agent for SwiftPulse,
an AI wearable tech and e-commerce company. Answer the question using ONLY
the context below. If the answer isn't in the context, say so clearly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        answer = llm.invoke(prompt).strip()
    except Exception:
        # Ollama not running — return context-only response
        answer = (
            f"[Ollama not running — install from https://ollama.com and run: "
            f"ollama pull mistral]\n\nRelevant information found:\n{context[:500]}..."
        )

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
        "context_chunks": [
            {"text": d.page_content[:200] + "...", "metadata": d.metadata}
            for d in docs
        ],
    }


if __name__ == "__main__":
    # Build index and run a few test queries
    build_index()

    test_questions = [
        "Is the smart glasses waterproof?",
        "What is the return policy for opened items?",
        "How long does shipping take internationally?",
        "Can I use prescription lenses with the glasses?",
    ]

    print("\n" + "="*60)
    print("RAG Query Test")
    print("="*60)
    for q in test_questions:
        result = query(q)
        print(f"\n❓ {q}")
        print(f"📄 Sources: {result['sources']}")
        print(f"💬 Answer: {result['answer'][:300]}")
        print("-"*40)
