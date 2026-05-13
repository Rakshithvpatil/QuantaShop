"""
rag/build_index.py
──────────────────
Builds a ChromaDB vector index from product catalog + policy docs.
Uses a free local embedding model (all-MiniLM-L6-v2).
LLM is served by Ollama locally (free, no API key).

Setup (one-time):
    1. Install Ollama: https://ollama.ai  (free, Mac/Linux/Windows)
    2. Run: ollama pull mistral
    3. Then: python rag/build_index.py

Run: python rag/build_index.py
"""

import os
import json
import duckdb
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

DB_PATH      = os.getenv("DUCKDB_PATH", "./data/swiftpulse.duckdb")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
CHROMA_DIR   = "./data/chroma_db"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # free, runs locally


# ── Document Sources ──────────────────────────────────────────────────────────
POLICY_DOCS = [
    {
        "id": "return-policy",
        "content": """
        SwiftPulse Return Policy
        ========================
        All products carry a 30-day return window from date of delivery.
        Items must be in original condition with tags attached.
        Electronics (including wearable devices) have a 15-day return window.
        To initiate a return, visit /returns and enter your order ID.
        Refunds are processed within 5-7 business days to the original payment method.
        Final sale items, gift cards, and customized products are non-returnable.
        """,
    },
    {
        "id": "shipping-policy",
        "content": """
        SwiftPulse Shipping Policy
        ==========================
        Standard shipping (5-7 business days): Free on orders over $50.
        Expedited shipping (2-3 business days): $12.99.
        Overnight shipping: $24.99.
        International orders ship via DHL. Duties are the buyer's responsibility.
        Orders placed before 2PM CST ship same day (Mon-Fri).
        Wearable devices ship with a 2-year limited warranty card.
        """,
    },
    {
        "id": "wearable-faq",
        "content": """
        SwiftPulse AI Glasses FAQ
        =========================
        Q: How long does the battery last?
        A: The AI glasses battery lasts 8 hours on a single charge with active use.

        Q: Is the device waterproof?
        A: The device is water-resistant (IPX4) but not waterproof.

        Q: What sensors are included?
        A: Accelerometer, ambient light sensor, heart rate monitor, and front camera.

        Q: How do I update the firmware?
        A: Open the SwiftPulse app, go to Settings > Device > Check for Updates.

        Q: Is my health data private?
        A: All health data is stored locally on the device. Cloud sync is opt-in.
        """,
    },
]


def load_product_docs_from_db() -> list[Document]:
    """Pulls product catalog from DuckDB and converts to LangChain Documents."""
    con = duckdb.connect(DB_PATH, read_only=True)
    products = con.execute("SELECT * FROM products LIMIT 50").df()
    con.close()

    docs = []
    for _, row in products.iterrows():
        content = (
            f"Product: {row['name']}\n"
            f"Category: {row['category']}\n"
            f"SKU: {row['sku']}\n"
            f"Price: ${row['price']:.2f}\n"
            f"Description: {row['description']}\n"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": "product_catalog", "sku": row["sku"]},
        ))
    return docs


def build_index() -> Chroma:
    """Builds or loads the ChromaDB vector index."""
    print("📚 Building RAG index...")

    # Combine policy docs + product catalog
    all_docs = []

    # Policy documents
    for p in POLICY_DOCS:
        all_docs.append(Document(
            page_content=p["content"].strip(),
            metadata={"source": "policy", "doc_id": p["id"]},
        ))

    # Product catalog from DuckDB
    product_docs = load_product_docs_from_db()
    all_docs.extend(product_docs)
    print(f"  Loaded {len(POLICY_DOCS)} policy docs + {len(product_docs)} products")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(all_docs)
    print(f"  Split into {len(chunks)} chunks")

    # Embed using free local model (downloads ~90MB on first run)
    print(f"  Embedding with {EMBED_MODEL} (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Store in ChromaDB (persisted to disk)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR
    )
    print(f"  ✓ Index saved to {CHROMA_DIR}")
    return vectorstore


def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """Wraps the vector store in a RetrievalQA chain using Ollama."""
    prompt = PromptTemplate(
        template="""You are a helpful e-commerce support assistant for SwiftPulse.
Use the context below to answer the customer's question concisely and accurately.
If you cannot find the answer in the context, say so honestly.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    # Ollama runs locally — no API key needed
    llm = Ollama(base_url=OLLAMA_URL, model=OLLAMA_MODEL, temperature=0.1)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def demo_queries(chain: RetrievalQA):
    """Runs a few test queries to verify the pipeline works."""
    questions = [
        "What is the return policy for electronics?",
        "How long does the AI glasses battery last?",
        "How much does expedited shipping cost?",
    ]
    print("\n🧪 Running demo queries...")
    for q in questions:
        print(f"\n  Q: {q}")
        try:
            result = chain({"query": q})
            print(f"  A: {result['result'].strip()}")
        except Exception as e:
            print(f"  ⚠️  Ollama not running ({e})")
            print("     Install Ollama from https://ollama.ai and run: ollama pull mistral")
            break


def main():
    vectorstore = build_index()
    chain = build_qa_chain(vectorstore)
    demo_queries(chain)
    print("\n✅ RAG pipeline ready.")
    print("   Next: python wearable/sensor_simulator.py  (in a separate terminal)")
    print("   Then: python api/main.py")


if __name__ == "__main__":
    main()
