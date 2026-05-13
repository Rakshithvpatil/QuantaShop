"""
dashboard/rag_chat.py
──────────────────────
Gradio chat interface for RAG-powered product/policy Q&A.
Run: python dashboard/rag_chat.py
(Requires: python rag/build_index.py  +  ollama pull mistral)
"""

import os
import requests
import gradio as gr

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

EXAMPLE_QUESTIONS = [
    "What is the return policy for electronics?",
    "How long does the AI glasses battery last?",
    "How much does expedited shipping cost?",
    "Is the wearable device waterproof?",
    "How do I update the firmware on my glasses?",
]


def ask_question(message: str, history: list) -> str:
    """Sends the question to the RAG API and returns the answer."""
    if not message.strip():
        return "Please enter a question."
    try:
        resp = requests.post(
            f"{API_BASE}/rag/query",
            json={"question": message},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            answer  = data.get("answer", "No answer returned.")
            sources = data.get("sources", [])
            source_note = f"\n\n*Sources: {', '.join(sources)}*" if sources else ""
            return answer.strip() + source_note
        else:
            return f"API error {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        return (
            "Cannot reach the API. Make sure:\n"
            "1. `uvicorn api.main:app --port 8000` is running\n"
            "2. `python rag/build_index.py` has been executed\n"
            "3. Ollama is running with: `ollama serve`"
        )
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="SwiftPulse RAG Assistant") as demo:
    gr.Markdown("# SwiftPulse Product & Policy Assistant")
    gr.Markdown("Ask anything about products, shipping, returns, or wearable devices.")

    chatbot = gr.ChatInterface(
        fn=ask_question,
        examples=EXAMPLE_QUESTIONS,
        title="",
        description="Powered by LangChain + ChromaDB + Mistral (local, free)",
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
