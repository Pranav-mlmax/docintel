"""
AI engine:
- Local embeddings via sentence-transformers (free, no API key needed)
- LLM calls (QA, summarization) via OpenRouter (free-tier model)
"""
import os
import requests
from sentence_transformers import SentenceTransformer

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Loaded once at startup — 384-dim embeddings, fast + free
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    return _embed_model.encode(text).tolist()


def embed_batch(texts: list[str]):
    return _embed_model.encode(texts).tolist()


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str:
    if not OPENROUTER_API_KEY:
        return "[Error: OPENROUTER_API_KEY not set in .env]"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM call failed: {e}]"


def answer_question(query: str, retrieved_chunks: list[dict]) -> str:
    """
    retrieved_chunks: list of {filename, content, page_number, similarity}
    Builds a source-grounded QA prompt across multiple documents.
    """
    context_blocks = []
    for i, c in enumerate(retrieved_chunks):
        context_blocks.append(
            f"[Source {i+1}: {c['filename']}, page {c.get('page_number', '?')}]\n{c['content']}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a document research assistant. Answer the user's question using ONLY "
        "the provided source excerpts, which may come from multiple different documents. "
        "If the answer draws on more than one document, synthesize them and explicitly "
        "mention which sources (by filename) support each part of your answer. "
        "If the excerpts don't contain the answer, say so clearly instead of guessing."
    )
    user_prompt = f"Sources:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

    return call_llm(system_prompt, user_prompt)


def summarize_document(text: str, filename: str) -> str:
    system_prompt = (
        "You summarize organizational documents concisely and factually, "
        "in 3-5 sentences, highlighting key figures, entities, and status information."
    )
    user_prompt = f"Document: {filename}\n\nContent:\n{text[:6000]}\n\nSummary:"
    return call_llm(system_prompt, user_prompt, max_tokens=300)


def multi_doc_summary(doc_summaries: list[dict]) -> str:
    """doc_summaries: list of {filename, summary}"""
    blocks = "\n\n".join(f"{d['filename']}: {d['summary']}" for d in doc_summaries)
    system_prompt = (
        "You synthesize summaries of multiple related organizational documents into "
        "one combined overview, highlighting connections and overlaps between documents."
    )
    user_prompt = f"Individual document summaries:\n\n{blocks}\n\nWrite a combined overview:"
    return call_llm(system_prompt, user_prompt, max_tokens=400)
