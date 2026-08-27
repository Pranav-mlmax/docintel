"""
AI engine:
- Embeddings via Hugging Face Inference API (free tier, no heavy local model —
  keeps memory footprint small enough for Render's free 512MB instance)
- LLM calls (QA, summarization) via OpenRouter (free-tier model)
"""
import os
import time
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HF_API_KEY = os.getenv("HF_API_KEY")  # optional but recommended (free, higher rate limit)
HF_EMBED_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


def _hf_headers():
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    return headers


def _call_hf_embed(texts: list[str], retries: int = 5):
    """Calls HF feature-extraction endpoint. Retries on cold-start (503) or transient network errors."""
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(HF_EMBED_URL, headers=_hf_headers(), json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
            continue
        if resp.status_code == 200:
            data = resp.json()
            # API returns token-level embeddings sometimes; mean-pool if needed
            result = []
            for item in data:
                if isinstance(item[0], list):  # token-level -> mean pool
                    n = len(item)
                    dim = len(item[0])
                    avg = [sum(tok[d] for tok in item) / n for d in range(dim)]
                    result.append(avg)
                else:
                    result.append(item)
            return result
        elif resp.status_code == 503:
            time.sleep(3)
            continue
        else:
            raise RuntimeError(f"HF embed failed: {resp.status_code} {resp.text}")
    raise RuntimeError(f"HF embed failed after {retries} retries: {last_err}")


def embed_text(text: str):
    return _call_hf_embed([text])[0]


def embed_batch(texts: list[str]):
    # HF free tier is happier with smaller batches; chunk into groups of 20
    results = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i + 20]
        results.extend(_call_hf_embed(batch))
    return results


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
  
