"""
Document processing pipeline:
1. Text extraction (PDF via PyMuPDF, TXT direct read)
2. Text cleaning (whitespace, dehyphenation)
3. Chunking (sliding window over sentences, page-aware)
"""
import re
import fitz  # PyMuPDF


def extract_text_by_page(filepath: str, doc_type: str):
    """Returns list of (page_number, raw_text)."""
    pages = []
    if doc_type == "pdf":
        pdf = fitz.open(filepath)
        for i, page in enumerate(pdf):
            text = page.get_text()
            pages.append((i + 1, text))
        pdf.close()
    else:  # txt
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            pages.append((1, f.read()))
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)          # de-hyphenate line-broken words
    text = re.sub(r"\n+", "\n", text)         # collapse multiple newlines
    text = re.sub(r"[ \t]+", " ", text)       # collapse repeated spaces/tabs
    text = re.sub(r" \n", "\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100):
    """
    Simple sliding-window chunker over characters, breaking on sentence
    boundaries where possible. chunk_size/overlap are in characters
    (kept small here since assessment docs are short).
    """
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""

    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current.strip())
            # start new chunk with overlap from the end of the previous one
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = (overlap_text + " " + sent).strip()

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 20]  # drop near-empty fragments


def process_document(filepath: str, doc_type: str):
    """
    Full pipeline for one file.
    Returns: list of dicts: {content, page_number, chunk_index}
    """
    pages = extract_text_by_page(filepath, doc_type)
    all_chunks = []
    chunk_idx = 0

    for page_num, raw in pages:
        cleaned = clean_text(raw)
        page_chunks = chunk_text(cleaned)
        for c in page_chunks:
            all_chunks.append({
                "content": c,
                "page_number": page_num,
                "chunk_index": chunk_idx
            })
            chunk_idx += 1

    return all_chunks
