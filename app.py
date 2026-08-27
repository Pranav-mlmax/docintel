import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client

from doc_processor import process_document
from ai_engine import embed_batch, embed_text, answer_question, summarize_document, multi_doc_summary

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

ALLOWED_EXT = {"pdf", "txt"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/upload", methods=["POST"])
def upload_documents():
    """Accepts multiple files, processes and indexes each one."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    results = []

    for file in files:
        if not file or not allowed_file(file.filename):
            results.append({"filename": file.filename if file else "unknown", "status": "skipped (unsupported type)"})
            continue

        doc_type = file.filename.rsplit(".", 1)[1].lower()
        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
        file.save(save_path)

        try:
            # 1. Extract + clean + chunk
            chunks = process_document(save_path, doc_type)

            if not chunks:
                results.append({"filename": file.filename, "status": "no extractable text"})
                continue

            # 2. Insert document row
            doc_row = supabase.table("documents").insert({
                "filename": file.filename,
                "doc_type": doc_type,
                "num_chunks": len(chunks),
            }).execute()
            document_id = doc_row.data[0]["id"]

            # 3. Embed chunks (batched for speed)
            texts = [c["content"] for c in chunks]
            embeddings = embed_batch(texts)

            # 4. Insert chunks with embeddings
            chunk_rows = [{
                "document_id": document_id,
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "page_number": c["page_number"],
                "embedding": emb,
            } for c, emb in zip(chunks, embeddings)]

            supabase.table("chunks").insert(chunk_rows).execute()

            # 5. Generate + store document summary
            full_text = "\n".join(texts)
            summary = summarize_document(full_text, file.filename)
            supabase.table("documents").update({"summary": summary}).eq("id", document_id).execute()

            results.append({
                "filename": file.filename,
                "status": "processed",
                "document_id": document_id,
                "num_chunks": len(chunks),
                "summary": summary,
            })
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    return jsonify({"results": results})


@app.route("/api/documents", methods=["GET"])
def list_documents():
    docs = supabase.table("documents").select("id, filename, doc_type, uploaded_at, summary, num_chunks").order("uploaded_at", desc=True).execute()
    return jsonify({"documents": docs.data})


@app.route("/api/search", methods=["POST"])
def search():
    """
    Natural language query -> semantic search across all docs -> LLM answer with citations.
    Body: {"query": "...", "top_k": 6}
    """
    body = request.get_json(force=True)
    query = (body.get("query") or "").strip()
    top_k = int(body.get("top_k", 6))

    if not query:
        return jsonify({"error": "query is required"}), 400

    query_embedding = embed_text(query)

    match_res = supabase.rpc("match_chunks", {
        "query_embedding": query_embedding,
        "match_count": top_k
    }).execute()

    retrieved = match_res.data or []

    if not retrieved:
        return jsonify({
            "answer": "No documents have been indexed yet, or no relevant matches were found.",
            "sources": []
        })

    answer = answer_question(query, retrieved)

    sources = [{
        "filename": r["filename"],
        "page_number": r.get("page_number"),
        "similarity": round(r.get("similarity", 0), 3),
        "excerpt": r["content"][:300]
    } for r in retrieved]

    return jsonify({"answer": answer, "sources": sources})


@app.route("/api/summarize-all", methods=["GET"])
def summarize_all():
    """Multi-document summarization across the whole collection."""
    docs = supabase.table("documents").select("filename, summary").execute()
    doc_summaries = [d for d in docs.data if d.get("summary")]

    if not doc_summaries:
        return jsonify({"summary": "No documents indexed yet."})

    combined = multi_doc_summary(doc_summaries)
    return jsonify({"summary": combined, "documents": doc_summaries})


@app.route("/api/documents/<document_id>", methods=["DELETE"])
def delete_document(document_id):
    supabase.table("documents").delete().eq("id", document_id).execute()
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
