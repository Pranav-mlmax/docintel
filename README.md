# DocIntel — Multi-Document Search & Q&A System

An AI/ML prototype that lets a user upload multiple documents (PDF/TXT),
processes them into a searchable knowledge base, and answers natural
language questions across the **entire collection** — synthesizing
information from multiple documents and citing exact sources (filename +
page + similarity score).

Built for the IAIP 2026–27 AI/ML hands-on assessment.

---

## Architecture

```
┌─────────────┐      ┌──────────────────────┐      ┌────────────────────┐
│  Frontend    │ ---> │  Flask backend        │ ---> │  Supabase (pgvector)│
│  (HTML/JS)   │      │  - upload/process      │      │  documents + chunks │
│              │ <--- │  - embed (local)       │ <--- │  vector similarity  │
└─────────────┘      │  - search + LLM QA      │      └────────────────────┘
                       │  (OpenRouter)          │
                       └──────────────────────┘
```

### Pipeline (maps to the assessment requirements)

1. **Multiple document input** — drag-and-drop / file picker, accepts
   multiple PDF/TXT files in one upload batch (`/api/upload`).
2. **Document processing**
   - **Text extraction**: PyMuPDF (`fitz`) for PDFs, direct read for TXT.
   - **Cleaning**: de-hyphenation, whitespace normalization.
   - **Segmentation/chunking**: sentence-aware sliding window (~700 chars,
     100-char overlap), page-tagged.
   - **Metadata extraction**: filename, doc type, page number, chunk index
     stored alongside each chunk.
3. **Natural language search** — query is embedded with the same model
   (`all-MiniLM-L6-v2`, local/free via `sentence-transformers`) and matched
   against all chunks in Supabase using pgvector cosine similarity
   (`match_chunks` SQL function), across **all documents at once**.
4. **AI-based question answering** — top-k matched chunks (which may span
   multiple different documents) are passed to an LLM via **OpenRouter**
   with a prompt instructing it to synthesize across sources and cite
   which document supports which part of the answer.
5. **Source identification** — every answer is returned with the backing
   chunks: filename, page number, similarity score, and excerpt — shown
   in the UI under "Sources".

### Extra features implemented
- **Per-document summarization** on upload (stored in `documents.summary`).
- **Multi-document summarization** — "Summarize Entire Collection" button
  synthesizes all individual summaries into one combined overview,
  surfacing connections between documents.
- **Semantic search** (not keyword matching) — vector similarity means a
  query like "computing infrastructure spend" can match a chunk that says
  "procurement of computing infrastructure" even without exact word overlap.

### Not implemented (time-boxed scope — mention if asked)
- OCR for scanned/image-only PDFs (would add `pytesseract` fallback when
  `page.get_text()` returns near-empty).
- Cross-encoder reranking of retrieved chunks.
- Conversational follow-up memory (currently each query is independent).
- Doc classification/tagging.

---

## Setup

### 1. Supabase
1. Create a project at supabase.com (free tier).
2. Open the SQL editor and run `supabase_schema.sql` from this repo.
3. Grab your Project URL and `anon`/`service_role` key from Project Settings → API.

### 2. OpenRouter
1. Sign up at openrouter.ai, generate an API key.
2. Pick a free model, e.g. `meta-llama/llama-3.1-8b-instruct:free` (default in `.env.example`).

### 3. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: paste SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY

python app.py
# Flask runs on http://localhost:5000
```

### 4. Frontend
Just open `frontend/index.html` in a browser (it calls `localhost:5000`
by default — edit the `API` constant at the top of the `<script>` if your
backend is hosted elsewhere).

---

## Deploying (free tier)

**Backend → Render.com (free web service)**
1. Push this repo to GitHub.
2. On Render: New → Web Service → connect repo → root dir `backend`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add environment variables (`SUPABASE_URL`, `SUPABASE_KEY`,
   `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`) in Render's dashboard.

**Frontend → GitHub Pages / Netlify / Vercel**
1. Update the `API` constant in `frontend/index.html` to your deployed
   Render backend URL.
2. Deploy the `frontend` folder as a static site (Netlify drag-and-drop,
   or GitHub Pages from the `frontend` directory).

**Supabase** is already hosted — no deploy step needed there.

---

## Demo flow (for the interview)

1. Upload the 5 sample IAIP documents.
2. Ask: *"What is the budget allocated to Project C and what is its
   procurement status?"* — answer should synthesize the Budget Allocation
   Report + Procurement Status Report + Server Procurement Proposal,
   citing all three.
3. Click "Summarize Entire Collection" to show multi-doc synthesis.
4. Point at the Sources panel to show grounding/citation, not hallucination.
