-- Run this in Supabase SQL editor before starting the app

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Documents table (one row per uploaded file)
create table if not exists documents (
    id uuid primary key default gen_random_uuid(),
    filename text not null,
    doc_type text,              -- e.g. pdf, txt
    uploaded_at timestamptz default now(),
    summary text,                -- AI-generated doc summary (filled after processing)
    num_chunks int default 0
);

-- 3. Chunks table (one row per text chunk, with embedding)
create table if not exists chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid references documents(id) on delete cascade,
    chunk_index int not null,
    content text not null,
    page_number int,
    embedding vector(384)        -- 384 = dim for all-MiniLM-L6-v2 (sentence-transformers)
);

-- 4. Index for fast similarity search (cosine distance)
create index if not exists chunks_embedding_idx
    on chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- 5. Similarity search function used by the backend
create or replace function match_chunks (
    query_embedding vector(384),
    match_count int default 6
)
returns table (
    id uuid,
    document_id uuid,
    filename text,
    content text,
    page_number int,
    similarity float
)
language sql stable
as $$
    select
        chunks.id,
        chunks.document_id,
        documents.filename,
        chunks.content,
        chunks.page_number,
        1 - (chunks.embedding <=> query_embedding) as similarity
    from chunks
    join documents on documents.id = chunks.document_id
    order by chunks.embedding <=> query_embedding
    limit match_count;
$$;
