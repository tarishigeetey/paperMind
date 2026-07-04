import json
from sqlalchemy import create_engine, text
from src.services.opensearch.factory import make_opensearch_client_fresh
from src.config import get_settings

settings = get_settings()
os_client = make_opensearch_client_fresh(settings)
db_url = settings.postgres_database_url
if not db_url.startswith("postgresql+psycopg2://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
engine = create_engine(db_url)

response = os_client.client.search(
    index=os_client.index_name,
    body={"query": {"match_all": {}}, "size": 500}
)
hits = response["hits"]["hits"]
print(f"Found {len(hits)} chunks in OpenSearch")

success = 0
errors = 0
with engine.connect() as conn:
    for hit in hits:
        src = hit["_source"]

        # Extract and parse embedding — remove from src completely
        raw_embedding = src.pop("embedding", None)
        if not raw_embedding:
            continue
        if isinstance(raw_embedding, str):
            embedding_list = json.loads(raw_embedding)
        else:
            embedding_list = raw_embedding
        embedding_str = "[" + ",".join(str(float(x)) for x in embedding_list) + "]"

        chunk_id = src.get("chunk_id") or f"{src.get('arxiv_id')}_{src.get('chunk_index', 0)}"

        try:
            conn.execute(text("""
                INSERT INTO paper_chunks
                    (chunk_id, arxiv_id, title, chunk_text, chunk_index,
                     categories, published_date, authors, abstract, pdf_url, embedding)
                VALUES
                    (:chunk_id, :arxiv_id, :title, :chunk_text, :chunk_index,
                     :categories, :published_date, :authors, :abstract, :pdf_url,
                     CAST(:embedding AS vector))
                ON CONFLICT (chunk_id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding  = EXCLUDED.embedding
            """), {
                "chunk_id":       chunk_id,
                "arxiv_id":       src.get("arxiv_id"),
                "title":          src.get("title"),
                "chunk_text":     src.get("chunk_text", ""),
                "chunk_index":    src.get("chunk_index", 0),
                "categories":     src.get("categories"),
                "published_date": src.get("published_date"),
                "authors":        src.get("authors"),
                "abstract":       src.get("abstract"),
                "pdf_url":        src.get("pdf_url"),
                "embedding":      embedding_str,
            })
            success += 1
        except Exception as e:
            print(f"Error on {chunk_id}: {e}")
            errors += 1

    conn.commit()

print(f"Migrated {success}/{len(hits)} chunks to pgvector ({errors} errors)")
