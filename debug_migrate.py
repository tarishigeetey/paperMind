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
    body={"query": {"match_all": {}}, "size": 1}
)
hit = response["hits"]["hits"][0]
src = hit["_source"]
raw_embedding = src.pop("embedding", None)
print(f"embedding type: {type(raw_embedding)}")
print(f"embedding[:100]: {str(raw_embedding)[:100]}")
if isinstance(raw_embedding, str):
    embedding_list = json.loads(raw_embedding)
else:
    embedding_list = raw_embedding
print(f"list type: {type(embedding_list)}")
print(f"list len: {len(embedding_list)}")
print(f"first element type: {type(embedding_list[0])}")
embedding_str = "[" + ",".join(str(float(x)) for x in embedding_list) + "]"
print(f"embedding_str[:50]: {embedding_str[:50]}")

with engine.connect() as conn:
    try:
        conn.execute(text("""
            INSERT INTO paper_chunks (chunk_id, arxiv_id, chunk_text, embedding)
            VALUES (:chunk_id, :arxiv_id, :chunk_text, CAST(:embedding AS vector))
        """), {
            "chunk_id": "test_001",
            "arxiv_id": src.get("arxiv_id"),
            "chunk_text": src.get("chunk_text", "test"),
            "embedding": embedding_str,
        })
        conn.commit()
        print("SUCCESS — minimal insert worked")
    except Exception as e:
        print(f"ERROR: {e}")
