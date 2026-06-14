# The index name — used everywhere we reference this index
ARXIV_PAPERS_INDEX = "arxiv_papers"

# Complete index configuration
# This is sent to OpenSearch once when creating the index
ARXIV_PAPERS_MAPPING = {
    "settings": {
        # Single shard for local dev — no need for distributed search
        # Production would use 3-5 shards for larger datasets
        "number_of_shards": 1,
        "number_of_replicas": 0,  # no replicas for dev — saves RAM
        "analysis": {
            "analyzer": {
                # standard_analyzer: tokenise + lowercase + remove stopwords
                # Used for authors — simple word splitting is enough
                "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                # text_analyzer: full NLP pipeline for academic text
                # Used for title and abstract — where BM25 matters most
                "text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",  # "Attention" → "attention"
                        "stop",  # remove "is", "the", "a" etc
                        "snowball",  # stem: "running" → "run"
                    ],
                },
            }
        },
    },
    "mappings": {
        # strict = reject any field not defined here
        # Prevents accidental indexing of extra fields
        # Like a strict database schema with no extra columns allowed
        "dynamic": "strict",
        "properties": {
            # ── Identifier fields ────────────────────────────────
            # keyword = exact match, no analysis
            # Used for filtering and aggregations
            "arxiv_id": {"type": "keyword"},
            # ── Searchable text fields ───────────────────────────
            # text = full-text search with BM25
            # .keyword subfield = exact match on the same data
            # Having both lets you search AND filter/sort on the same field
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256,  # don't index very long titles as keyword
                    }
                },
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {
                "type": "text",
                "analyzer": "text_analyzer",  # full NLP pipeline
            },
            # ── Filter fields ────────────────────────────────────
            # keyword = exact match for category filtering
            # e.g. categories=["cs.AI"] filters to AI papers only
            "categories": {"type": "keyword"},
            # Full paper text — used in Week 5 for RAG context
            "raw_text": {"type": "text", "analyzer": "text_analyzer"},
            # ── Non-searchable fields ────────────────────────────
            # keyword = stored but not full-text searched
            "pdf_url": {"type": "keyword"},
            # ── Date fields ──────────────────────────────────────
            # date type enables date range filtering and date sorting
            # OpenSearch parses ISO format automatically
            "published_date": {"type": "date"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        },
    },
}
