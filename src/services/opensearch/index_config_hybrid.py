"""OpenSearch index configuration for hybrid search (BM25 + Vector)."""

ARXIV_PAPERS_CHUNKS_INDEX = "arxiv-papers-chunks"

ARXIV_PAPERS_CHUNKS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "index.knn.space_type": "cosinesimil",
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                "text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"],
                },
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "paper_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "chunk_word_count": {"type": "integer"},
            "start_char": {"type": "integer"},
            "end_char": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16,
                    },
                },
            },
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {"type": "text", "analyzer": "text_analyzer"},
            "categories": {"type": "keyword"},
            "published_date": {"type": "date"},
            "section_title": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        },
    },
}

HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",
                    "rank_constant": 60,
                }
            }
        }
    ],
}
