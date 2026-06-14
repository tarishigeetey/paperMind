import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PaperQueryBuilder:
    """
    Builds OpenSearch Query DSL for paper search.

    Uses the Builder pattern — configure once, call build() to get the query.
    Like JPA CriteriaBuilder or Spring's QueryDSL.

    Encapsulates all query complexity:
    - Multi-field BM25 search with field boosting
    - Category filtering
    - Highlighting matched text
    - Sorting by relevance or date
    - Pagination
    """

    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
    ):
        self.query = query
        self.size = size
        self.from_ = from_

        # Default field boosting:
        # title^3  = title matches are 3x more important
        # abstract^2 = abstract matches are 2x more important
        # authors^1 = author matches count at face value
        # This mirrors how humans judge paper relevance

        self.fields = fields or ["title^3", "abstract^2", "author^2"]

        self.categories = categories
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers

    def build(self) -> Dict[str, Any]:
        """
        Assemble the complete OpenSearch query.
        Calls each builder method and combines results.
        """

        query_body = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }

        # Only add sort if needed — omitting it defaults to _score (relevance)
        sort = self._build_sort()
        if sort:
            query_body["sort"] = sort

        return query_body

    def _build_query(self) -> Dict[str, Any]:
        """
        Build the main bool query with must + filter clauses.

        must  → text search (affects BM25 score)
        filter → category filter (yes/no, doesn't affect score)
        """
        must_clauses = []

        if self.query.strip():
            # Has text → use BM25 multi-field search
            must_clauses.append(self._build_text_query())
        else:
            # No text → match everything
            # Used for browsing all papers in a category
            must_clauses.append({"match_all": {}})

        filter_clauses = self._build_filters()

        bool_query = {"must": must_clauses}

        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {"bool": bool_query}

    def _build_text_query(self) -> Dict[str, Any]:
        """
        Build the multi-field BM25 text search query.

        multi_match searches across multiple fields simultaneously.
        type: best_fields → uses the highest scoring field
        fuzziness: AUTO → handles typos ("attnetion" finds "attention")
        prefix_length: 2 → first 2 chars must be correct (reduces false positives)
        """
        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,
                # best_fields: score = best single field match
                # Prevents gaming by repeating terms across many fields
                "type": "best_fields",
                # OR = any term can match (more recall)
                # AND = all terms must match (more precision)
                # OR is better for research papers — partial matches still useful
                "operator": "or",
                # AUTO = edit distance based on word length
                # 1-2 chars: exact, 3-5 chars: 1 edit, 6+ chars: 2 edits
                "fuzziness": "AUTO",
                # First 2 characters must match exactly
                # Prevents "cat" from matching "bat" via fuzziness
                "prefix_length": 2,
            }
        }

    def _build_filters(self) -> List[Dict[str, Any]]:
        """
        Build filter clauses — applied after scoring, don't affect score.
        Like a WHERE clause in SQL.
        """
        filters = []

        if self.categories:
            # terms = match any of the provided values
            # Like SQL: WHERE categories IN ('cs.AI', 'cs.LG')
            filters.append({"terms": {"categories": self.categories}})

        return filters

    def _build_source_fields(self) -> List[str]:
        """
        Define which fields to return in the response.
        Like SELECT arxiv_id, title, ... instead of SELECT *
        Excludes raw_text — potentially huge, not needed in search results
        """
        return [
            "arxiv_id",
            "title",
            "authors",
            "abstract",
            "categories",
            "published_date",
            "pdf_url",
        ]

    def _build_highlight(self) -> Dict[str, Any]:
        """
        Configure result highlighting.

        Highlighting wraps matched terms in <mark> tags so the UI
        can show users exactly WHY a result matched their query.
        Like Google's bold text in search snippets.

        title: fragment_size=0 means return the entire title, not a snippet
        abstract: fragment_size=150 means return 150 char snippets
        """
        return {
            "fields": {
                "title": {
                    "fragment_size": 0,  # return full title
                    "number_of_fragments": 0,  # as one piece
                },
                "abstract": {
                    "fragment_size": 150,  # 150 char snippets
                    "number_of_fragments": 3,  # up to 3 snippets
                    "pre_tags": ["<mark>"],  # wrap match start
                    "post_tags": ["</mark>"],  # wrap match end
                },
                "authors": {
                    "fragment_size": 0,
                    "number_of_fragments": 0,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
            },
            # highlight even if match is in a different field
            "require_field_match": False,
        }

    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        """
        Build sort configuration.

        Three modes:
        1. latest_papers=True → sort by date descending
        2. Has text query → no explicit sort (uses _score/relevance)
        3. No text query → sort by date (browsing mode)
        """
        if self.latest_papers:
            # Explicit date sort — user wants newest papers
            # _score as tiebreaker
            return [{"published_date": {"order": "desc"}}, "_score"]

        if self.query.strip():
            # Text search — let BM25 score determine order
            # Omitting sort = OpenSearch uses _score automatically
            return None

        # No text, no latest_papers — browsing mode
        # Show newest papers first
        return [{"published_date": {"order": "desc"}}, "_score"]


def build_search_query(
    query: str,
    size: int = 10,
    from_: int = 0,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function — builds a search query with minimal config.
    Used in tests and notebooks where you don't need all options.
    """
    builder = PaperQueryBuilder(
        query=query,
        size=size,
        from_=from_,
        categories=categories,
    )
    return builder.build()
