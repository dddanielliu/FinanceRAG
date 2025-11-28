import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from financerag.common import CrossEncoder, Reranker

logger = logging.getLogger(__name__)

def merge_retrieval_results_tuple(retrievers: Tuple[Dict[str, Dict[str, float]], ...]) -> Dict[str, List[Dict[str, float]]]:
    """
    Merge multiple retrieval results (provided as a tuple) into RRF-compatible format.

    Args:
        retrievers (Tuple[Dict[str, Dict[str, float]], ...]):
            A tuple of retrieval result dicts, each mapping query_id -> {doc_id: score}

    Returns:
        Dict[str, List[Dict[str, float]]]:
            Each query maps to a list of result dicts from each retrieval method
    """

    merged_results: Dict[str, List[Dict[str, float]]] = {}
    for idx, retriever in enumerate(retrievers):
        for qid, result in retriever.items():
            if qid not in merged_results:
                merged_results[qid] = []
            merged_results[qid].append(result)

    return merged_results

# Adapted from https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py
class RRFReranker(Reranker):
    """
    A reranker class that utilizes Reciprocal Rank Fusion (RRF) to combine multiple
    retrieval results for each query into a single fused ranking. 

    RRF is a simple yet effective rank aggregation method that assigns higher weight
    to documents appearing near the top of individual result lists but allows contributions
    from all sources. This class is suitable for hybrid search pipelines where multiple
    retrieval methods (e.g., BM25, dense embeddings) provide candidate documents.

    Methods:
        rerank:
            Takes in a corpus, queries, and initial retrieval results from multiple sources,
            and returns the top-k documents per query after applying RRF.
            The results are returned as a dictionary mapping query IDs to dictionaries
            of document IDs and their fused scores.
    """

    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}

    def rerank(
            self,
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str],
            results: Dict[str, List[Dict[str, float]]] | Tuple,
            top_k: int,
            batch_size: Optional[int] = None,
            k: int = 60,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Rerank results using Reciprocal Rank Fusion (RRF).

        Args:
            corpus: Dictionary of document_id -> document content (unused in RRF but included for signature)
            queries: Dictionary of query_id -> query text
            results: Dictionary of query_id -> List of retrieval dicts [{'doc_id': score, ...}, ...]
            top_k: Number of top results to return per query
            k: RRF parameter controlling rank impact

        Returns:
            Dict[query_id, Dict[doc_id, fused_score]]
        """

        if isinstance(results, tuple):
            results = merge_retrieval_results_tuple(results)

        reranked_results: Dict[str, Dict[str, float]] = {}

        for qid, result_lists in results.items():
            fused_scores = defaultdict(float)

            for result in result_lists:
                # Sort documents by their score descending
                sorted_docs = sorted(result.items(), key=lambda x: x[1], reverse=True)
                for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                    fused_scores[doc_id] += 1 / (k + rank)

            # Take top_k documents
            top_docs = dict(
                sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            )
            reranked_results[qid] = top_docs

        self.results = reranked_results
        return reranked_results
