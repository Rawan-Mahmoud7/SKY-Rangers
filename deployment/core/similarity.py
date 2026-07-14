"""
=============================================================================
Modular Similarity Methods
=============================================================================
Strategy pattern for computing similarity between a query embedding and a
database of reference embeddings.

Default method: WeightedTopKSimilarity — 1/rank weighted cosine similarity
over the K nearest neighbors.

All methods assume L2-normalised embeddings so that dot-product == cosine.

Usage:
    >>> method = create_similarity_method("weighted_top_k", top_k=20)
    >>> score = method.compute(query_vec, database_matrix)
=============================================================================
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
# Abstract Base
# =========================================================================

class SimilarityMethod(ABC):
    """
    Base class for similarity computation between a query and a database.

    All embeddings are assumed L2-normalised (unit vectors).
    """

    @abstractmethod
    def compute(self, query: np.ndarray, database: np.ndarray) -> float:
        """
        Compute similarity score between query and database.

        Parameters
        ----------
        query : np.ndarray of shape (D,)
            Single L2-normalised query embedding.
        database : np.ndarray of shape (N, D)
            L2-normalised database embeddings.

        Returns
        -------
        float
            Similarity score in [0, 1] range.
        """
        ...


# =========================================================================
# Implementations
# =========================================================================

class WeightedTopKSimilarity(SimilarityMethod):
    """
    Weighted Top-K Similarity (default).

    1. Compute cosine similarity between query and all database vectors.
    2. Sort descending, take top-K.
    3. Weight each neighbor by 1/rank: w_i = 1/(i+1).
    4. Return weighted average of the top-K similarities.

    This favours the closest matches while still considering the
    broader neighbourhood — more robust than mean or max alone.
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def compute(self, query: np.ndarray, database: np.ndarray) -> float:
        if database.shape[0] == 0:
            return 0.0

        # Cosine similarity (dot product for L2-normalised vectors)
        sims = database @ query  # (N,)

        # Take top-K
        k = min(self.top_k, len(sims))
        top_k_indices = np.argpartition(sims, -k)[-k:]
        top_k_sims = sims[top_k_indices]

        # Sort descending for rank weighting
        sorted_sims = np.sort(top_k_sims)[::-1]

        # 1/rank weights
        weights = 1.0 / np.arange(1, k + 1, dtype=np.float64)
        weights /= weights.sum()

        return float(np.dot(weights, sorted_sims))


class MeanCosineSimilarity(SimilarityMethod):
    """
    Simple mean cosine similarity over the top-K neighbors.

    Less discriminative than weighted but useful as a baseline.
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def compute(self, query: np.ndarray, database: np.ndarray) -> float:
        if database.shape[0] == 0:
            return 0.0

        sims = database @ query
        k = min(self.top_k, len(sims))
        top_k_sims = np.partition(sims, -k)[-k:]

        return float(np.mean(top_k_sims))


class ThresholdVotingSimilarity(SimilarityMethod):
    """
    Fraction of database vectors with cosine similarity above a threshold.

    Returns a value in [0, 1] representing what fraction of the database
    "agrees" that this is a match. Useful for verifying identity when
    the database is single-class (e.g., Egypt only).
    """

    def __init__(self, threshold: float = 0.6, top_k: int = 20):
        self.threshold = threshold
        self.top_k = top_k

    def compute(self, query: np.ndarray, database: np.ndarray) -> float:
        if database.shape[0] == 0:
            return 0.0

        sims = database @ query
        k = min(self.top_k, len(sims))
        top_k_sims = np.partition(sims, -k)[-k:]

        return float(np.mean(top_k_sims >= self.threshold))


# =========================================================================
# Factory
# =========================================================================

_METHODS = {
    "weighted_top_k": WeightedTopKSimilarity,
    "mean_cosine": MeanCosineSimilarity,
    "threshold_voting": ThresholdVotingSimilarity,
}


def create_similarity_method(name: str, **kwargs) -> SimilarityMethod:
    """
    Create a similarity method by name.

    Parameters
    ----------
    name : str
        One of: "weighted_top_k", "mean_cosine", "threshold_voting"
    **kwargs
        Passed to the constructor (e.g., top_k=20).

    Returns
    -------
    SimilarityMethod instance
    """
    if name not in _METHODS:
        raise ValueError(
            f"Unknown similarity method '{name}'. "
            f"Available: {list(_METHODS.keys())}"
        )
    method = _METHODS[name](**kwargs)
    logger.info(f"Created similarity method: {name} ({kwargs})")
    return method
