import logging
from typing import List, Optional
from datetime import datetime
import math
import numpy as np

from src.services.memory.models import MemoryItem
from src.services.memory.embeddings import EmbeddingHandler

logger = logging.getLogger(__name__)

class MemoryScorer:
    def __init__(self,
                 semantic_weight: float = 0.65,
                 priority_weight: float = 0.15,
                 recency_weight: float = 0.15,
                 usage_weight: float = 0.05,
                 recency_halflife: float = 14.0):
        self.recency_halflife = recency_halflife
        self.weights = {
            'semantic': semantic_weight,
            'priority': priority_weight,
            'recency': recency_weight,
            'usage': usage_weight
        }
        self.metadata_weights = {
            'priority': 0.5,
            'recency': 0.3,
            'usage': 0.2
        }
        self.no_emb_penalty = 0.4

    def calculate(self, items: List[MemoryItem], query_emb: Optional[np.ndarray], emb_handler: EmbeddingHandler) -> np.ndarray:
        if not items:
            return np.array([])  
        if query_emb is not None and any(item.embedding is not None for item in items):
            return self._semantic_scoring(items, query_emb, emb_handler)
        return self._metadata_scoring(items)

    def _semantic_scoring(self, items: List[MemoryItem], query: np.ndarray,
                        handler: EmbeddingHandler) -> np.ndarray:
        count = len(items)
        sim_scores = np.zeros(count, dtype=np.float32)
        
        vectors = []
        indices = []
        for i, item in enumerate(items):
            if item.embedding is not None:
                vectors.append(item.embedding)
                indices.append(i)
        
        if vectors:
            similarities = handler.compute_similarity_matrix(vectors, query)
            for idx, sim_idx in enumerate(indices):
                sim_scores[sim_idx] = similarities[idx]

        recency = self._compute_recency([item.last_used for item in items])
        priority = np.array([item.priority for item in items], dtype=np.float32)
        usage = np.log1p(np.array([item.use_count for item in items], dtype=np.float32)) / 5.0

        scores = (
            sim_scores * self.weights['semantic'] +
            priority * self.weights['priority'] +
            recency * self.weights['recency'] +
            usage * self.weights['usage']
        )

        for i, item in enumerate(items):
            if item.embedding is None:
                scores[i] *= self.no_emb_penalty

        return scores

    def _metadata_scoring(self, items: List[MemoryItem]) -> np.ndarray:
        recency = self._compute_recency([item.last_used for item in items])
        priority = np.array([item.priority for item in items], dtype=np.float32)
        usage = np.log1p(np.array([item.use_count for item in items], dtype=np.float32)) / 5.0

        return (
            priority * self.metadata_weights['priority'] +
            recency * self.metadata_weights['recency'] +
            usage * self.metadata_weights['usage']
        )

    def _compute_recency(self, timestamps: List[datetime]) -> np.ndarray:
        if not timestamps:
            return np.array([], dtype=np.float32)

        try:
            now = datetime.now()
            scores = []
            for ts in timestamps:
                delta_seconds = (now - ts).total_seconds()
                delta_days = max(delta_seconds, 0) / 86400.0
                scores.append(math.exp(-delta_days / self.recency_halflife))
                
            return np.array(scores, dtype=np.float32)
            
        except Exception as e:
            logger.debug(f"[RECENCY] Calculation error: {e}")
            return np.zeros(len(timestamps), dtype=np.float32)
