import functools
import json
import logging
import numpy as np
from typing import Optional, Tuple, List
from src.config import EMBEDDING_OUTPUT_DIM

logger = logging.getLogger(__name__)

class EmbeddingHandler:
    def __init__(self):
        self.expected_dim = EMBEDDING_OUTPUT_DIM

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec

    def validate_dimension(self, vec: np.ndarray) -> bool:
        if vec is None or len(vec) == 0:
            return False
        return len(vec) == self.expected_dim

    @functools.lru_cache(maxsize=3000)
    def _parse_bytes(self, data: bytes) -> Optional[np.ndarray]:
        try:
            vec = np.frombuffer(data, dtype=np.float32)
            if self.validate_dimension(vec):
                return self.normalize(vec)
        except Exception as e:
            logger.debug(f"[EMB-PARSE] Failed to parse bytes embedding: {e}")
        return None

    def parse(self, data: any) -> Optional[np.ndarray]:
        if data is None:
            return None

        if isinstance(data, bytes):
            return self._parse_bytes(data)

        try:
            vec = None
            if isinstance(data, bytes):
                vec = np.frombuffer(data, dtype=np.float32)
            elif isinstance(data, str):
                vec = np.array(json.loads(data), dtype=np.float32)
            elif isinstance(data, (list, np.ndarray)):
                vec = np.array(data, dtype=np.float32)

            if vec is not None and self.validate_dimension(vec):
                return self.normalize(vec)
        except Exception as e:
            logger.debug(f"[EMB-PARSE] Failed to parse embedding: {e}")
        return None

    def prepare(self, embedding: List[float]) -> Optional[Tuple[np.ndarray, bytes]]:
        try:
            vec = np.array(embedding, dtype=np.float32)
            if self.validate_dimension(vec):
                normalized = self.normalize(vec)
                return normalized, normalized.tobytes()
        except Exception as e:
            logger.error(f"[EMB-PREPARE] {e}")
        return None

    def compute_similarity_matrix(self, vectors: List[np.ndarray], query: np.ndarray) -> np.ndarray:
        if not vectors:
            return np.array([])
        try:
            matrix = np.stack(vectors)
            return np.dot(matrix, query)
        except Exception as e:
            logger.error(f"[EMB-SIMILARITY] {e}")
            return np.zeros(len(vectors), dtype=np.float32)
