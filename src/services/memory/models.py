from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

@dataclass(slots=True)
class MemoryItem:
    id: int
    summary: str
    priority: float
    embedding: Optional[np.ndarray]
    use_count: int
    last_used: datetime
    memory_type: str
    embedding_namespace: str = "memory"
