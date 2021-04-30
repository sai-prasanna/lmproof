from typing import List, Optional, Protocol


class SentenceScorer(Protocol):
    def score(self, sentences: List[str]) -> List[Optional[float]]:
        ...
