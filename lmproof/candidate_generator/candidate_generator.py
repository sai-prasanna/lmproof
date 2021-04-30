from typing import List, Protocol

from lmproof.edit import Edit


class CandidateEditGenerator(Protocol):
    def candidate_edits(self, text: str) -> List[Edit]:
        ...
