# pylint:disable=bad-continuation
from typing import List, Set


def english() -> List[Set[str]]:
    return [
        # Article/Determiner
        {"a", "an", "the", ""},
        # Prepositions
        {"about", "at", "by", "for", "from", "in", "of", "on", "to", "with", ""},
        # Commonly confused words
        {"bear", "bare"},
        {"lose", "loose"},
    ]
