# pylint:disable=bad-continuation
from typing import List, Set, Dict, Optional


def english() -> Dict[str, Set[str]]:
    substitutions = [
        # Article/Determiner
        ["a", "an", "the", ""],
        # Prepositions
        ["about", "at", "by", "for", "from", "in", "of", "on", "to", "with", ""],
        # Commonly confused words
        ["bear", "bare"],
        ["lose", "loose"],
    ]
    return {
        each_element: set(replacement_list) - {each_element}
        for replacement_list in substitutions
        for each_element in replacement_list
    }
