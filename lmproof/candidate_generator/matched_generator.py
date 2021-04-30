from typing import Dict, Set, List

import spacy
from spacy.lang.en import English

from lmproof.candidate_generator.candidate_generator import CandidateEditGenerator
from lmproof.candidate_generator import utils
from lmproof.edit import Edit


class MatchedGenerator(CandidateEditGenerator):
    def __init__(
        self, substitutions: Dict[str, Set[str]], spacy_model: spacy.language.Language,
    ):
        self._spacy = spacy_model
        self._word2substitutes = substitutions

    @classmethod
    def load(cls, language: str) -> "MatchedGenerator":
        if language == "en":
            spacy_model = English()
            return cls(substitutions=en_substitutions(), spacy_model=spacy_model)
        else:
            raise RuntimeError(f"The language {language} is currently not language.")

    def candidate_edits(self, text: str) -> List[Edit]:
        tokenized = self._spacy.tokenizer(text)
        candidate_edits = []
        for token in tokenized:
            # Currently we enumerate tokens, if it matches anything word
            # that can be susbtituted, we generate full sentence with only
            # that word substituted with its alternatives.
            current_token_lower = token.lower_
            if current_token_lower in self._word2substitutes:
                # Get other substitutes, we don't do it during creation to avoid
                # creating new substitution set for each word.)
                substitutes = self._word2substitutes[current_token_lower] - {
                    current_token_lower
                }
                current_candidate_edits = utils.get_edits(
                    token.i, tokenized, substitutes
                )
                candidate_edits.extend(current_candidate_edits)
        return candidate_edits


def en_substitutions() -> Dict[str, Set[str]]:
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
