from typing import List, Set, Optional, Tuple, Dict, Callable

import numpy as np
import spacy

from lmproof.edit import Edit, Span
from lmproof.scorer import TransformerLMScorer, SentenceScorer
from lmproof.candidate_generators import (
    MatchedGenerator,
    CandidateEditGenerator,
    EnglishInflectedGenerator,
    SpellCorrectGenerator,
)


def apply_all_edits(text: str, edits: List[Edit], ignore: Set[str]) -> List[str]:
    candidates = []
    for edit in edits:
        candidate = text[: edit.span.start] + edit.text + text[edit.span.end :]
        if candidate not in ignore:
            candidates.append(candidate)
    return candidates


ApplyEdits = Callable[[str, List[Edit], Set[str]], List[str]]


class Proofreader:
    def __init__(
        self,
        candidate_generators: List[CandidateEditGenerator],
        scorer: SentenceScorer,
        threshold: float,
        apply_edits: ApplyEdits = apply_all_edits,
        max_iterations: int = 4,
    ):
        self._candidate_generators = candidate_generators
        self._scorer = scorer
        self._threshold = threshold
        self._apply_edits = apply_edits
        self._max_iterations = max_iterations
        # TODO: Weight language model score by the prior probability of the edit made.
        #  The prior probability of edit can be obtained by the
        #  frequency of the error in grammar error correction corpora.
        #  P(edited_sentence) = P_lm(edited_sentence) * P(edit_type)

    @classmethod
    def load(cls, language: str, device: str = "cpu") -> "Proofreader":
        if language == "en":
            scorer = TransformerLMScorer.load(language, device=device)
            match_gen = MatchedGenerator.load(language)
            inflect_gen = EnglishInflectedGenerator()
            spell_correct_gen = SpellCorrectGenerator.load(language)
            threshold = 0.1
            return cls([match_gen, inflect_gen, spell_correct_gen], scorer, threshold)
        else:
            raise RuntimeError("Currently unsupported language.")

    def _better_alternative(
        self, text: str, previous_candidates: Set[str]
    ) -> Tuple[Optional[str], Set[str]]:

        candidate_edits = [
            candidate_edit
            for g in self._candidate_generators
            for candidate_edit in g.candidate_edits(text)
        ]

        candidates = self._apply_edits(text, candidate_edits, previous_candidates)

        best_candidate: Optional[str] = None
        if candidates:
            # Do Scoring in one shot to use batching internally.
            source_score, *candidate_scores = self._scorer.score([text] + candidates)
            # Add the threshold to bias towards source sentence.
            biased_source_score = source_score + self._threshold
            candidate_scores = np.array(candidate_scores)
            best_idx = np.argmax(candidate_scores)
            if candidate_scores[best_idx] > biased_source_score:
                best_candidate = candidates[best_idx]
        return best_candidate, set(candidates)

    def proofread(self, sentence: str) -> str:
        correction = sentence
        previous_candidates = set([sentence])
        i = 0
        while i < self._max_iterations:
            better_alternative, candidates = self._better_alternative(
                correction, previous_candidates
            )
            if not better_alternative:
                break
            else:
                correction = better_alternative
                previous_candidates.union(set(candidates))
            i += 1
        return correction
