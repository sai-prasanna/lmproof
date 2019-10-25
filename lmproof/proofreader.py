from typing import List, Set, Optional, Tuple, Dict

import numpy as np
from fitbert import FitBert

from .edit import Edit
from .scorer import TransformerLMScorer, SentenceScorer
from .candidate_generators import (
    MatchedGenerator,
    CandidateEditGenerator,
    EnglishInflectedGenerator,
    SpellCorrectGenerator,
)


class Proofreader:
    def __init__(
        self,
        candidate_generators: List[CandidateEditGenerator],
        scorer: SentenceScorer,
        threshold: float,
        fitbert: Optional[FitBert] = None,
        max_iterations: int = 4
    ):
        self._candidate_generators = candidate_generators
        self._scorer = scorer
        self._threshold = threshold
        self._fitbert = fitbert
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
            fitbert = FitBert(model_name='distilbert-base-uncased', 
                              disable_gpu=(device == "cpu"))
            threshold = 0.1
            return cls([match_gen, inflect_gen, spell_correct_gen], scorer, threshold, fitbert)
        else:
            raise RuntimeError("Currently unsupported language.")

    def _better_alternative(
        self, text: str, previous_candidates: Set[Edit]
    ) -> Tuple[Optional[str], Set[Edit]]:
       
        candidate_edits = [
            candidate
            for g in self._candidate_generators
            for candidate in g.candidates(text)
            if candidate not in previous_candidates
        ]
        candidates = []
        if self._fitbert:
            grouped_edits : Dict[Span, List[str]] = {}
            for edit in candidate_edits:
                if edit.span in grouped_edits:
                    grouped_edits[edit.span].append(edit)
                else:
                    grouped_edits[edit.span] = [edit]
            for edit_span, edits in grouped_edits.items():
                masked = text[:edit_span.start] + self._fitbert.mask_token + text[edit_span.end:]
                candidate = self._fitbert.fitb(masked, [e.text for e in edits] + [text[edit_span.start:edit_span.end]])
                if candidate != text:
                    candidates.append(candidate)
        else:
            for edit in candidate_edits:
                candidate = text[:edit.span.start] + edit.text + text[edit.span.end:]
                candidates.append(candidate)
        best_candidate = None
        if candidates:
            # Do Scoring in one shot to use batching internally.
            source_score, *candidate_scores = self._scorer.score([text] + candidates)
            # Add the threshold to bias towards source sentence.
            biased_source_score = source_score + self._threshold
            candidate_scores = np.array(candidate_scores)
            best_idx = np.argmax(candidate_scores)
            if candidate_scores[best_idx] > biased_source_score:
                best_candidate = candidates[best_idx]
        return best_candidate, candidates

    def proofread(self, sentence: str) -> str:
        correction = sentence
        previous_candidates = set([sentence])
        i = 0
        while i < (self._max_iterations - 1):
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
