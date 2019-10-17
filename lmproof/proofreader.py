from typing import List, Set, Optional, Tuple

import numpy as np

from .scorer import TransformerLMScorer, SentenceScorer
from .candidate_generators import (MatchedGenerator, CandidateGenerator,
                                   EnglishInflectedGenerator,
                                   SpellCorrectGenerator)


class Proofreader:

    def __init__(self,
                 candidate_generators: List[CandidateGenerator],
                 scorer: SentenceScorer,
                 threshold: float):
        self._candidate_generators = candidate_generators
        self._scorer = scorer
        self._threshold = threshold
        # TODO: Weight language model score by the prior probability of the edit made.
        #  The prior probability of edit can be obtained by the
        #  frequency of the error in grammar error correction corpora.
        #  P(edited_sentence) = P_lm(edited_sentence) * P(edit_type)

    @classmethod
    def load(cls, language: str, device: str = 'cpu') -> 'ProofReader':
        if language == 'en':
            scorer = TransformerLMScorer.load(language, device=device)
            match_gen = MatchedGenerator.load(language)
            inflect_gen = EnglishInflectedGenerator()
            spell_correct_gen = SpellCorrectGenerator.load(language)
            threshold = 0.1
            return cls([match_gen, inflect_gen, spell_correct_gen], scorer, threshold)
        else:
            raise RuntimeError('Currently unsupported language.')

    def _better_alternative(self, text: str, previous_candidates: Set[str]) -> Tuple[Optional[str], Set[str]]:
        candidates = [candidate for g in self._candidate_generators
                      for candidate in g.candidates(text)
                      if candidate not in previous_candidates]
        
        # Do Scoring in one shot to use batching internally.
        source_score, *candidate_scores = self._scorer.score([text] + candidates)
        # Add the threshold to bias towards source sentence.
        biased_source_score = source_score + self._threshold
        thresholded_scores = np.array(candidate_scores)
        best_idx = np.argmax(thresholded_scores)
        if candidate_scores[best_idx] > biased_source_score:
            best_candidate = candidates[best_idx]
        else:
            best_candidate = None

        return best_candidate, candidates

    def proofread(self, sentence: str) -> str:
        correction = sentence
        previous_candidates = set([sentence])
        while True:
            better_alternative, candidates = self._better_alternative(correction,
                                                                      previous_candidates)
            if not better_alternative:
                break
            else:
                correction = better_alternative
                previous_candidates.union(set(candidates))
        return correction
