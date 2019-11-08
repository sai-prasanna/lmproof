from typing import List, Set, Optional, Tuple, Callable, Dict

from lmproof.edit import Edit, Span
from lmproof.scorer import TransformerLMScorer, SentenceScorer
from lmproof.candidate_generators import (
    MatchedGenerator,
    CandidateEditGenerator,
    EnglishInflectedGenerator,
    SpellCorrectGenerator,
)


EditsReducer = Callable[[str, List[Edit]], List[Edit]]


class Proofreader:
    def __init__(
        self,
        candidate_generators: List[CandidateEditGenerator],
        scorer: SentenceScorer,
        threshold: float,
        linear_errors: bool = True,
        edits_reducer: Optional[EditsReducer] = None,
        max_iterations: int = 4,
    ):
        self._candidate_generators = candidate_generators
        self._scorer = scorer
        self._threshold = threshold
        self._linear_errors = linear_errors
        self._edits_reducer = edits_reducer
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

    def proofread(self, sentence: str) -> str:
        correction = sentence
        if self._linear_errors:
            candidate_edits = [
                candidate_edit
                for g in self._candidate_generators
                for candidate_edit in g.candidate_edits(sentence)
            ]
            if self._edits_reducer:
                candidate_edits = self._edits_reducer(sentence, candidate_edits)

            candidates = [
                sentence[: edit.span.start] + edit.text + sentence[edit.span.end :]
                for edit in candidate_edits
            ]
            source_score, *candidate_scores = self._scorer.score(
                [sentence] + candidates
            )
            if source_score is not None:
                best_edit_scores: Dict[Span, Tuple[Edit, float]] = {}
                for edit, score in zip(candidate_edits, candidate_scores):
                    if score is not None and score > (source_score + self._threshold):
                        if (
                            edit.span not in best_edit_scores
                            or score > best_edit_scores[edit.span][1]
                        ):
                            best_edit_scores[edit.span] = (edit, score)
                best_edits = sorted(
                    [edit_score[0] for edit_score in best_edit_scores.values()],
                    key=lambda edit: edit.span.start,
                )
                current_idx = 0
                correction = ""
                for edit in best_edits:
                    if edit.span.start >= current_idx:
                        correction += sentence[current_idx : edit.span.start]
                        correction += edit.text
                    current_idx = edit.span.end
                correction += sentence[current_idx:]
        else:
            previous_candidates = set([sentence])
            i = 0
            while i < self._max_iterations:
                better_alternative, new_candidates = self._better_alternative(
                    correction, previous_candidates
                )
                if better_alternative is not None:
                    correction = better_alternative
                    previous_candidates.union(new_candidates)
                else:
                    break
                i += 1
        return correction

    def _better_alternative(
        self, text: str, previous_candidates: Set[str]
    ) -> Tuple[Optional[str], Set[str]]:

        candidate_edits = [
            candidate_edit
            for g in self._candidate_generators
            for candidate_edit in g.candidate_edits(text)
            if text[: candidate_edit.span.start]
            + candidate_edit.text
            + text[candidate_edit.span.end :]
            not in previous_candidates
        ]

        if self._edits_reducer:
            candidate_edits = self._edits_reducer(text, candidate_edits)
        candidates = [
            text[: edit.span.start] + edit.text + text[edit.span.end :]
            for edit in candidate_edits
        ]
        best_candidate: Optional[str] = None
        if candidates:
            # Do Scoring in one shot to use batching internally.
            source_score, *candidate_scores = self._scorer.score([text] + candidates)
            if source_score is not None:
                # Add the threshold to bias towards source sentence.
                best_score = source_score + self._threshold
                for candidate, candidate_score in zip(candidates, candidate_scores):
                    if candidate_score is not None and candidate_score > best_score:
                        best_candidate = candidate
                        best_score = candidate_score
        return best_candidate, set(candidates)
