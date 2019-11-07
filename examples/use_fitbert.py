from typing import List, Dict
import logging

import lmproof
from lmproof.edit import Edit, Span
from fitbert import FitBert

logger = logging.getLogger(__name__)


class FitBertApplyEdits:
    def __init__(self, fitbert: FitBert):
        self._fitbert = fitbert

    @classmethod
    def load(cls, language: str = "en", device: str = "cpu"):
        if language == "en":
            fitbert = FitBert(
                model_name="distilbert-base-uncased", disable_gpu=(device == "cpu")
            )
            return cls(fitbert=fitbert)

    def __call__(self, text: str, edits: List[Edit]) -> List[str]:
        candidates = []
        grouped_edits: Dict[Span, List[Edit]] = {}
        for edit in edits:
            if edit.span in grouped_edits:
                grouped_edits[edit.span].append(edit)
            else:
                grouped_edits[edit.span] = [edit]
        for edit_span, edits in grouped_edits.items():
            masked = (
                text[: edit_span.start]
                + self._fitbert.mask_token
                + text[edit_span.end :]
            )
            logger.debug(f"Masked sequence - {masked}")

            candidate = self._fitbert.fitb(
                masked,
                [e.text for e in edits] + [text[edit_span.start : edit_span.end]],
                # All the edit texts and the original edit span text are the options
            )
            if candidate != text:
                candidates.append(candidate)
        return candidates


if __name__ == "__main__":
    # Use Bert to reduce the number of edited candidates to be scored by
    # the next model.
    # This is experimental, in practise this is slower and results in reduction in recall.
    #  pip install fitbert
    logging.basicConfig(level=logging.DEBUG)
    proofer = lmproof.load("en")
    proofer.apply_edits = FitBertApplyEdits.load("en")
    print(proofer.proofread("Foxes lived on the Shire today."))
