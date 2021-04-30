from typing import List
import lemminflect
from spacy.lang.en import English

from lmproof.candidate_generator.candidate_generator import CandidateEditGenerator
from lmproof.edit import Edit
from lmproof.candidate_generator import utils


class EnglishInflectedGenerator(CandidateEditGenerator):
    def __init__(self):
        self._spacy = English()

    def candidate_edits(self, text: str) -> List[Edit]:
        tokenized = self._spacy.tokenizer(text)
        candidate_edits = []
        for token in tokenized:
            lemmas = {
                lemma
                for lemmas in lemminflect.getAllLemmas(token.text).values()
                for lemma in lemmas
            }
            inflections = {
                inflection
                for lemma in lemmas
                for inflections in lemminflect.getAllInflections(lemma).values()
                for inflection in inflections
            }
            substitutes = inflections - {token.text}
            current_candidate_edits = utils.get_edits(token.i, tokenized, substitutes)
            candidate_edits.extend(current_candidate_edits)
        return candidate_edits
