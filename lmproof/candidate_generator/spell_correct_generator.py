from typing import List
import pathlib

from symspellpy import SymSpell
import spacy
from symspellpy.symspellpy import Verbosity

from lmproof.candidate_generator.candidate_generator import CandidateEditGenerator
from lmproof.candidate_generator import utils

from lmproof.edit import Edit


class SpellCorrectGenerator(CandidateEditGenerator):
    def __init__(self, sym_spell: SymSpell, spacy_model: spacy.language.Language):
        self._sym_spell = sym_spell
        self._spacy = spacy_model

    @classmethod
    def load(cls, language: str) -> "SpellCorrectGenerator":
        # maximum edit distance per dictionary pre-calculation
        max_edit_distance_dictionary = 2
        prefix_length = 7
        # create object
        sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        if language == "en":
            dict_path = (
                pathlib.Path(__file__).parent
                / "resources"
                / "frequency_dictionary_en_82_765.txt"
            )
            sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
            spacy_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        else:
            raise RuntimeError(f"The language {language} is currently not language.")
        return cls(sym_spell, spacy_model)

    def candidate_edits(self, text: str) -> List[Edit]:
        tokenized = self._spacy(text)
        candidate_edits = []
        for token in tokenized:
            if token.is_alpha and token.pos_ != "PROPN":
                suggestions = self._sym_spell.lookup(
                    token.text, Verbosity.CLOSEST, 2, transfer_casing=True
                )
                substitutes = {s.term for s in suggestions} - {token.text, token.lower_}
                if substitutes:
                    current_candidate_edits = utils.get_edits(
                        token.i, tokenized, substitutes
                    )
                    candidate_edits.extend(current_candidate_edits)
        return candidate_edits
