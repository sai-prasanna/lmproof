import pathlib
from typing import List, Set

import lemminflect
import spacy
from spacy.lang.en import English
from symspellpy.symspellpy import SymSpell, Verbosity

from .substitutions import english
from .edit import Edit, Span


class CandidateEditGenerator:
    def candidates(self, text: str) -> List[Edit]:
        raise NotImplementedError()


class MatchedGenerator(CandidateEditGenerator):
    def __init__(
        self, substitutions: List[Set[str]], spacy_model: spacy.language.Language
    ):
        self._spacy = spacy_model
        self._substitutions = substitutions
        self._word2substitutes = {
            word: substs for substs in substitutions for word in substs if word
        }

    @classmethod
    def load(cls, language: str) -> "MatchedGenerator":
        if language == "en":
            spacy_model = English()
            return cls(substitutions=english(), spacy_model=spacy_model)
        else:
            raise RuntimeError(f"The language {language} is currently not language.")

    def candidates(self, text: str) -> List[Edit]:
        tokenized = self._spacy.tokenizer(text)
        candidates = []
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
                current_candidates = _edits(token.i, tokenized, substitutes)
                candidates.extend(current_candidates)
        return candidates


class EnglishInflectedGenerator(CandidateEditGenerator):
    def __init__(self):
        self._spacy = English()

    def candidates(self, text: str) -> List[Edit]:
        tokenized = self._spacy.tokenizer(text)
        candidates = []
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
            current_candidates = _edits(token.i, tokenized, substitutes)
            candidates.extend(current_candidates)
        return candidates


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
            sym_spell.create_dictionary(str(dict_path))
            spacy_model = spacy.load('en', disable=['parser', 'ner'])
        else:
            raise RuntimeError(f"The language {language} is currently not language.")
        return cls(sym_spell, spacy_model)

    def candidates(self, text: str) -> List[Edit]:
        tokenized = self._spacy(text)
        candidates = []
        for token in tokenized:
            if token.is_alpha and token.pos_ != "PROPN":
                suggestions = self._sym_spell.lookup(
                    token.text, Verbosity.CLOSEST, 2, transfer_casing=True
                )
                substitutes = {s.term for s in suggestions} - {token.text, token.lower_}
                if substitutes:
                    current_candidates = _edits(token.i, tokenized, substitutes)
                    candidates.extend(current_candidates)
        return candidates

def _edits(
    token_idx: int, tokenized_sentence: spacy.tokens.Doc, substitutes: List[str]
) -> List[Edit]:
    candidates = []
    replaced_token = tokenized_sentence[token_idx]
    for substitute in substitutes:
        candidate = ''
        if substitute:  # Could be empty for deletions.
            if replaced_token.is_title:
                substitute = substitute.title()
            elif replaced_token.is_upper:
                substitute = substitute.upper()
            candidate = Edit(Span(
                replaced_token.idx,
                replaced_token.idx + len(replaced_token)
            ), substitute)
            candidates.append(candidate)
    return candidates
