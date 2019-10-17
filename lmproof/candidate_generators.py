import pathlib
from typing import List, Set

import lemminflect
import spacy
from symspellpy.symspellpy import SymSpell, Verbosity

from .substitutions import english

class CandidateGenerator:
    def candidates(self, text: str) -> List[str]:
        raise NotImplementedError()

class MatchedGenerator(CandidateGenerator):

    def __init__(self, substitutions: List[Set[str]], spacy_model_name: str):
        self._spacy = spacy.load(spacy_model_name, disable=['ner', 'parser', 'tagger'])
        self._substitutions = substitutions
        self._word2substitutes = {word:substs for substs in substitutions
                                  for word in substs if word}

    @classmethod
    def load(cls, language: str) -> 'MatchedGenerator':
        if language == 'en':
            return cls(spacy_model_name='en',
                       substitutions=english())
        else:
            raise RuntimeError(f'The language {language} is currently not language.')

    def candidates(self, text: str) -> List[str]:
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
                substitutes = self._word2substitutes[current_token_lower] - {current_token_lower}
                sentences = _substituted_sentences(token.i, tokenized, substitutes)
                candidates.extend(sentences)
        return candidates


class EnglishInflectedGenerator(CandidateGenerator):

    def __init__(self):
        self._spacy = spacy.load('en', disable=['ner', 'parser', 'tagger'])

    def candidates(self, text: str) -> List[str]:
        tokenized = self._spacy.tokenizer(text)
        candidates = []
        for token in tokenized:
            lemmas = {lemma for lemmas in lemminflect.getAllLemmas(token.text).values()
                      for lemma in lemmas}
            inflections = {inflection for lemma in lemmas
                           for inflections in lemminflect.getAllInflections(lemma).values()
                           for inflection in inflections}
            substitutes = inflections - {token.text}
            sentences = _substituted_sentences(token.i, tokenized, substitutes)
            candidates.extend(sentences)
        return candidates

class SpellCorrectGenerator(CandidateGenerator):

    def __init__(self, sym_spell: SymSpell, spacy_model_name: str):
        self._sym_spell = sym_spell
        self._spacy = spacy.load(spacy_model_name, disable=['ner', 'parser'])

    @classmethod
    def load(cls, language: str) -> 'SpellCorrectGenerator':
        # maximum edit distance per dictionary pre-calculation
        max_edit_distance_dictionary = 2
        prefix_length = 7
        # create object
        sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        if language == 'en':
            dict_path = pathlib.Path(__file__).parent / 'resources' / 'frequency_dictionary_en_82_765.txt'
            sym_spell.create_dictionary(str(dict_path))
            spacy_model_name = 'en'
        else:
            raise RuntimeError(f'The language {language} is currently not language.')
        return cls(sym_spell, spacy_model_name)

    def candidates(self, text: str) -> List[str]:
        tokenized = self._spacy(text)
        candidates = []
        for token in tokenized:
            if token.is_alpha and token.pos_ != "PROPN":
                suggestions = self._sym_spell.lookup(token.text, Verbosity.CLOSEST, 2, transfer_casing=True)
                substitutes = {s.term for s in suggestions} - {token.text, token.lower_}
                if substitutes:
                    sentences = _substituted_sentences(token.i, tokenized, substitutes)
                    candidates.extend(sentences)
        return candidates


def _substituted_sentences(token_idx: int,
                           tokenized_sentence: spacy.tokens.Doc,
                           substitutes: List[str]) -> List[str]:
    candidates = []
    prefix = tokenized_sentence[:token_idx]
    suffix = tokenized_sentence[token_idx+1:]
    replaced_token = tokenized_sentence[token_idx]
    # Loop through the input alternative candidates
    for substitute in substitutes:
        candidate = prefix.text_with_ws
        if substitute: # Could be empty for deletions.
            if replaced_token.is_title:
                substitute = substitute.title()
            elif replaced_token.is_upper:
                substitute = substitute.upper()
            candidate += substitute + replaced_token.whitespace_
        candidate += suffix.text_with_ws
        candidates.append(candidate)
    return candidates
