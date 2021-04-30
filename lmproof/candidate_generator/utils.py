from typing import Set, List

import spacy
from lmproof.edit import Edit, Span


def get_edits(
    token_idx: int, tokenized_sentence: spacy.tokens.Doc, substitutes: Set[str]
) -> List[Edit]:
    candidate_edits = []
    replaced_token = tokenized_sentence[token_idx]
    for substitute in substitutes:
        if replaced_token.is_title:
            substitute = substitute.title()
        elif replaced_token.is_upper:
            substitute = substitute.upper()

        if token_idx == 0 and substitute == "" and len(tokenized_sentence) > 1:
            candidate = Edit(
                Span(replaced_token.idx, tokenized_sentence[1].idx + 1),
                tokenized_sentence[1].text[0].upper(),
            )
        else:
            candidate = Edit(
                Span(replaced_token.idx, replaced_token.idx + len(replaced_token)),
                substitute,
            )
        candidate_edits.append(candidate)
    return candidate_edits
