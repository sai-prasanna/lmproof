from typing import List, Optional
import logging

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    PreTrainedTokenizer,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)


class SentenceScorer:
    def score(self, sentences: List[str]) -> List[Optional[float]]:
        raise NotImplementedError()


class TransformerLMScorer(SentenceScorer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        device: str = "cpu",
        batch_size: int = 1,
        add_special_tokens: bool = False,
        normalize: bool = False,
    ):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.normalize = normalize
        self._add_special_tokens = add_special_tokens
        self._loss_fn = CrossEntropyLoss(ignore_index=-1)

    @classmethod
    def load(cls, language: str, device: str = "cpu") -> "TransformerLMScorer":
        if language == "en":
            # Setting an aribitary batch_size
            # Ideally we should find the max number of tokens that will compute
            # And dynamically batch.
            batch_size = 1 if device == "cpu" else 32
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelWithLMHead.from_pretrained("distilgpt2")
            return cls(tokenizer, model, device, batch_size)
        else:
            raise RuntimeError(f"Language {language} is not supported.")

    def score(self, sentences: List[str]) -> List[Optional[float]]:
        scores: List[Optional[float]] = []
        for start_idx in range(0, len(sentences), self.batch_size):
            batched_sentences = sentences[start_idx : start_idx + self.batch_size]

            batch_scores: List[Optional[float]] = [None] * len(batched_sentences)
            batch_scored_idx = []

            tokenized_batch = []
            for i, sentence in enumerate(batched_sentences):
                tokens = self.tokenizer.encode(
                    sentence, add_special_tokens=self._add_special_tokens
                )
                if len(tokens) <= self.tokenizer.max_len:
                    tokenized_batch.append(torch.LongTensor(tokens))  # type: ignore
                    batch_scored_idx.append(i)
            if tokenized_batch:
                # Pad inputs by a valid embedding id (0),
                # we will mask it during loss calculation and future is masked
                # in casual language models, so will not affect attention.
                inputs_batch = torch.nn.utils.rnn.pad_sequence(
                    tokenized_batch, batch_first=True, padding_value=0
                )
                # Pad by -1 for labels so that CrossEntropyLoss will ignore the ids.
                labels_batch = torch.nn.utils.rnn.pad_sequence(
                    tokenized_batch, batch_first=True, padding_value=-1
                )
                inputs_batch = inputs_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                lm_logits = self.model(inputs_batch)[0]
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels_batch[..., 1:].contiguous()

                for i in range(shift_logits.size(0)):
                    loss = self._loss_fn(shift_logits[i], shift_labels[i])
                    score = -loss.item()
                    if self.normalize:
                        score = score / (tokenized_batch[i].size(0) - 1)
                    batch_scores[batch_scored_idx[i]] = score
            scores.extend(batch_scores)

        return scores
