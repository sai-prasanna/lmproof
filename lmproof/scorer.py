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
        batch_size: int = 32,
        add_special_tokens: bool = False,
        normalize: bool = False,
        max_token_limit: int = 128,
    ):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.normalize = normalize
        self._add_special_tokens = add_special_tokens
        self._loss_ignore_idx = -100
        self._loss_fn = CrossEntropyLoss(ignore_index=self._loss_ignore_idx)
        self.max_token_limit = max_token_limit

    @classmethod
    def load(cls, language: str, device: str = "cpu") -> "TransformerLMScorer":
        if language == "en":
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelWithLMHead.from_pretrained("distilgpt2")
            return cls(tokenizer, model, device)
        else:
            raise RuntimeError(f"Language {language} is not supported.")

    def _predict(
        self, input_ids: List[torch.LongTensor], attention_mask: List[torch.LongTensor]
    ) -> List[float]:
        padded_batch = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        ).to(self.device)
        padded_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        ).to(self.device)
        labels_batch = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self._loss_ignore_idx
        ).to(self.device)
        with torch.no_grad():
            lm_logits = self.model(input_ids=padded_batch, attention_mask=padded_mask)[
                0
            ]
            shift_logits_batch = lm_logits[..., :-1, :].contiguous()
            shift_labels_batch = labels_batch[..., 1:].contiguous()
            scores = []
            for shift_logits, shift_labels in zip(
                shift_logits_batch, shift_labels_batch
            ):
                loss = self._loss_fn(shift_logits, shift_labels)
                score = -loss
                if self.normalize:
                    score = score / (padded_mask.sum() - 1)
                scores.append(score.cpu().item())
        return scores

    def score(self, sentences: List[str]) -> List[Optional[float]]:
        encoded = self.tokenizer.batch_encode_plus(sentences)
        scores: List[Optional[float]] = [None] * len(sentences)
        batch_input_ids = []
        batch_attention_mask = []
        batch_idx = []
        batch_size = 2
        for i in range(len(sentences)):
            if len(encoded["input_ids"][i]) < self.max_token_limit:
                batch_idx.append(i)
                input_ids = torch.LongTensor(encoded["input_ids"][i])  # type: ignore
                attention_mask = torch.LongTensor(encoded["attention_mask"][i])  # type: ignore
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)

            if len(batch_input_ids) == self.batch_size:
                batch_scores = self._predict(batch_input_ids, batch_attention_mask)
                for idx, score in zip(batch_idx, batch_scores):
                    scores[idx] = score
                batch_input_ids = []
                batch_attention_mask = []
                batch_idx = []
        if len(batch_input_ids) > 0:
            batch_scores = self._predict(batch_input_ids, batch_attention_mask)
            for idx, score in zip(batch_idx, batch_scores):
                scores[idx] = score
        return scores
