from typing import List, Optional, Protocol
import logging

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from lmproof.scorer.scorer import SentenceScorer

logger = logging.getLogger(__name__)


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
        self.max_token_limit = max_token_limit

    @classmethod
    def load(cls, language: str, device: str = "cpu") -> "TransformerLMScorer":
        if language == "en":
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("distilgpt2")
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
            input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id
        ).to(self.device)
        with torch.no_grad():
            lm_logits = self.model(input_ids=padded_batch, attention_mask=padded_mask)[
                0
            ]
            shift_logits_batch = lm_logits[..., :-1, :].contiguous()
            shift_labels_batch = labels_batch[..., 1:].contiguous()
            shift_label_mask = padded_mask[..., 1:].contiguous()
            # Log softmax and select the log probabilities which correspond to next token label
            log_prob = torch.gather(
                F.log_softmax(shift_logits_batch, dim=-1),
                -1,
                shift_labels_batch.unsqueeze(-1),
            )
            # Remove the log prob corresponding to padded tokens and Normalize by length
            normalized_log_prob = (
                torch.sum(log_prob.squeeze(-1) * shift_label_mask, -1)
                * 1
                / shift_label_mask.sum(-1)
            )
        return [o.item() for o in normalized_log_prob.cpu()]

    def score(self, sentences: List[str]) -> List[Optional[float]]:
        encoded = self.tokenizer.batch_encode_plus(sentences)
        scores: List[Optional[float]] = [None] * len(sentences)
        batch_input_ids = []
        batch_attention_mask = []
        batch_idx = []
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
