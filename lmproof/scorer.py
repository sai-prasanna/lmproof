from typing import List

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    PreTrainedTokenizer,
    PreTrainedModel,
)


class SentenceScorer:
    def score(self, sentences: List[str]) -> List[float]:
        raise NotImplementedError()


class TransformerLMScorer(SentenceScorer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        device: str = "cpu",
        batch_size: int = 1,
        normalize: bool = False,
    ):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.normalize = False

    @classmethod
    def load(cls, language: str, device: str = "cpu") -> "TransformerLMScorer":
        if language == "en":
            # Setting an aribitary batch_size
            # Ideally we should find the max number of tokens that will compute
            # And dynamically batch.
            batch_size = 1 if device == "cpu" else 32
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelWithLMHead.from_pretrained("gpt2")
            return cls(tokenizer, model, device, batch_size)
        else:
            raise RuntimeError("Language {language} is not supported.")

    def score(self, sentences: List[str]) -> List[float]:
        scores = []
        for start_idx in range(0, len(sentences), self.batch_size):
            batch = sentences[start_idx : start_idx + self.batch_size]
            tokens_batch = [torch.LongTensor(self.tokenizer.encode(s)) for s in batch]
            # Pad inputs by a valid embedding id (0),
            # we will mask it during loss calculation and future is masked
            # in casual language models, so will not affect attention.
            inputs_batch = torch.nn.utils.rnn.pad_sequence(
                tokens_batch, batch_first=True, padding_value=0
            )
            # Pad by -1 for labels so that CrossEntropyLoss will ignore the ids.
            labels_batch = torch.nn.utils.rnn.pad_sequence(
                tokens_batch, batch_first=True, padding_value=-1
            )
            inputs_batch = inputs_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)

            lm_logits = self.model(inputs_batch)[0]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_batch[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            for i in range(shift_logits.size(0)):
                loss = loss_fct(shift_logits[i], shift_labels[i])
                score = -loss
                if self.normalize:
                    score /= len(tokens_batch[i]) - 1
                scores.append(score.item())
        return scores
