import torch
from pytorch_transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                                  TransfoXLTokenizer, TransfoXLLMHeadModel,
                                  OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)

class SentenceScorer:
    def score(self, sentence: str) -> float:
        raise NotImplementedError()


class LanguageModelScorer(SentenceScorer):
    def __init__(self, model_name='gpt2', gpu: bool = False):
        if model_name == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
        elif model_name == 'transfo-xl-wt103':
            tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
            model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        elif model_name == 'openai-gpt':
            tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        else:
            raise Exception(f'Unknown model {model_name}')
        # Load pre-trained model tokenizer (vocabulary)
        self._tokenizer = tokenizer
        self._model = model
        self._model.eval()
        self._device = torch.device("cuda:0") if gpu else torch.device("cpu")
        self._model.to(self._device)

    @classmethod
    def load(cls, language: str, gpu: bool = False) -> 'LanguageModelScorer':
        if language == 'en':
            return cls(model_name='gpt2', gpu=gpu)
        else:
            raise RuntimeError('Unknown language')

    def score(self, sentence: str):
        tokens = self._tokenizer.encode(sentence)
        input_ids = torch.tensor(tokens).unsqueeze(0)  #pylint:disable=not-callable
        input_ids = input_ids.to(self._device)
        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
        loss, _ = outputs[:2]
         # Normalize by subword token length. -1 to remove BOS token.
        sentence_prob = loss.item() / (len(tokens) - 1)
        return -sentence_prob
