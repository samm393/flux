from sentencepiece import SentencePieceProcessor
import torch
spp = SentencePieceProcessor(model_file="/root/.cache/huggingface/hub/models--google--t5-v1_1-xxl/snapshots/3db67ab1af984cf10548a73467f0e5bca2aaaeb2" + "/spiece.model")

class T5TokenizerMine:
    def __init__(self):
        self.spp = SentencePieceProcessor(model_file="/root/.cache/huggingface/hub/models--google--t5-v1_1-xxl/snapshots/3db67ab1af984cf10548a73467f0e5bca2aaaeb2" + "/spiece.model")

    def __call__(self, text, max_length, *args, **kwargs):
        if isinstance(text, str):
            text = [text]
        encoded = self.spp.Encode(text)
        ret = torch.zeros((len(encoded), max_length), dtype=torch.int)
        for i, row in enumerate(encoded):
            ret[i, :len(row) + 1] = torch.tensor(row + [1])
        return {"input_ids":ret}
    
tokenizer = T5TokenizerMine()

print(tokenizer("beautiful woman", 10))