from sentencepiece import SentencePieceProcessor
import torch
from transformers import T5Tokenizer, T5EncoderModel

# spp = SentencePieceProcessor(model_file="/root/.cache/huggingface/hub/models--google--t5-v1_1-xxl/snapshots/3db67ab1af984cf10548a73467f0e5bca2aaaeb2" + "/spiece.model")

# class T5TokenizerMine:
#     def __init__(self):
#         self.spp = SentencePieceProcessor(model_file="/root/.cache/huggingface/hub/models--google--t5-v1_1-xxl/snapshots/3db67ab1af984cf10548a73467f0e5bca2aaaeb2" + "/spiece.model")

#     def __call__(self, text, max_length, *args, **kwargs):
#         if isinstance(text, str):
#             text = [text]
#         encoded = self.spp.Encode(text)
#         ret = torch.zeros((len(encoded), max_length), dtype=torch.int)
#         for i, row in enumerate(encoded):
#             ret[i, :len(row) + 1] = torch.tensor(row + [1])
#         return {"input_ids":ret}
    
# tokenizer = T5TokenizerMine()
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small", max_length=64)
toks = tokenizer("cat eating a bagel", truncation=True,
            max_length=64,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )


encoder_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small", max_length=64, torch_dtype=torch.bfloat16)
encoder_model = encoder_model.eval().requires_grad_(False)

from t5 import T5EncoderModel as myT5
from t5 import T5Config


config = T5Config(**{
  "d_ff": 1024,
  "d_kv": 64,
  "d_model": 512,
  "layer_norm_epsilon": 1e-06,
  "num_decoder_layers": 8,
  "num_heads": 6,
  "num_layers": 8,
  "relative_attention_num_buckets": 32,
  "vocab_size": 32128,
})

my_encoder = myT5(config)

state_dict = torch.load("small.bin", weights_only=True)

for k in state_dict:
   state_dict[k].to(torch.bfloat16)

# print(state_dict.keys())

# print(my_encoder(toks["input_ids"]))

my_encoder.to(torch.bfloat16)
my_encoder.load_state_dict(state_dict, strict=False)

# my_encoder = T5Stack.from_pretrained("google/t5-v1_1-small", max_length=64, torch_dtype=torch.bfloat16)
my_encoder = my_encoder.eval().requires_grad_(False)

print(encoder_model(toks["input_ids"], attention_mask=None))
print(my_encoder(toks["input_ids"], attention_mask=None))

# print(my_encoder(toks["input_ids"], attention_mask=None)["last_hidden_states"].shape)
