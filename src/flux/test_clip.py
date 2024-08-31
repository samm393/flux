import torch
from transformers import CLIPTokenizer, CLIPTextModel

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", max_length = 77)
hf_module = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", max_length = 77, torch_dtype = torch.bfloat16)


text = "a yellow cat"

batch_encoding = tokenizer(
            text,
            truncation=True,
            max_length=77,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
# print(batch_encoding["input_ids"])
print(hf_module(batch_encoding["input_ids"]))

import tinygrad
from extra.models.clip import Tokenizer, Closed
from tinygrad import Tensor, nn

tiny_tokenizer = Tokenizer.ClipTokenizer()

batch_encoding = Tensor([tiny_tokenizer.encode(text)])

# print(batch_encoding.numpy())

tiny_text_model = Closed.ClipTextModel(None)

state_dict = nn.state.torch_load("clip.bin")

del state_dict["logit_scale"]
del state_dict["text_model.embeddings.position_ids"]

nn.state.load_state_dict(tiny_text_model, state_dict)

print(tiny_text_model.text_model(batch_encoding).numpy())