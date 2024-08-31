from t5tiny import T5EncoderModel, T5Config
from tinygrad import nn, Tensor, dtypes
from tinygrad.nn.state import torch_load

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

model = T5EncoderModel(config)

state_dict = nn.state.get_state_dict(model)

for key in state_dict:
    state_dict[key].replace(state_dict[key].cast("bfloat16").realize())

# print(nn.state.get_state_dict(model).keys())

load_state_dict = torch_load("small.bin")

for key in load_state_dict:
    load_state_dict[key].replace(load_state_dict[key].to("NV").cast("bfloat16").realize())

nn.state.load_state_dict(model, load_state_dict)

toks = {'input_ids': Tensor([[1712, 3182,    3,    9, 4698, 1803,    1,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0]], dtype=dtypes.long)}


with Tensor.test():
    print(model(toks["input_ids"])["last_hidden_states"].numpy())