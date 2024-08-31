import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import torch
from PIL import Image
import argparse

import math
from typing import Callable

from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import torch_load

##math###############################################################################################################
def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = Tensor.scaled_dot_product_attention(q, k, v)
    x = x.rearrange("B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = Tensor.arange(0, dim, 2, dtype=dtypes.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    ##out = Tensor.einsum("...n,d->...nd", pos, omega) ##not suported in tinygrad
    out = pos.unsqueeze(-1) * omega.unsqueeze(0)


    out = Tensor.stack([Tensor.cos(out), -Tensor.sin(out), Tensor.sin(out), Tensor.cos(out)], dim=-1)
    out = out.rearrange("b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


##Layers#############################################################################################################

class EmbedND():
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = Tensor.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = Tensor.exp(-math.log(max_period) * Tensor.arange(start=0, end=half, dtype=dtypes.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = Tensor.cat([Tensor.cos(args), Tensor.sin(args)], dim=-1)
    if dim % 2:
        embedding = Tensor.cat([embedding, Tensor.zeros_like(embedding[:, :1])], dim=-1)
    if Tensor.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder():
    def __init__(self, in_dim: int, hidden_dim: int):
        
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm():
    def __init__(self, dim: int):
        
        self.scale = Tensor.ones(dim)

    def __call__(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = Tensor.rsqrt(Tensor.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).cast(dtype=x_dtype) * self.scale


class QKNorm():
    def __init__(self, dim: int):
        
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention():
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation():
    def __init__(self, dim: int, double: bool):
        
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def __call__(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock():
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def __call__(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = Tensor.cat((txt_q, img_q), dim=2)
        k = Tensor.cat((txt_k, img_k), dim=2)
        v = Tensor.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock():
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def __call__(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = Tensor.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(Tensor.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer():
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def __call__(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x





##Conditioner########################################################################################################
from sentencepiece import SentencePieceProcessor

class T5TokenizerMine:
    def __init__(self):
        self.spp = SentencePieceProcessor(model_file="t5spiece.model")

    def __call__(self, text, max_length, *args, **kwargs):
        if isinstance(text, str):
            text = [text]
        encoded = self.spp.Encode(text)
        ret = Tensor.zeros((len(encoded), max_length), dtype=dtypes.int)
        for i, row in enumerate(encoded):
            ret[i, :len(row) + 1] = Tensor(row + [1])
        return {"input_ids":ret}

# class HFEmbedder():
#     def __init__(self, version: str, max_length: int, **hf_kwargs):
#         print(f"{version=}")
        
#         self.is_clip = version.startswith("openai")
#         self.max_length = max_length
#         self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

#         if self.is_clip:
#             self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
#             self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
#         else:
#             self.tokenizer = T5TokenizerMine()
#             self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

#         self.hf_module = self.hf_module.eval().requires_grad_(False)

#     def __call__(self, text: list[str]) -> Tensor:
#         # if isinstance(self.tokenizer, T5Tokenizer):
#         #     return torch.tensor(self.tokenizer(text), device=self.device)
#         batch_encoding = self.tokenizer(
#             text,
#             truncation=True,
#             max_length=self.max_length,
#             return_length=False,
#             return_overflowing_tokens=False,
#             padding="max_length",
#             return_tensors="pt",
#         )
#         # if not self.is_clip:
#         #     print(batch_encoding)
#         #     print(self.my_tokenizer(text))

#         outputs = self.hf_module(
#             input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
#             attention_mask=None,
#             output_hidden_states=False,
#         )
#         return outputs[self.output_key]

from t5 import T5EncoderModel
from t5 import T5Config

class T5Embedder():
    def __init__(self):
        self.tokenizer = T5TokenizerMine()

        config = T5Config(**{
            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "layer_norm_epsilon": 1e-06,
            "num_decoder_layers": 24,
            "num_heads": 64,
            "num_layers": 24,
            "relative_attention_num_buckets": 32,
            "vocab_size": 32128,
            })

        self.encoder = T5EncoderModel(config)

        state_dict = nn.state.get_state_dict(self.encoder)

        for key in state_dict:
            state_dict[key].replace(state_dict[key].cast("bfloat16").realize())

        load_state_dict = torch_load("t5.bin")

        for key in load_state_dict:
            load_state_dict[key].replace(load_state_dict[key].to("cuda").cast("bfloat16").realize())

        nn.state.load_state_dict(model, load_state_dict)

    def __call__(self, text:str):
        toks = self.tokenizer(text)
        return self.encoder(toks)

from extra.models.clip import Tokenizer, Closed
from tinygrad import Tensor, nn

class ClipEmbedder():
    def __init__(self):
        self.tokenizer = Tokenizer.ClipTokenizer()
        self.encoder = Closed.ClipTextModel(None)
        state_dict = nn.state.torch_load("clip.bin")

        del state_dict["logit_scale"]
        del state_dict["text_model.embeddings.position_ids"]

        nn.state.load_state_dict(self.encoder, state_dict)
    
    def __call__(self, text):
        batch_encoding = Tensor([self.tokenizer.encode(text)])
        return self.encoder(batch_encoding)






##########Autoencoder#################################################################################################
@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: Tensor) -> Tensor:
    return x * Tensor.sigmoid(x)


class AttnBlock():
    def __init__(self, in_channels: int):
        
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.rearrange("b c h w -> b 1 (h w) c").contiguous()
        k = k.rearrange("b c h w -> b 1 (h w) c").contiguous()
        v = v.rearrange("b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return h_.rearrange("b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def __call__(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock():
    def __init__(self, in_channels: int, out_channels: int):
        
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample():
    def __init__(self, in_channels: int):
        
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample():
    def __init__(self, in_channels: int):
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder():
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = []
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder():
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian():
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        
        self.sample = sample
        self.chunk_dim = chunk_dim

    def __call__(self, z: Tensor) -> Tensor:
        mean, logvar = Tensor.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = Tensor.exp(0.5 * logvar)
            return mean + std * Tensor.randn_like(mean)
        else:
            return mean


class AutoEncoder():
    def __init__(self, params: AutoEncoderParams):
        
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def __call__(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))




######################################################################################################################

class Model:

    @dataclass
    class FluxParams:
        in_channels: int
        vec_in_dim: int
        context_in_dim: int
        hidden_size: int
        mlp_ratio: float
        num_heads: int
        depth: int
        depth_single_blocks: int
        axes_dim: list[int]
        theta: int
        qkv_bias: bool
        guidance_embed: bool


    class Flux():
        """
        Transformer model for flow matching on sequences.
        """

        def __init__(self, params):
            

            self.params = params
            self.in_channels = params.in_channels
            self.out_channels = self.in_channels
            if params.hidden_size % params.num_heads != 0:
                raise ValueError(
                    f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
                )
            pe_dim = params.hidden_size // params.num_heads
            if sum(params.axes_dim) != pe_dim:
                raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
            self.hidden_size = params.hidden_size
            self.num_heads = params.num_heads
            self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
            self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
            self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
            self.guidance_in = (
                MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
            )
            self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

            self.double_blocks = [
                    DoubleStreamBlock(
                        self.hidden_size,
                        self.num_heads,
                        mlp_ratio=params.mlp_ratio,
                        qkv_bias=params.qkv_bias,
                    )
                    for _ in range(params.depth)
                ]
            

            self.single_blocks = [
                    SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                    for _ in range(params.depth_single_blocks)
                ]
            

            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        def __call__(
            self,
            img: Tensor,
            img_ids: Tensor,
            txt: Tensor,
            txt_ids: Tensor,
            timesteps: Tensor,
            y: Tensor,
            guidance: Tensor | None = None,
        ) -> Tensor:
            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError("Input img and txt tensors must have 3 dimensions.")

            # running on sequences img
            img = self.img_in(img)
            vec = self.time_in(timestep_embedding(timesteps, 256))
            if self.params.guidance_embed:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
            vec = vec + self.vector_in(y)
            txt = self.txt_in(txt)

            ids = Tensor.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)

            for block in self.double_blocks:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            img = Tensor.cat((txt, img), 1)
            for block in self.single_blocks:
                img = block(img, vec=vec, pe=pe)
            img = img[:, txt.shape[1] :, ...]

            img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
            return img
















#####################################################################################################################

class Util:
    @dataclass
    class ModelSpec:
        params: Model.FluxParams
        ae_params: AutoEncoderParams
        ckpt_path: str | None
        ae_path: str | None
        repo_id: str | None
        repo_flow: str | None
        repo_ae: str | None


    configs = {
        "flux-dev": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-dev",
            repo_flow="flux1-dev.safetensors",
            repo_ae="ae.safetensors",
            ckpt_path=os.getenv("FLUX_DEV"),
            params=Model.FluxParams(
                in_channels=64,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=3072,
                mlp_ratio=4.0,
                num_heads=24,
                depth=19,
                depth_single_blocks=38,
                axes_dim=[16, 56, 56],
                theta=10_000,
                qkv_bias=True,
                guidance_embed=True,
            ),
            ae_path=os.getenv("AE"),
            ae_params=AutoEncoderParams(
                resolution=256,
                in_channels=3,
                ch=128,
                out_ch=3,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                z_channels=16,
                scale_factor=0.3611,
                shift_factor=0.1159,
            ),
        ),
        "flux-schnell": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-schnell",
            repo_flow="flux1-schnell.safetensors",
            repo_ae="ae.safetensors",
            ckpt_path=os.getenv("FLUX_SCHNELL"),
            params=Model.FluxParams(
                in_channels=64,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=3072,
                mlp_ratio=4.0,
                num_heads=24,
                depth=19,
                depth_single_blocks=38,
                axes_dim=[16, 56, 56],
                theta=10_000,
                qkv_bias=True,
                guidance_embed=False,
            ),
            ae_path=os.getenv("AE"),
            ae_params=AutoEncoderParams(
                resolution=256,
                in_channels=3,
                ch=128,
                out_ch=3,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                z_channels=16,
                scale_factor=0.3611,
                shift_factor=0.1159,
            ),
        ),
    }


    def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
        if len(missing) > 0 and len(unexpected) > 0:
            print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
            print("\n" + "-" * 79 + "\n")
            print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
        elif len(missing) > 0:
            print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        elif len(unexpected) > 0:
            print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


    def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
        # Loading Flux
        print("Init model")
        ckpt_path = Util.configs[name].ckpt_path
        if (
            ckpt_path is None
            and Util.configs[name].repo_id is not None
            and Util.configs[name].repo_flow is not None
            and hf_download
        ):
            ckpt_path = hf_hub_download(Util.configs[name].repo_id, Util.configs[name].repo_flow)

        with torch.device("meta" if ckpt_path is not None else device):
            model = Model.Flux(Util.configs[name].params).cast(dtypes.bfloat16)

        if ckpt_path is not None:
            print("Loading checkpoint")
            # load_sft doesn't support torch.device
            sd = load_sft(ckpt_path, device=str(device))
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            Util.print_load_warning(missing, unexpected)
        return model


    def load_t5(device: str | torch.device = "cuda", max_length: int = 512):
        # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
        return T5Embedder()


    def load_clip(device: str | torch.device = "cuda"):
        return ClipEmbedder()


    def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
        ckpt_path = Util.configs[name].ae_path
        if (
            ckpt_path is None
            and Util.configs[name].repo_id is not None
            and Util.configs[name].repo_ae is not None
            and hf_download
        ):
            ckpt_path = hf_hub_download(Util.configs[name].repo_id, Util.configs[name].repo_ae)

        # Loading the autoencoder
        print("Init AE")
        with torch.device("meta" if ckpt_path is not None else device):
            ae = AutoEncoder(Util.configs[name].ae_params)

        if ckpt_path is not None:
            sd = load_sft(ckpt_path, device=str(device))
            missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
            Util.print_load_warning(missing, unexpected)
        return ae


####################################################################################################################################

class Sampling:
    def get_noise(
        num_samples: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ):
        return torch.randn(
            num_samples,
            16,
            # allow for packing
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )


    def prepare(t5, clip, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
        bs, c, h, w = img.shape
        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)

        img = img.rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            ##img = repeat(img, "1 ... -> bs ...", bs=bs) ## not supported
            img = img.expand((bs, *img.shape[1:]))

        img_ids = Tensor.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + Tensor.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + Tensor.arange(w // 2)[None, :]
        ##img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs) ##not supported
        img_ids = img_ids.rearrange("h w c -> 1 (h w) c")
        img_ids = img_ids.expand((bs, *img_ids.shape[1:]))

        if isinstance(prompt, str):
            prompt = [prompt]
        txt = t5(prompt)
        if txt.shape[0] == 1 and bs > 1:
            ##txt = repeat(txt, "1 ... -> bs ...", bs=bs)
            txt = txt.expand((bs, *txt.shape[1:]))
        txt_ids = Tensor.zeros(bs, txt.shape[1], 3)

        vec = clip(prompt)
        if vec.shape[0] == 1 and bs > 1:
            ##vec = repeat(vec, "1 ... -> bs ...", bs=bs)
            vec = vec.expand((bs, *vec.shape[1:]))

        return {
            "img": img,
            "img_ids": img_ids.to(img.device),
            "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
            "vec": vec.to(img.device),
        }


    def time_shift(mu: float, sigma: float, t: Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


    def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b


    def get_schedule(
        num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ) -> list[float]:
        # extra step for zero
        timesteps = Tensor.linspace(1, 0, num_steps + 1)

        # shifting the schedule to favor high timesteps for higher signal images
        if shift:
            # estimate mu based on linear estimation between two points
            mu = Sampling.get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            timesteps = Sampling.time_shift(mu, 1.0, timesteps)

        return timesteps.tolist()


    def denoise(
        model: Model.Flux,
        # model input
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        # sampling parameters
        timesteps: list[float],
        guidance: float = 4.0,
    ):
        # this is ignored for schnell
        guidance_vec = Tensor.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = Tensor.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            img = img + (t_prev - t_curr) * pred

        return img


    def unpack(x: Tensor, height: int, width: int) -> Tensor:
        return x.rearrange(
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

#########################################################################################################

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


if __name__ == "__main__":
    default_prompt = "a horse sized cat eating a bagel"
    parser = argparse.ArgumentParser(description="Run Flux.1", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name',       type=str,       default="flux-schnell", help="Name of the model to load")
    parser.add_argument('--width',      type=int,       default=512,            help="width of the sample in pixels (should be a multiple of 16)")
    parser.add_argument('--height',     type=int,       default=512,            help="height of the sample in pixels (should be a multiple of 16)")
    parser.add_argument('--seed',       type=int,       default=None,           help="Set a seed for sampling")
    parser.add_argument('--prompt',     type=str,       default=default_prompt, help="Prompt used for sampling")
    parser.add_argument('--device',     type=str,       default="cuda" if torch.cuda.is_available() else "cpu", help="Pytorch device")
    parser.add_argument('--num_steps',  type=int,       default=None,           help="number of sampling steps (default 4 for schnell, 50 for guidance distilled)")
    parser.add_argument('--guidance',   type=float,     default=3.5,            help="guidance value used for guidance distillation")
    parser.add_argument('--offload',    type=bool,      default=False,          help="offload to cpu")
    parser.add_argument('--output_dir', type=str,       default = "output",     help="output directory")
    args = parser.parse_args()

    if args.name not in Util.configs:
        available = ", ".join(Util.configs.keys())
        raise ValueError(f"Got unknown model name: {args.name}, chose from {available}")

    torch_device = args.device
    if args.num_steps is None:
        num_steps = 4 if args.name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (args.height // 16)
    width = 16 * (args.width // 16)

    output_name = os.path.join(args.output_dir, "img_{idx}.jpg")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    with Tensor.test():
        # init all components
        t5 = Util.load_t5(torch_device, max_length=256 if args.name == "flux-schnell" else 512)
        clip = Util.load_clip(torch_device)
        model = Util.load_flow_model(args.name, device="cpu" if args.offload else torch_device)
        ae = Util.load_ae(args.name, device="cpu" if args.offload else torch_device)

        rng = torch.Generator(device="cpu")
        opts = SamplingOptions(
            prompt=args.prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=args.guidance,
            seed=args.seed,
        )

        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # prepare input
        x = Sampling.get_noise(
            1,
            opts.height,
            opts.width,
            device=torch_device,
            dtype=dtypes.bfloat16,
            seed=opts.seed,
        )
        opts.seed = None
        if args.offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = Sampling.prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = Sampling.get_schedule(opts.num_steps, inp["img"].shape[1], shift=(args.name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if args.offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # denoise initial noise
        x = Sampling.denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # offload model, load autoencoder to gpu
        if args.offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        x = Sampling.unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)
    t1 = time.perf_counter()

    fn = output_name.format(idx=idx)
    print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = x[0].rearrange("c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    
    img.save(fn, quality=95, subsampling=0)
    idx += 1