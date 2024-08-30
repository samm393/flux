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

from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder

#####################################################################################################################

class Util:
    @dataclass
    class ModelSpec:
        params: FluxParams
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
            params=FluxParams(
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
            params=FluxParams(
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
            model = Flux(Util.configs[name].params).to(torch.bfloat16)

        if ckpt_path is not None:
            print("Loading checkpoint")
            # load_sft doesn't support torch.device
            sd = load_sft(ckpt_path, device=str(device))
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            Util.print_load_warning(missing, unexpected)
        return model


    def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
        # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
        return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


    def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
        return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


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


    def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
        bs, c, h, w = img.shape
        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        if isinstance(prompt, str):
            prompt = [prompt]
        txt = t5(prompt)
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        vec = clip(prompt)
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)

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
        timesteps = torch.linspace(1, 0, num_steps + 1)

        # shifting the schedule to favor high timesteps for higher signal images
        if shift:
            # estimate mu based on linear estimation between two points
            mu = Sampling.get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            timesteps = Sampling.time_shift(mu, 1.0, timesteps)

        return timesteps.tolist()


    def denoise(
        model: Flux,
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
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
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
        return rearrange(
            x,
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

    torch_device = torch.device(args.device)
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

    with torch.inference_mode():
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
            dtype=torch.bfloat16,
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
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    
    img.save(fn, quality=95, subsampling=0)
    idx += 1