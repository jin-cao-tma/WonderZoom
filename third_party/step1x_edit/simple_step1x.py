# Load-once wrapper around Step1X-Edit v1.0.
#
# Adapted from stepfun-ai/Step1X-Edit (https://github.com/stepfun-ai/Step1X-Edit), gradio_app.py and
# inference.py at commit 4174b2855f7d206a580a4710b945211dc7bf8be7. Step1X-Edit is licensed under the
# Apache License, Version 2.0, and so is this file:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations.
#
# Modifications by the WonderZoom authors: configurable Qwen2.5-VL path (qwen_path), FP8 weights
# cast on the fly from the official bf16 checkpoint (as upstream inference.py does), the Qwen encoder
# is moved to the CPU right after loading when offloading, suspend()/resume() for GPU parking, and
# the demo code was removed.
#
# scripts/setup_third_party.sh copies this file into external/Step1X-Edit, next to modules/ and
# sampling.py.
import itertools
import math
import os
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import functional as F

# Step1X-Edit modules live next to this file.
_STEP1X_ROOT = os.path.dirname(os.path.abspath(__file__))
if _STEP1X_ROOT not in sys.path:
    sys.path.insert(0, _STEP1X_ROOT)

from modules.autoencoder import AutoEncoder  # noqa: E402
from modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder  # noqa: E402
from modules.model_edit import Step1XParams, Step1XEdit  # noqa: E402
import sampling  # noqa: E402

DIT_FILENAME = "step1x-edit-i1258.safetensors"
AE_FILENAME = "vae.safetensors"
DEFAULT_QWEN_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"


class SimpleStep1XEdit:
    """
    Simplified interface for the Step1X-Edit model.
    Load once, then call generate() with an image and a prompt.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        offload: bool = False,
        quantized: bool = False,
        qwen_path: str = DEFAULT_QWEN_PATH,
    ):
        """
        Initialize the Step1X-Edit model.

        Args:
            model_path: Directory with step1x-edit-i1258.safetensors and vae.safetensors
            device: Device to run inference on
            dtype: Data type for the Qwen2.5-VL encoder
            offload: Keep the models in host RAM and move each one to the GPU only while it runs
            quantized: Cast the bf16 DiT weights to float8_e4m3fn at load time (as upstream inference.py)
            qwen_path: Qwen2.5-VL-7B-Instruct, HF id or local directory
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.offload = offload
        self.quantized = quantized
        self.qwen_path = qwen_path

        self._load_models(model_path)

    def _load_models(self, model_path: str):
        """Load the VAE, the DiT and the text encoder."""

        # Load VAE
        with torch.device("meta"):
            self.ae = AutoEncoder(
                resolution=256,
                in_channels=3,
                ch=128,
                out_ch=3,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                z_channels=16,
                scale_factor=0.3611,
                shift_factor=0.1159,
            )

        ae_path = os.path.join(model_path, AE_FILENAME)
        self.ae = self._load_state_dict(self.ae, ae_path, 'cpu')
        self.ae = self.ae.to(dtype=torch.float32)

        # Load DiT
        with torch.device("meta"):
            step1x_params = Step1XParams(
                in_channels=64,
                out_channels=64,
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
                mode="flash"
            )
            self.dit = Step1XEdit(step1x_params)

        # The FP8 variant is cast from the official bf16 checkpoint, as upstream inference.py does.
        dit_path = os.path.join(model_path, DIT_FILENAME)
        self.dit = self._load_state_dict(self.dit, dit_path, 'cpu')

        if not self.quantized:
            self.dit = self.dit.to(dtype=torch.bfloat16)
        else:
            self.dit = self.dit.to(dtype=torch.float8_e4m3fn)

        # Text encoder (Qwen2.5-VL-7B, HF id or local directory). It is always created on the GPU
        # (modules/conditioner.py), so it is moved to the CPU right away when offloading.
        self.llm_encoder = Qwen2VLEmbedder(
            self.qwen_path,
            device=self.device,
            max_length=640,
            dtype=self.dtype,
        )
        if self.offload:
            self.llm_encoder = self.llm_encoder.cpu()
            torch.cuda.empty_cache()

        # Move models to device if not offloading
        if not self.offload:
            self.dit = self.dit.to(device=self.device)
            self.ae = self.ae.to(device=self.device)

    def _load_state_dict(self, model, ckpt_path, device="cuda", strict=False, assign=True):
        """Load model state dict from checkpoint."""
        if Path(ckpt_path).suffix == ".safetensors":
            state_dict = load_file(ckpt_path, device)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        missing, unexpected = model.load_state_dict(
            state_dict, strict=strict, assign=assign
        )
        return model

    def _load_image(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert various image formats to tensor."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, Image.Image):
            image = F.to_tensor(image.convert("RGB"))
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, str):
            image = F.to_tensor(Image.open(image).convert("RGB"))
            image = image.unsqueeze(0)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _input_process_image(self, img: Image.Image, img_size: int = 512):
        """Process input image to appropriate size."""
        w, h = img.size
        r = w / h

        if w > h:
            w_new = math.ceil(math.sqrt(img_size * img_size * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(img_size * img_size / r))
            w_new = math.ceil(h_new * r)
        h_new = math.ceil(h_new) // 16 * 16
        w_new = math.ceil(w_new) // 16 * 16

        img_resized = img.resize((w_new, h_new))
        return img_resized, img.size

    def _prepare(self, prompt: Union[str, List[str]], img: torch.Tensor, ref_image: torch.Tensor, ref_image_raw: torch.Tensor):
        """Prepare inputs for the model."""
        bs, _, h, w = img.shape
        bs, _, ref_h, ref_w = ref_image.shape

        assert h == ref_h and w == ref_w

        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        elif bs >= 1 and isinstance(prompt, str):
            prompt = [prompt] * bs

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_image, "b c (ref_h ph) (ref_w pw) -> b (ref_h ref_w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)
        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "ref_h ref_w c -> b (ref_h ref_w) c", b=bs)

        if isinstance(prompt, str):
            prompt = [prompt]

        if self.offload:
            self.llm_encoder = self.llm_encoder.to(self.device)
        txt, mask = self.llm_encoder(prompt, ref_image_raw)
        if self.offload:
            self.llm_encoder = self.llm_encoder.cpu()
            torch.cuda.empty_cache()

        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)

        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),
            "llm_embedding": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
        }

    @staticmethod
    def _process_diff_norm(diff_norm, k):
        """Process difference norm for CFG."""
        pow_result = torch.pow(diff_norm, k)
        result = torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )
        return result

    def _denoise(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: List[float],
        cfg_guidance: float = 6.0,
        mask=None,
        timesteps_truncate: float = 1.0,
    ):
        """Denoise the image using the diffusion model."""
        if self.offload:
            self.dit = self.dit.to(self.device)

        for t_curr, t_prev in itertools.pairwise(timesteps):
            if img.shape[0] == 1 and cfg_guidance != -1:
                img = torch.cat([img, img], dim=0)
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )

            pred = self.dit(
                img=img,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=t_vec,
                llm_embedding=llm_embedding,
                t_vec=t_vec,
                mask=mask,
            )

            if cfg_guidance != -1:
                cond, uncond = (
                    pred[0 : pred.shape[0] // 2, :],
                    pred[pred.shape[0] // 2 :, :],
                )
                if t_curr > timesteps_truncate:
                    diff = cond - uncond
                    diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                    pred = uncond + cfg_guidance * (
                        cond - uncond
                    ) / self._process_diff_norm(diff_norm, k=0.4)
                else:
                    pred = uncond + cfg_guidance * (cond - uncond)

            tem_img = img[0 : img.shape[0] // 2, :] + (t_prev - t_curr) * pred
            img_input_length = img.shape[1] // 2
            img = torch.cat(
                [
                tem_img[:, :img_input_length],
                img[ : img.shape[0] // 2, img_input_length:],
                ], dim=1
            )

        if self.offload:
            self.dit = self.dit.cpu()
            torch.cuda.empty_cache()

        return img[:, :img.shape[1] // 2]

    @staticmethod
    def _unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Unpack tensor to image format."""
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @torch.inference_mode()
    def generate(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 28,
        cfg_guidance: float = 6.0,
        seed: int = 42,
        size_level: int = 512,
    ) -> Image.Image:
        """
        Generate an edited image from an input image and a prompt.

        Args:
            image: Input image to edit
            prompt: Text prompt describing the desired edit
            negative_prompt: Negative prompt (optional)
            num_steps: Number of diffusion steps
            cfg_guidance: CFG guidance strength
            seed: Random seed
            size_level: Generation size level (the image is processed at about size_level^2 pixels)

        Returns:
            Edited PIL Image, resized back to the input size
        """
        # Process input image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            # Convert tensor/array to PIL
            if isinstance(image, torch.Tensor):
                image = F.to_pil_image(image.squeeze(0) if image.dim() == 4 else image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8))

        ref_images_raw, img_info = self._input_process_image(image, img_size=size_level)
        width, height = ref_images_raw.width, ref_images_raw.height

        # Load and encode reference image
        ref_images_raw_tensor = self._load_image(ref_images_raw).to(self.device)

        # Set random seed
        if seed < 0:
            seed = torch.Generator(device="cpu").seed()

        if self.offload:
            self.ae = self.ae.to(self.device)
        # AutoEncoder.encode samples the posterior with torch.randn_like on the global CUDA RNG,
        # which upstream never seeds; seed it from the request seed so edits are reproducible.
        with torch.random.fork_rng(devices=[torch.device(self.device).index or 0]):
            torch.manual_seed(seed)
            ref_images = self.ae.encode(ref_images_raw_tensor * 2 - 1)
        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()

        # Generate random noise
        x = torch.randn(
            1,
            16,
            height // 8,
            width // 8,
            device=self.device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        # Get timestep schedule
        timesteps = sampling.get_schedule(
            num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True
        )

        # Prepare for CFG
        x = torch.cat([x, x], dim=0)
        ref_images = torch.cat([ref_images, ref_images], dim=0)
        ref_images_raw_tensor = torch.cat([ref_images_raw_tensor, ref_images_raw_tensor], dim=0)

        # Prepare inputs
        inputs = self._prepare([prompt, negative_prompt], x, ref_image=ref_images, ref_image_raw=ref_images_raw_tensor)

        # Denoise
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self._denoise(
                **inputs,
                cfg_guidance=cfg_guidance,
                timesteps=timesteps,
                timesteps_truncate=1.0,
            )

        # Unpack and decode
        x = self._unpack(x.float(), height, width)
        if self.offload:
            self.ae = self.ae.to(self.device)
        x = self.ae.decode(x)
        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()

        x = x.clamp(-1, 1)
        x = x.mul(0.5).add(0.5)

        # Convert to PIL Image
        result_image = F.to_pil_image(x[0].float())
        result_image = result_image.resize(img_info)

        return result_image

    # ---- GPU parking (used by the WonderZoom GPU arbiter) ----

    def suspend(self):
        """Move every model to host RAM and release the cached GPU memory."""
        self.dit = self.dit.cpu()
        self.ae = self.ae.cpu()
        self.llm_encoder = self.llm_encoder.cpu()
        torch.cuda.empty_cache()

    def resume(self):
        """Restore the load-time placement: everything on the GPU unless offloading."""
        if not self.offload:
            self.llm_encoder = self.llm_encoder.to(self.device)
            self.dit = self.dit.to(device=self.device)
            self.ae = self.ae.to(device=self.device)


def load_step1x_edit_model(model_path: str, **kwargs) -> SimpleStep1XEdit:
    """
    Convenience function to load the Step1X-Edit model.

    Args:
        model_path: Directory with step1x-edit-i1258.safetensors and vae.safetensors
        **kwargs: Additional arguments for SimpleStep1XEdit (device, dtype, offload, quantized, qwen_path)

    Returns:
        Loaded Step1X-Edit model
    """
    return SimpleStep1XEdit(model_path, **kwargs)
