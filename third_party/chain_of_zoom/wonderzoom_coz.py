# One-step Chain-of-Zoom super-resolution used by WonderZoom for zoom-in.
#
# Built on bryanswkim/Chain-of-Zoom (MIT License, Copyright (c) 2025 Bryan Sangwoo Kim) @ 42deeda:
# OSEDiff-SD3 one-step SR (osediff_sd3.py) prompted by Qwen2.5-VL-3B-Instruct with the upstream
# 'recursive_multiscale' system prompt. Each call performs exactly one zoom step: the VLM looks at
# the previous scale and the current (zoomed) image, and the SR network restores the current image
# at its native resolution.
#
# scripts/setup_third_party.sh copies this file into external/Chain-of-Zoom, next to osediff_sd3.py.
# The WonderZoom worker (services/workers/coz_worker.py) runs it with cwd and PYTHONPATH set to that
# directory.
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# System prompt of upstream inference_coz.py ('recursive_multiscale').
VLM_SYSTEM_PROMPT = ("The second image is a zoom-in of the first image. Based on this knowledge, "
                     "what is in the second image? Give me a set of words.")
DEFAULT_PROMPT = "high quality, detailed image"


class SimpleChainZoomModel:
    """Chain-of-Zoom model for single-step inference (one zoom step per call)."""

    def __init__(self,
                 lora_path: str,
                 vae_path: str,
                 sd3_model: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 lora_rank: int = 4,
                 device: str = "cuda",
                 pad_to_multiple: int = 16):
        """
        Args:
            lora_path: SR LoRA weights (ckpt/SR_LoRA/model_20001.pkl).
            vae_path: SR VAE-encoder weights (ckpt/SR_VAE/vae_encoder_20001.pt).
            sd3_model: Stable Diffusion 3 Medium (diffusers format), HF id or local directory.
            vlm_model: Qwen2.5-VL-3B-Instruct, HF id or local directory.
            lora_rank: LoRA rank of the SR checkpoints.
            device: Device for all models.
            pad_to_multiple: The SD3 one-step SR needs H and W divisible by 16. Inputs of other
                sizes are reflect-padded to this multiple and the output is cropped back.
                0 disables padding (such inputs then fail inside the SR network).
        """
        self.device = device
        self.lora_rank = lora_rank
        self.pad_to_multiple = int(pad_to_multiple or 0)
        self.last_prompt = None

        self.sr_model = None
        self.vlm_model = None
        self.vlm_processor = None

        self._load_sr_model(lora_path, vae_path, sd3_model)
        self._load_vlm_model(vlm_model)

        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _load_sr_model(self, lora_path: str, vae_path: str, sd3_model: str):
        """Load the OSEDiff-SD3 one-step SR model."""
        from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler

        print("Loading SR model...")

        base_model = SD3Euler(model_key=sd3_model, device=self.device)
        base_model.text_enc_1.to(self.device)
        base_model.text_enc_2.to(self.device)
        base_model.text_enc_3.to(self.device)
        base_model.transformer.to(self.device, dtype=torch.float32)
        base_model.vae.to(self.device, dtype=torch.float32)

        # Inference only.
        for p in [base_model.text_enc_1, base_model.text_enc_2, base_model.text_enc_3,
                  base_model.transformer, base_model.vae]:
            p.requires_grad_(False)

        class Args:
            def __init__(self):
                self.lora_path = lora_path
                self.vae_path = vae_path
                self.lora_rank = 0

        args = Args()
        args.lora_path = lora_path
        args.vae_path = vae_path
        args.lora_rank = self.lora_rank

        self.sr_model = OSEDiff_SD3_TEST(args, base_model)
        print("SR model loaded.")

    def _load_vlm_model(self, vlm_model: str):
        """Load the Qwen2.5-VL prompt model.

        The original wrapper used device_map='auto'. Loading it in full and moving it with .to()
        gives the same placement when the model fits on the GPU, and lets suspend()/resume() move it.
        """
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"Loading VLM model: {vlm_model}")
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model)
        print("VLM model loaded.")

    def _generate_prompt_from_images(self, prev_image: Image.Image, current_image: Image.Image) -> str:
        """Ask the VLM what is in the zoomed image, given the previous scale.

        Args:
            prev_image: previous image (before zooming in)
            current_image: current image (zoomed in)

        Returns:
            The generated prompt.
        """
        try:
            from qwen_vl_utils import process_vision_info

            # PIL images are passed directly (qwen_vl_utils accepts them); the original wrapper
            # round-tripped them through temporary PNG files, which is lossless for RGB images.
            messages = [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": prev_image},
                        {"type": "image", "image": current_image}
                    ]
                }
            ]

            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(self.vlm_model.device)

            generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vlm_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

        except Exception as e:
            # Same fallback as the original wrapper, but the error is no longer silent.
            import traceback
            print(f"Error generating prompt, using the default prompt: {e}")
            traceback.print_exc()
            return DEFAULT_PROMPT

    def _super_resolve(self, current_image: Image.Image, prompt: str) -> Image.Image:
        """One-step SR of current_image at its native resolution."""
        lq = self.tensor_transforms(current_image).unsqueeze(0).to(self.device)
        lq = lq * 2 - 1  # to [-1, 1]

        H, W = lq.shape[-2:]
        m = self.pad_to_multiple
        pad_h = (-H) % m if m else 0
        pad_w = (-W) % m if m else 0
        if pad_h or pad_w:
            # Reflect-pad to a multiple of 16 (never needed at 720x1088 or 480x720).
            lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            output_image = self.sr_model(lq, prompt=prompt)
            if pad_h or pad_w:
                output_image = output_image[..., :H, :W]
            output_image = torch.clamp(output_image[0].cpu(), -1.0, 1.0)
            output_pil = transforms.ToPILImage()(output_image * 0.5 + 0.5)

        return output_pil

    @staticmethod
    def _seed_everything(seed: int):
        import random

        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def inference(self,
                  prev_image: Union[Image.Image, str],
                  current_image: Union[Image.Image, str],
                  custom_prompt: Optional[str] = None,
                  seed: Optional[int] = None) -> Image.Image:
        """
        Run one zoom step.

        Args:
            prev_image: previous image (PIL.Image or file path)
            current_image: current image (PIL.Image or file path)
            custom_prompt: optional prompt; the VLM prompt is used when None
            seed: optional random seed

        Returns:
            The super-resolved current image (same size as the input).
        """
        if seed is not None:
            print(f"Setting random seed for inference: {seed}")
            self._seed_everything(seed)

        if isinstance(prev_image, str):
            prev_image = Image.open(prev_image).convert('RGB')
        if isinstance(current_image, str):
            current_image = Image.open(current_image).convert('RGB')

        if custom_prompt is None:
            prompt = self._generate_prompt_from_images(prev_image, current_image)
        else:
            prompt = custom_prompt

        print(f"Generated prompt: {prompt}")
        self.last_prompt = prompt

        return self._super_resolve(current_image, prompt)

    def inference_simple(self,
                         current_image: Union[Image.Image, str],
                         prompt: str = DEFAULT_PROMPT,
                         seed: Optional[int] = None) -> Image.Image:
        """
        Single-image SR with a given prompt (no VLM).

        Args:
            current_image: current image (PIL.Image or file path)
            prompt: prompt
            seed: optional random seed

        Returns:
            The super-resolved image.
        """
        if seed is not None:
            print(f"Setting random seed for simple inference: {seed}")
            self._seed_everything(seed)

        if isinstance(current_image, str):
            current_image = Image.open(current_image).convert('RGB')

        self.last_prompt = prompt
        return self._super_resolve(current_image, prompt)

    # ---- GPU parking (used by the WonderZoom GPU arbiter) ----

    def _gpu_modules(self):
        base = self.sr_model.model
        return [base.text_enc_1, base.text_enc_2, base.text_enc_3, base.transformer, base.vae, self.vlm_model]

    def suspend(self):
        """Move every model to host RAM and release the cached GPU memory."""
        for module in self._gpu_modules():
            module.to("cpu")
        torch.cuda.empty_cache()

    def resume(self):
        """Move every model back to the GPU (dtypes are unchanged)."""
        for module in self._gpu_modules():
            module.to(self.device)


def create_model(lora_path: str,
                 vae_path: str,
                 sd3_model: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 lora_rank: int = 4,
                 device: str = "cuda",
                 pad_to_multiple: int = 16) -> SimpleChainZoomModel:
    """Create the model (see SimpleChainZoomModel for the arguments)."""
    return SimpleChainZoomModel(
        lora_path=lora_path,
        vae_path=vae_path,
        sd3_model=sd3_model,
        vlm_model=vlm_model,
        lora_rank=lora_rank,
        device=device,
        pad_to_multiple=pad_to_multiple,
    )
