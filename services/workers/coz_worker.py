"""Chain-of-Zoom worker (environment wz-coz).

Runs with cwd = PYTHONPATH = external/Chain-of-Zoom (bryanswkim/Chain-of-Zoom @ 42deeda plus
third_party/chain_of_zoom/wonderzoom_coz.py copied there by scripts/setup_third_party.sh coz).
osediff_sd3.py imports 'lora.lora_layers' relative to that directory.

Port of the WonderZoom Chain-of-Zoom service. Op 'super_resolve_dual': the VLM prompt is computed
from the previous and the current (zoomed) image, then the current image is super-resolved at its
native resolution. A seed of None means random.randint(1, 999999), as in the paper-era service.

Smoke test without weights: python coz_worker.py --dry-import
"""
import os
import random

import _wz_protocol as wzp

SERVICE = "coz"


def dry_import(cfg):
    import numpy as np
    import torch
    import osediff_sd3  # noqa: F401
    import wonderzoom_coz  # noqa: F401
    from qwen_vl_utils import process_vision_info  # noqa: F401
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: F401
    return {"torch": torch.__version__, "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(), "cwd": os.getcwd()}


def _looks_like_path(value):
    return isinstance(value, str) and value.startswith(("/", "./", "../", "~"))


class CozWorker:
    def __init__(self, cfg):
        repo_dir = cfg.get("repo_dir")
        if repo_dir:
            os.chdir(repo_dir)
        for name in ("osediff_sd3.py", "wonderzoom_coz.py"):
            if not os.path.isfile(name):
                raise FileNotFoundError(
                    f"{name} not found in {os.getcwd()}; run scripts/setup_third_party.sh coz")

        lora_path = os.path.abspath(os.path.expanduser(cfg["lora_path"]))
        vae_path = os.path.abspath(os.path.expanduser(cfg["vae_path"]))
        missing = [p for p in (lora_path, vae_path) if not os.path.isfile(p)]
        sd3_model = cfg.get("sd3_model") or "stabilityai/stable-diffusion-3-medium-diffusers"
        vlm_model = cfg.get("vlm_model") or "Qwen/Qwen2.5-VL-3B-Instruct"
        missing += [p for p in (sd3_model, vlm_model) if _looks_like_path(p) and not os.path.isdir(os.path.expanduser(p))]
        if missing:
            raise FileNotFoundError(f"Chain-of-Zoom weights not found: {', '.join(missing)}")

        import numpy as np
        import torch
        from PIL import Image
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the Chain-of-Zoom worker "
                               f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
        from wonderzoom_coz import create_model

        self.np, self.torch, self.Image = np, torch, Image
        self.pad_to_multiple_of_16 = bool(cfg.get("pad_to_multiple_of_16", True))
        self.model = create_model(
            lora_path,
            vae_path,
            sd3_model=os.path.expanduser(sd3_model) if _looks_like_path(sd3_model) else sd3_model,
            vlm_model=os.path.expanduser(vlm_model) if _looks_like_path(vlm_model) else vlm_model,
            lora_rank=int(cfg.get("lora_rank", 4)),
            device="cuda",
            pad_to_multiple=16 if self.pad_to_multiple_of_16 else 0,
        )
        self.sd3_model, self.vlm_model = sd3_model, vlm_model

    def set_random_seed(self, seed=None):
        """Seed every RNG; a new random seed is drawn when seed is None (paper behaviour)."""
        np, torch = self.np, self.torch
        if seed is None:
            seed = random.randint(1, 999999)

        print(f"Setting random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return seed

    def suspend(self):
        self.model.suspend()

    def resume(self):
        self.model.resume()

    def info(self):
        torch = self.torch
        return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "sd3_model": self.sd3_model, "vlm_model": self.vlm_model,
                "pad_to_multiple_of_16": self.pad_to_multiple_of_16}

    def op_super_resolve_dual(self, prev_png, cur_png, out_png, prompt=None, seed=None):
        # Never let a stale output from an earlier call look like a result.
        if os.path.exists(out_png) and out_png not in (prev_png, cur_png):
            os.remove(out_png)
        custom_prompt = prompt if prompt else None
        try:
            actual_seed = self.set_random_seed(seed)

            prev_image = self.Image.open(prev_png).convert('RGB')
            current_image = self.Image.open(cur_png).convert('RGB')
            width, height = current_image.size
            if not self.pad_to_multiple_of_16 and (width % 16 or height % 16):
                raise ValueError(
                    f"Chain-of-Zoom needs image sizes divisible by 16, got {width}x{height} "
                    "(enable services.coz.pad_to_multiple_of_16 or use a 720x1088 / 480x720 render)")

            print(f"Running Chain-of-Zoom dual inference with seed {actual_seed}...")
            result = self.model.inference(prev_image, current_image, custom_prompt, actual_seed)

            out_dir = os.path.dirname(out_png)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            result.save(out_png)
            print(f"Dual super-resolution completed: {out_png}")
            return {"out_png": out_png, "seed": actual_seed, "prompt": self.model.last_prompt,
                    "size": [width, height]}
        finally:
            wzp.empty_cuda_cache()


def load(cfg):
    return CozWorker(cfg)


if __name__ == "__main__":
    wzp.run_worker(SERVICE, load, dry_import)
