"""Step1X-Edit worker (environment wz-step1x).

Runs with cwd = PYTHONPATH = external/Step1X-Edit (stepfun-ai/Step1X-Edit @ 4174b28 plus
third_party/step1x_edit/simple_step1x.py copied there by scripts/setup_third_party.sh step1x).

Port of the WonderZoom Step1X-Edit service. Op 'edit': edits an image with a text prompt at
size_level 512 (28 steps, CFG 6.0, seed 42 by default); the output is resized to the input size.
Prompts travel as JSON, so quotes and newlines in GPT prompts are safe.

Smoke test without weights: python step1x_worker.py --dry-import (needs a visible GPU, because
liger_kernel inspects the device at import time).
"""
import os

import _wz_protocol as wzp

SERVICE = "step1x"
DIT_FILENAME = "step1x-edit-i1258.safetensors"
AE_FILENAME = "vae.safetensors"


def dry_import(cfg):
    import flash_attn
    import torch
    import transformers
    import simple_step1x  # noqa: F401
    from transformers.utils import is_flash_attn_2_available
    return {"torch": torch.__version__, "flash_attn": flash_attn.__version__,
            "transformers": transformers.__version__, "flash_attn_2_available": is_flash_attn_2_available(),
            "cuda_available": torch.cuda.is_available(), "cwd": os.getcwd()}


def _looks_like_path(value):
    return isinstance(value, str) and value.startswith(("/", "./", "../", "~"))


class Step1XWorker:
    def __init__(self, cfg):
        repo_dir = cfg.get("repo_dir")
        if repo_dir:
            os.chdir(repo_dir)
        if not os.path.isfile("simple_step1x.py"):
            raise FileNotFoundError(
                f"simple_step1x.py not found in {os.getcwd()}; run scripts/setup_third_party.sh step1x")

        checkpoint_dir = os.path.abspath(os.path.expanduser(cfg["checkpoint_dir"]))
        missing = [name for name in (DIT_FILENAME, AE_FILENAME)
                   if not os.path.isfile(os.path.join(checkpoint_dir, name))]
        qwen_path = cfg.get("qwen_model") or "Qwen/Qwen2.5-VL-7B-Instruct"
        if _looks_like_path(qwen_path):
            qwen_path = os.path.expanduser(qwen_path)
            if not os.path.isdir(qwen_path):
                missing.append(qwen_path)
        if missing:
            raise FileNotFoundError(
                f"Step1X-Edit weights not found in {checkpoint_dir}: {', '.join(missing)}. "
                "Run scripts/download_checkpoints.sh --step1x (or set services.step1x.checkpoint_dir).")

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the Step1X-Edit worker "
                               f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
        from simple_step1x import load_step1x_edit_model

        self.torch = torch
        self.offload = bool(cfg.get("offload", False))
        self.quantized = bool(cfg.get("quantized", False))
        self.qwen_path = qwen_path
        self.checkpoint_dir = checkpoint_dir
        print(f"Loading Step1X-Edit (offload={self.offload}, quantized={self.quantized}, qwen={qwen_path})")
        self.model = load_step1x_edit_model(checkpoint_dir, device="cuda", offload=self.offload,
                                            quantized=self.quantized, qwen_path=qwen_path)

    def suspend(self):
        self.model.suspend()

    def resume(self):
        self.model.resume()

    def info(self):
        torch = self.torch
        return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "checkpoint_dir": self.checkpoint_dir, "qwen_model": self.qwen_path,
                "offload": self.offload, "quantized": self.quantized}

    def op_edit(self, image_png, prompt, out_png, seed=42, num_steps=28, cfg_guidance=6.0, size_level=512):
        # Never let a stale output from an earlier call look like a result.
        if os.path.exists(out_png) and out_png != image_png:
            os.remove(out_png)
        print(f"Starting image editing with prompt: {prompt!r}")
        try:
            result = self.model.generate(
                image=image_png,
                prompt=prompt,
                seed=seed,
                num_steps=num_steps,
                cfg_guidance=cfg_guidance,
                size_level=size_level,
            )
            out_dir = os.path.dirname(out_png)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            result.save(out_png)
            print(f"Image editing completed: {out_png}")
            return {"out_png": out_png, "size": list(result.size), "seed": seed}
        finally:
            wzp.empty_cuda_cache()


def load(cfg):
    return Step1XWorker(cfg)


if __name__ == "__main__":
    wzp.run_worker(SERVICE, load, dry_import)
