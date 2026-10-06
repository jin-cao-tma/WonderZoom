"""Gen3C worker (environment wz-gen3c).

Runs with cwd = PYTHONPATH = external/GEN3C (nv-tlabs/GEN3C @ 2b50e0b, unmodified). The working
directory matters: the pipeline loads the relative config file cosmos_predict1/diffusion/config/config.py.

Port of the WonderZoom Gen3C service. Gen3cPipeline is built directly with the arguments the
original service passed through Gen3cPersistentModel (guidance 1.0, 704x1280, 24 fps, 121 frames,
guardrail and prompt upsampler disabled, seed 1), without MoGe, which WonderZoom never used.
The guardrail stays disabled unless services.gen3c.disable_guardrail is false; see the NVIDIA Open
Model License note in THIRD_PARTY_LICENSES.md.

Op 'generate': a condition image, up to 121 warped frames and as many hole masks (holes = 255)
are resized to 1280x704; fewer than 121 frames are padded by repeating the last frame and mask,
as in the original service. Gen3C returns 121 frames, written to <out_dir>/frames/frame_%08d.png
and <out_dir>/gen3c_video.mp4.

Smoke test without weights: python gen3c_worker.py --dry-import
"""
import contextlib
import os
import sys

import _wz_protocol as wzp

SERVICE = "gen3c"
CHECKPOINT_NAME = "Gen3C-Cosmos-7B"
TOKENIZER_DIR = "Cosmos-Tokenize1-CV8x8x8-720p"
T5_DIR = os.path.join("google-t5", "t5-11b")
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
TEXT_CACHE_MAX_ENTRIES = 256
# Guardrail models, under checkpoint_dir, where GEN3C's guardrail code looks for them.
GUARDRAIL_DIRS = (os.path.join("nvidia", "Cosmos-Guardrail1"), os.path.join("meta-llama", "Llama-Guard-3-8B"))


def dry_import(cfg):
    import cv2
    import numpy as np
    import torch
    from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline  # noqa: F401
    from cosmos_predict1.utils import misc  # noqa: F401
    from cosmos_predict1.utils.io import save_video  # noqa: F401
    import transformer_engine  # noqa: F401
    import amp_C  # noqa: F401
    return {"torch": torch.__version__, "numpy": np.__version__, "cv2": cv2.__version__,
            "cuda_available": torch.cuda.is_available(), "cwd": os.getcwd()}


def check_checkpoints(checkpoint_dir, checkpoint_name):
    """Fail at start-up (fatal) instead of at the first request when weights are missing."""
    required = [
        os.path.join(checkpoint_name, "model.pt"),
        os.path.join(TOKENIZER_DIR, "encoder.jit"),
        os.path.join(TOKENIZER_DIR, "decoder.jit"),
        os.path.join(TOKENIZER_DIR, "mean_std.pt"),
        os.path.join(TOKENIZER_DIR, "image_mean_std.pt"),
        os.path.join(T5_DIR, "config.json"),
    ]
    missing = [rel for rel in required if not os.path.isfile(os.path.join(checkpoint_dir, rel))]
    t5_dir = os.path.join(checkpoint_dir, T5_DIR)
    weights = ("pytorch_model.bin", "pytorch_model.bin.index.json", "model.safetensors", "model.safetensors.index.json")
    if not any(os.path.isfile(os.path.join(t5_dir, w)) for w in weights):
        missing.append(os.path.join(T5_DIR, "pytorch_model.bin"))
    if missing:
        raise FileNotFoundError(
            f"Gen3C checkpoints are incomplete in {checkpoint_dir}; missing: {', '.join(missing)}. "
            "Run scripts/download_checkpoints.sh --gen3c (or set services.gen3c.checkpoint_dir).")


def check_guardrail_checkpoints(checkpoint_dir):
    """With the guardrail enabled, fail at start-up when its models are not downloaded."""
    missing = [rel for rel in GUARDRAIL_DIRS if not os.path.isdir(os.path.join(checkpoint_dir, rel))]
    if missing:
        raise FileNotFoundError(
            f"services.gen3c.disable_guardrail is false, but the guardrail models are missing in "
            f"{checkpoint_dir}: {', '.join(missing)}. Download them into that directory (see "
            "docs/INSTALL.md, 'Gen3C guardrail'), or set services.gen3c.disable_guardrail: true.")


# ---- image loading, ported from the original service ----

def list_images(frames_dir):
    image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
    image_files.sort()
    return image_files


def load_and_resize_image(cv2, np, image_path, target_size=(1280, 704)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"cannot read image {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img.astype(np.float32) / 255.0


def load_and_resize_frames(cv2, np, frames_dir, target_size=(1280, 704)):
    frames = []
    for file in list_images(frames_dir):
        frame_path = os.path.join(frames_dir, file)
        frame = load_and_resize_image(cv2, np, frame_path, target_size)
        frames.append(frame)
    return np.stack(frames, axis=0)


def _pipeline_class(Gen3cPipeline, torch):
    class WonderZoomGen3cPipeline(Gen3cPipeline):
        """Gen3cPipeline that can move the T5 encoder to host RAM right after it is created.

        CosmosT5TextEncoder always builds the encoder on the GPU. Parking it before the DiT is
        loaded keeps the start-up peak at about 20 GB instead of about 35 GB. Weights, call order
        and outputs are unchanged.
        """

        def __init__(self, *args, park_text_encoder_on_load=False, **kwargs):
            self._wz_park_t5_on_load = park_text_encoder_on_load
            super().__init__(*args, **kwargs)
            self._wz_park_t5_on_load = False

        def _load_text_encoder_model(self):
            super()._load_text_encoder_model()
            t5 = getattr(self.text_encoder, "text_encoder", None)
            if getattr(self, "_wz_park_t5_on_load", False) and t5 is not None:
                t5.to("cpu")
                torch.cuda.empty_cache()

    return WonderZoomGen3cPipeline


class Gen3cWorker:
    def __init__(self, cfg):
        repo_dir = cfg.get("repo_dir")
        if repo_dir:
            os.chdir(repo_dir)
        config_file = os.path.join("cosmos_predict1", "diffusion", "config", "config.py")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"{os.getcwd()} is not a GEN3C checkout (missing {config_file}); "
                "run scripts/setup_third_party.sh gen3c")

        self.checkpoint_dir = os.path.abspath(os.path.expanduser(cfg["checkpoint_dir"]))
        self.checkpoint_name = cfg.get("checkpoint_name") or CHECKPOINT_NAME
        check_checkpoints(self.checkpoint_dir, self.checkpoint_name)

        # Guardrail: disabled by default (null = default), as in the paper runs, the original
        # WonderZoom service and GEN3C's GUI server. offload_guardrail_models only matters with the
        # guardrail on: true loads the guardrail models for each check and frees them afterwards,
        # false keeps them on the GPU (they are not parked under the exclusive policy).
        disable_guardrail = cfg.get("disable_guardrail")
        self.disable_guardrail = True if disable_guardrail is None else bool(disable_guardrail)
        offload_guardrail = cfg.get("offload_guardrail_models")
        self.offload_guardrail_models = (not self.disable_guardrail
                                         and (True if offload_guardrail is None else bool(offload_guardrail)))
        if not self.disable_guardrail:
            check_guardrail_checkpoints(self.checkpoint_dir)
        print(f"[{SERVICE}] guardrail {'disabled' if self.disable_guardrail else 'enabled'} "
              f"(services.gen3c.disable_guardrail={self.disable_guardrail}, "
              f"offload_guardrail_models={self.offload_guardrail_models})", file=sys.stderr, flush=True)

        import cv2
        import numpy as np
        import torch
        from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
        from cosmos_predict1.utils import misc
        from cosmos_predict1.utils.io import save_video

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the Gen3C worker "
                               f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
        self.cv2, self.np, self.torch, self.misc, self.save_video = cv2, np, torch, misc, save_video
        self.device = torch.device("cuda")

        # Paper settings: gen3c_persistent defaults plus the arguments of the original service,
        # except the default step count (below).
        self.seed = int(cfg.get("seed", 1))
        request_seed = cfg.get("request_seed")
        self.request_seed = None if request_seed is None else int(request_seed)
        self.height = int(cfg.get("height", 704))
        self.width = int(cfg.get("width", 1280))
        self.fps = int(cfg.get("fps", 24))
        self.num_video_frames = int(cfg.get("num_video_frames", 121))
        self.guidance = float(cfg.get("guidance", 1.0))
        # Steps for a request without num_steps. NOT the original service's default: it started
        # its pipeline with --num_steps 3, which no paper call used (every paper call passes
        # num_steps: 18 for camera moves, 10 for HQ views).
        self.default_num_steps = int(cfg.get("num_steps", 18))
        # prompt None -> the literal 'None' the paper-era REPL f-string produced (False: '').
        self.legacy_none_prompt = bool(cfg.get("legacy_none_prompt", True))
        self.video_quality = int(cfg.get("video_quality", 5))
        self.minmax_normalize_frames = bool(cfg.get("minmax_normalize_frames", True))

        self.offload_network = bool(cfg.get("offload_network", False))
        self.offload_tokenizer = bool(cfg.get("offload_tokenizer", False))
        self.offload_text_encoder_model = bool(cfg.get("offload_text_encoder_model", False))
        self.cache_text_embeddings = bool(cfg.get("cache_text_embeddings", True))
        self.park_text_encoder = bool(cfg.get("park_text_encoder", False))
        self.park_tokenizer = bool(cfg.get("park_tokenizer", False))

        pipeline_cls = _pipeline_class(Gen3cPipeline, torch)
        # Gen3cPersistentModel.__init__ ran under torch.no_grad() and seeded every RNG first.
        with torch.no_grad():
            misc.set_random_seed(self.seed)
            self.pipeline = pipeline_cls(
                inference_type="video2world",
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_name=self.checkpoint_name,
                prompt_upsampler_dir="Pixtral-12B",
                enable_prompt_upsampler=False,
                offload_network=self.offload_network,
                offload_tokenizer=self.offload_tokenizer,
                offload_text_encoder_model=self.offload_text_encoder_model,
                offload_prompt_upsampler=True,
                offload_guardrail_models=self.offload_guardrail_models,
                disable_guardrail=self.disable_guardrail,
                guidance=self.guidance,
                num_steps=self.default_num_steps,
                height=self.height,
                width=self.width,
                fps=self.fps,
                num_video_frames=self.num_video_frames,
                seed=self.seed,
                park_text_encoder_on_load=self.park_text_encoder and not self.offload_text_encoder_model,
            )

        self._t5_parked = self._t5_module() is not None and self.park_text_encoder
        self._t5_suspended = False
        self._tokenizer_suspended = False
        self._text_cache = {}
        if self.cache_text_embeddings:
            self._install_text_embedding_cache()
        wzp.release_cuda_memory()

    # ---- T5 handling ----

    def _t5_module(self):
        """The HF T5EncoderModel inside CosmosT5TextEncoder (None when offloaded or disabled)."""
        return getattr(getattr(self.pipeline, "text_encoder", None), "text_encoder", None)

    @contextlib.contextmanager
    def _t5_on_gpu(self):
        """Bring a parked T5 encoder to the GPU for one encoding call, then park it again."""
        t5 = self._t5_module()
        if t5 is not None and self._t5_parked:
            t5.to(self.device)
            self._t5_parked = False
        try:
            yield
        finally:
            t5 = self._t5_module()
            if t5 is not None and self.park_text_encoder and not self._t5_parked:
                t5.to("cpu")
                self._t5_parked = True
                wzp.release_cuda_memory()

    def _install_text_embedding_cache(self):
        """Cache T5 embeddings per prompt on the CPU.

        Wraps Gen3cPipeline._run_text_embedding_on_prompt_with_offload on this instance only; the
        cached tensors are exact copies, so results do not change. A prompt is encoded by T5 once.
        """
        pipeline = self.pipeline
        original = pipeline._run_text_embedding_on_prompt_with_offload
        cache = self._text_cache
        torch = self.torch
        device = self.device

        def cached_text_embedding(prompts, **kwargs):
            if kwargs:
                with self._t5_on_gpu():
                    return original(prompts, **kwargs)
            missing = [p for p in dict.fromkeys(prompts) if p not in cache]
            if missing:
                with self._t5_on_gpu():
                    embeddings, masks = original(missing)
                for prompt, embedding, mask in zip(missing, embeddings, masks):
                    if len(cache) >= TEXT_CACHE_MAX_ENTRIES:
                        cache.pop(next(iter(cache)))
                    cache[prompt] = (embedding.detach().to("cpu", copy=True), mask.detach().to("cpu", copy=True))
            with torch.no_grad():
                embeddings = [cache[p][0].to(device) for p in prompts]
                masks = [cache[p][1].to(device) for p in prompts]
            return embeddings, masks

        pipeline._run_text_embedding_on_prompt_with_offload = cached_text_embedding

    # ---- GPU parking ----

    def suspend(self):
        net = getattr(self.pipeline.model, "model", None)  # DiT + conditioner (None when offloaded)
        if net is not None:
            net.to("cpu")
        t5 = self._t5_module()
        if t5 is not None and not self._t5_parked:
            t5.to("cpu")
            self._t5_suspended = True
        tokenizer = getattr(self.pipeline.model, "tokenizer", None)
        if self.park_tokenizer and tokenizer is not None:
            tokenizer.to("cpu")
            self._tokenizer_suspended = True

    def resume(self):
        net = getattr(self.pipeline.model, "model", None)
        if net is not None:
            net.to(self.device)
        t5 = self._t5_module()
        if t5 is not None and self._t5_suspended:
            t5.to(self.device)
        self._t5_suspended = False
        tokenizer = getattr(self.pipeline.model, "tokenizer", None)
        if self._tokenizer_suspended and tokenizer is not None:
            tokenizer.to(self.device)
        self._tokenizer_suspended = False

    def info(self):
        torch = self.torch
        return {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "checkpoint_dir": self.checkpoint_dir,
            "size": [self.width, self.height],
            "num_video_frames": self.num_video_frames,
            "default_num_steps": self.default_num_steps,
            "cache_text_embeddings": self.cache_text_embeddings,
            "park_text_encoder": self.park_text_encoder,
            "offload": {"network": self.offload_network, "tokenizer": self.offload_tokenizer,
                        "text_encoder": self.offload_text_encoder_model},
        }

    # ---- op 'generate' ----

    def _write_frames(self, video, frames_output_dir):
        """Write the generated frames as PNG files, optionally with the paper-era min-max stretch."""
        np, cv2 = self.np, self.cv2
        os.makedirs(frames_output_dir, exist_ok=True)
        for name in os.listdir(frames_output_dir):  # never mix in frames of an earlier request
            if name.startswith("frame_") and name.endswith(".png"):
                os.remove(os.path.join(frames_output_dir, name))

        normalize = self.minmax_normalize_frames
        vmin, vmax = video.min(), video.max()
        if normalize and vmax == vmin:
            print("Constant video: skipping the min-max normalization", file=sys.stderr)
            normalize = False
        if normalize:
            # Same arithmetic as the original '(video - video.min()) / (video.max() - video.min())'
            # on the whole uint8 video, done frame by frame to avoid a 2.6 GB float64 copy.
            scale = vmax - vmin
        for i in range(video.shape[0]):
            frame_path = os.path.join(frames_output_dir, f"frame_{i:08d}.png")
            if normalize:
                frame_np = (video[i] - vmin) / scale
                frame_np = np.clip(frame_np, 0, 1)
                frame_uint8 = (frame_np * 255).astype(np.uint8)
            else:
                frame_uint8 = video[i]
            # cv2 expects BGR
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(frame_path, frame_bgr):
                raise IOError(f"cannot write {frame_path}")
        return video.shape[0]

    def op_generate(self, condition_image, frames_dir, masks_dir, prompt="", out_dir=None,
                    num_steps=None, seed=None):
        np, torch = self.np, self.torch
        if out_dir is None:
            raise ValueError("out_dir is required")
        if prompt is None:
            prompt = "None" if self.legacy_none_prompt else ""
        target_size = (self.width, self.height)

        # Gen3C generates exactly one chunk of num_video_frames frames: the tokenizer encodes
        # 121-frame chunks and the DiT state expects a single chunk. Fewer frames are padded below
        # (as in the original service); more cannot work.
        n_frames, n_masks = len(list_images(frames_dir)), len(list_images(masks_dir))
        if n_frames != n_masks or not 1 <= n_frames <= self.num_video_frames:
            raise ValueError(
                f"Gen3C needs 1 to {self.num_video_frames} warped frames with one mask each, got "
                f"{n_frames} frames in {frames_dir} and {n_masks} masks in {masks_dir}")

        if seed is None:
            seed = self.request_seed
        if seed is not None:
            # Optional per-request seed: reseeds the global RNGs here (initial noise) and, below,
            # the pipeline seed used for the condition-augmentation noise. By default the
            # process-global RNG (seeded once at start-up) is used, as in the paper-era service.
            seed = int(seed)
            self.misc.set_random_seed(seed)

        print(f"Starting video generation with prompt: '{prompt}'")
        try:
            condition_image = load_and_resize_image(self.cv2, np, condition_image, target_size)
            condition_images = condition_image[np.newaxis, ...]

            rendered_warp_images = load_and_resize_frames(self.cv2, np, frames_dir, target_size)
            rendered_warp_masks = load_and_resize_frames(self.cv2, np, masks_dir, target_size)

            print(f"   Loaded {rendered_warp_images.shape[0]} frames")

            if rendered_warp_masks.shape[-1] == 3:
                rendered_warp_masks = np.mean(rendered_warp_masks, axis=-1, keepdims=True)
            rendered_warp_masks = np.clip(rendered_warp_masks, 0, 1)

            # WonderZoom writes holes as 255; Gen3C expects 1 = valid pixel.
            rendered_warp_masks = 1.0 - rendered_warp_masks

            T = rendered_warp_images.shape[0]
            if T < self.num_video_frames:
                # Same as the original service: repeat the last warped frame and mask.
                print(f"Adjusting frame count from {T} to {self.num_video_frames}")
                pad = self.num_video_frames - T
                rendered_warp_images = np.concatenate(
                    [rendered_warp_images, np.repeat(rendered_warp_images[-1:], pad, axis=0)], axis=0)
                rendered_warp_masks = np.concatenate(
                    [rendered_warp_masks, np.repeat(rendered_warp_masks[-1:], pad, axis=0)], axis=0)

            rendered_warp_images = rendered_warp_images[np.newaxis, :, np.newaxis, :, :, :]
            rendered_warp_images = rendered_warp_images.transpose(0, 1, 2, 5, 3, 4)
            rendered_warp_masks = rendered_warp_masks[np.newaxis, :, np.newaxis, :, :, :]
            rendered_warp_masks = rendered_warp_masks.transpose(0, 1, 2, 5, 3, 4)

            condition_images = torch.from_numpy(condition_images)
            if condition_images.max() > 1.1:
                condition_images = condition_images / 255.0
            T_cond, H, W, C = condition_images.shape
            cond_tensor = condition_images.permute(3, 0, 1, 2).unsqueeze(0)
            cond_tensor = cond_tensor * 2 - 1

            rendered_warp_images = torch.from_numpy(rendered_warp_images)
            if rendered_warp_images.max() > 1.1:
                rendered_warp_images = rendered_warp_images / 255.0
            warp_tensor = rendered_warp_images * 2 - 1

            rendered_warp_masks = torch.from_numpy(rendered_warp_masks)
            if rendered_warp_masks.max() > 1.1:
                rendered_warp_masks = rendered_warp_masks / 255.0
            mask_tensor = rendered_warp_masks

            device = self.device
            cond_tensor = cond_tensor.to(device=device, dtype=torch.float32)
            warp_tensor = warp_tensor.to(device=device, dtype=torch.float32)
            mask_tensor = mask_tensor.to(device=device, dtype=torch.float32)

            # Temporary num_steps / seed overrides, restored even if generation fails.
            original_num_steps, original_seed = self.pipeline.num_steps, self.pipeline.seed
            if num_steps is not None:
                self.pipeline.num_steps = int(num_steps)
            if seed is not None:
                self.pipeline.seed = seed
            steps_used = self.pipeline.num_steps
            print(f"Running Gen3C inference with {steps_used} steps")
            try:
                generated_output = self.pipeline.generate(
                    prompt=prompt,
                    image_path=cond_tensor,
                    negative_prompt=None,
                    rendered_warp_images=warp_tensor,
                    rendered_warp_masks=mask_tensor,
                )
            finally:
                self.pipeline.num_steps = original_num_steps
                self.pipeline.seed = original_seed
            del cond_tensor, warp_tensor, mask_tensor

            if generated_output is None:
                raise RuntimeError("Gen3C returned no video (blocked by the guardrail?)")
            video, _ = generated_output

            os.makedirs(out_dir, exist_ok=True)
            video_path = os.path.join(out_dir, "gen3c_video.mp4")
            if os.path.exists(video_path):
                os.remove(video_path)
            self.save_video(video=video, fps=self.fps, H=H, W=W, video_save_quality=self.video_quality,
                            video_save_path=video_path)

            frames_output_dir = os.path.join(out_dir, "frames")
            n_written = self._write_frames(video, frames_output_dir)
            print(f"Saved {n_written} frames to {frames_output_dir} and the video to {video_path}")
            return {"video_path": video_path, "frames_dir": frames_output_dir, "n_frames": int(n_written),
                    "size": [int(W), int(H)], "num_steps": int(steps_used), "prompt": prompt,
                    "seed": seed}
        finally:
            wzp.empty_cuda_cache()


def load(cfg):
    return Gen3cWorker(cfg)


if __name__ == "__main__":
    wzp.run_worker(SERVICE, load, dry_import)
