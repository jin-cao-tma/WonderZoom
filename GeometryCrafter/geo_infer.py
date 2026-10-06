from pathlib import Path
import os

import torch
import torch.nn.functional as F
from diffusers.training_utils import set_seed

from third_party import MoGe
from geometrycrafter import (
    GeometryCrafterDiffPipeline,
    GeometryCrafterDetermPipeline,
    PMapAutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModelVid2vid
)


# Pinned Hugging Face revisions (kept in sync with third_party/pins.env, which overrides them when present).
_DEFAULT_HF_REVISIONS = {
    "HF_REV_GEOMETRYCRAFTER": "2e8eeeecd205018f0937e05f711011505ab61573",
    "HF_REV_SVD_XT": "9e43909513c6714f1bc78bcb44d96e733cd242aa",
    "HF_REV_MOGE_VITL": "979e84da9415762c30e6c0cf8dc0962896c793df",
}


def hf_revision(key):
    """Revision for an HF repo: <repo>/third_party/pins.env if it defines `key`, else the built-in pin."""
    pins = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "pins.env")
    try:
        with open(pins) as f:
            for line in f:
                line = line.strip()
                if line.startswith(key + "="):
                    value = line.split("=", 1)[1].strip()
                    if value:
                        return value
    except OSError:
        pass
    return _DEFAULT_HF_REVISIONS[key]


def get_moge_geo_model(
    cache_dir: str = None,  # None = standard Hugging Face cache
    seed: int = 42,
    model_type: str = 'determ',
):
    assert model_type in ['diff', 'determ']
    set_seed(seed)

    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='unet_diff' if model_type == 'diff' else 'unet_determ',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        revision=hf_revision("HF_REV_GEOMETRYCRAFTER"),
    ).requires_grad_(False).to("cuda", dtype=torch.float16)

    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='point_map_vae',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
        revision=hf_revision("HF_REV_GEOMETRYCRAFTER"),
    ).requires_grad_(False).to("cuda", dtype=torch.float32)

    prior_model = MoGe(
        cache_dir=cache_dir,
        revision=hf_revision("HF_REV_MOGE_VITL"),
    ).requires_grad_(False).to('cuda', dtype=torch.float32)

    if model_type == 'diff':
        pipe = GeometryCrafterDiffPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir,
            revision=hf_revision("HF_REV_SVD_XT"),
        ).to("cuda")
    else:
        pipe = GeometryCrafterDetermPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir,
            revision=hf_revision("HF_REV_SVD_XT"),
        ).to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(e)
        print("Xformers is not enabled")

    pipe.enable_attention_slicing()
    return prior_model, pipe, point_map_vae

def inf_geometry(
    prior_model, pipe, point_map_vae, frames, 
    input_path: str = "none",
    save_folder: str = "workspace/output/",
    height: int = 512,
    width: int = 768,
    downsample_ratio: float = 1.0,
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 110,
    decode_chunk_size: int = 8,
    overlap: int = 25,
    process_length: int = -1,
    process_stride: int = 1,
    seed: int = 42,
    model_type: str = 'determ',
    force_projection: bool = True,
    force_fixed_focal: bool = True,
    use_extract_interp: bool = False,
    track_time: bool = False,
    low_memory_usage: bool = False
):
    """
    frames: numpy Array, [T, H, W, 3]
    """
    assert model_type in ['diff', 'determ']
    set_seed(seed)

    print("frames",frames.shape)
    original_height, original_width = frames[0].shape[:2]
    video_base_name = os.path.basename(input_path).split('/')[-1].split('.')[0]
    if height is None or width is None:
        height = original_height
        width = original_width
    assert height % 64 == 0
    assert width % 64 == 0

    frames = frames[::process_stride]
    if process_length > 0:
        frames = frames[:process_length]
    process_length = len(frames)
    window_size = min(window_size, process_length)
    if window_size == process_length:
        overlap = 0

    frames_tensor = torch.tensor(frames, device='cuda').float().permute(0, 3, 1, 2)

    # t,3,h,w

    if downsample_ratio > 1.0:
        original_height, original_width = frames_tensor.shape[-2], frames_tensor.shape[-1]
        frames_tensor = F.interpolate(frames_tensor, (round(frames_tensor.shape[-2]/downsample_ratio), round(frames_tensor.shape[-1]/downsample_ratio)), mode='bicubic', antialias=True).clamp(0, 1)

    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        rec_point_map, rec_valid_mask = pipe(
            frames_tensor,
            point_map_vae,
            prior_model,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            decode_chunk_size=decode_chunk_size,
            overlap=overlap,
            force_projection=force_projection,
            force_fixed_focal=force_fixed_focal,
            use_extract_interp=use_extract_interp,
            track_time=track_time,
            low_memory_usage=low_memory_usage
        )

        if downsample_ratio > 1.0:
            rec_point_map = F.interpolate(rec_point_map.permute(0,3,1,2), (original_height, original_width), mode='bilinear').permute(0, 2, 3, 1)
            rec_valid_mask = F.interpolate(rec_valid_mask.float().unsqueeze(1), (original_height, original_width), mode='bilinear').squeeze(1) > 0.5
        return rec_point_map[...,-1], rec_valid_mask 
        # np.savez(
        #     str(save_path / f"{video_base_name}.npz"), 
        #     point_map=rec_point_map.detach().cpu().numpy().astype(np.float16), 
        #     mask=rec_valid_mask.detach().cpu().numpy().astype(np.bool_))


