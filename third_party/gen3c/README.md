# Gen3C in WonderZoom

WonderZoom uses [GEN3C](https://github.com/nv-tlabs/GEN3C) (Gen3C-Cosmos-7B, a Cosmos-Predict1
video DiT) for camera moves and high-quality novel views. It takes a 121-frame rendering of the
scene with holes blacked out, plus the hole masks, and generates the missing content.

## Upstream code: pinned, no patch

| Repository | Commit | Notes |
|---|---|---|
| `nv-tlabs/GEN3C` | `2b50e0b497f76ace046636398dfab23edf4c43de` | Used unmodified (`GEN3C_COMMIT` in `third_party/pins.env`). |
| `NVIDIA/apex` | `e74a67bba3ee679f778670e17edc21639008ae0a` | Build only, with `--cpp_ext --cuda_ext`. `cosmos_predict1` imports `amp_C` at import time even for inference. |

`scripts/setup_third_party.sh gen3c apex` clones both repositories into `external/`. No WonderZoom
file is copied into the GEN3C tree. Later upstream commits, up to `db2ffe1`, do not touch
`gen3c_pipeline.py` or the model code.

The authors' original working copy carried eight local edits. None of them is needed, and the
release does not ship them:
- a debug dump in `gen3c_pipeline.py` that wrote `debug_warp_*.png` files and copied the inputs to the host;
- a different default `--negative_prompt`, which WonderZoom never uses;
- `tqdm` progress bars in four model files;
- two GUI-only edits.

## How WonderZoom runs it

`services/workers/gen3c_worker.py` runs inside the `wz-gen3c` environment as a persistent worker.
It is started by `services/` (see `config/services.yaml`, section `services.gen3c`):

- Working directory and `PYTHONPATH` are `external/GEN3C`. The working directory matters because the
  pipeline loads the relative config file `cosmos_predict1/diffusion/config/config.py`.
- `CUDA_HOME` is set to the environment prefix, because Transformer Engine 1.12 finds NVRTC through
  `CUDA_HOME`.
- `HF_HUB_OFFLINE=1` is set. All weights are local; without this, `CosmosT5TextEncoder` silently falls
  back to downloading T5-11B (about 45 GB) when its directory is incomplete.
- `LD_LIBRARY_PATH` is not inherited, so torch uses the cuDNN from its own wheels.

The worker builds `Gen3cPipeline` directly, after `misc.set_random_seed(1)` and under
`torch.no_grad()`. It uses the arguments the original WonderZoom service passed through
`Gen3cPersistentModel`:

```python
Gen3cPipeline(inference_type="video2world", checkpoint_dir=<checkpoints/gen3c>,
              checkpoint_name="Gen3C-Cosmos-7B", prompt_upsampler_dir="Pixtral-12B",
              enable_prompt_upsampler=False, offload_prompt_upsampler=True,
              offload_guardrail_models=False, disable_guardrail=True,
              offload_network=False, offload_tokenizer=False, offload_text_encoder_model=False,
              guidance=1.0, num_steps=18, height=704, width=1280, fps=24,
              num_video_frames=121, seed=1)
```

The two guardrail arguments come from `config/services.yaml`. `disable_guardrail` is
`services.gen3c.disable_guardrail` (default `true`, as above). `offload_guardrail_models` is
`services.gen3c.offload_guardrail_models` (default `true`) when the guardrail is on, and `False`
otherwise. See [License and guardrail](#license-and-guardrail).

It does not use `gen3c_persistent.py` (`Gen3cPersistentModel`) or `gen3c_single_image.py`, for two
reasons:
- They always download and load MoGe ViT-L (`Ruicheng/moge-vitl`, about 1.3 GB of VRAM), which
  WonderZoom never uses.
- They need the `moge` package. Installing it into this environment upgrades `huggingface-hub` and
  `numpy` and breaks `transformers==4.49.0`.

`MoGeModel.from_pretrained` builds the model on the CPU and Gen3C draws its noise from the CUDA
generator, so dropping MoGe does not change the random stream of the first request.

### Request `generate`

`generate` keeps the exact conventions of the original service:
- `input.png`, the 121 warped frames and the 121 masks are loaded with OpenCV and resized to
  1280x704 with `cv2.INTER_LINEAR`. WonderZoom renders at 1088x720, so the aspect ratio changes; the
  main process undoes the change when it reads the frames back.
- Masks are averaged over the channels, clipped and inverted. WonderZoom writes holes as 255, while
  Gen3C expects 1 for valid pixels.
- Images become tensors in [-1, 1] and masks stay in [0, 1], all float32 on the GPU.
- `pipeline.num_steps` is temporarily overridden: 18 for camera moves, 10 for HQ views.
- The call is `pipeline.generate(prompt, image_path=<tensor>, negative_prompt=None, rendered_warp_images, rendered_warp_masks)`.
- Outputs are `<out_dir>/gen3c_video.mp4` (24 fps, quality 5) and
  `<out_dir>/frames/frame_%08d.png`. Frames get the original global min-max stretch;
  `minmax_normalize_frames: false` disables it.
- Exactly 121 frames are required. The tokenizer encodes 121-frame chunks and the DiT expects one
  chunk, so the old padding to a multiple of 121 could not work for longer inputs.
- Noise comes from the process-global RNG, which is seeded once at start-up, as in the paper-era
  service. `seed` per request, or `request_seed` in the config, reseeds the RNG before a request.

### Memory

| State | GPU memory |
|---|---|
| Resident: DiT bf16 + T5-11B fp32 + tokenizer | about 36 GB |
| T5 parked in host RAM (`park_text_encoder`, automatic under the exclusive policy) | about 15 GB |
| Suspended by the GPU arbiter (DiT and T5 in host RAM) | tokenizer + CUDA context |

The worker adds two plumbing changes around the pipeline. Neither changes the numerics.
- **T5 embedding cache.** `_run_text_embedding_on_prompt_with_offload` is wrapped on the pipeline
  instance with a per-prompt CPU cache, so T5 runs once per distinct prompt. When
  `park_text_encoder` is on, T5 is moved to the GPU only for a cache miss and then parked again.
- **T5 parked at load.** A small `Gen3cPipeline` subclass overrides `_load_text_encoder_model` to
  park T5 right after it is created. This keeps the start-up peak at about 20 GB instead of 35 GB.

The upstream `offload_*` flags delete a model after each use and reload it from disk:
- `offload_network`: the 29 GB fp32 DiT checkpoint;
- `offload_text_encoder_model`: the 45 GB T5 pickle;
- `offload_tokenizer`: the tokenizer.

They are exposed in `config/services.yaml` as a last resort. Loading needs at least 96 GB of host
RAM, because the T5 checkpoint is a legacy pickle and the DiT is stored in fp32.

## Checkpoints (`scripts/download_checkpoints.sh --gen3c`)

```
checkpoints/gen3c/
  Gen3C-Cosmos-7B/model.pt                       nvidia/GEN3C-Cosmos-7B @ 9bcfdb4
  Cosmos-Tokenize1-CV8x8x8-720p/                 nvidia/Cosmos-Tokenize1-CV8x8x8-720p @ b6af495
      encoder.jit decoder.jit mean_std.pt image_mean_std.pt config.json
  google-t5/t5-11b/                              google-t5/t5-11b @ 90f3770
      config.json pytorch_model.bin spiece.model tokenizer.json
```

None of these repositories is gated. The upstream downloader also fetches `tf_model.h5`, an extra
45 GB that is not needed. The worker checks for these files at start-up and reports a fatal error
that lists any missing file.

## License and guardrail

- The GEN3C code is Apache-2.0. The Gen3C-Cosmos-7B and Cosmos tokenizer weights are released under
  the NVIDIA Open Model License.
- By default, WonderZoom runs the pipeline with `disable_guardrail=True`
  (`services.gen3c.disable_guardrail: true` in `config/services.yaml`) and
  `enable_prompt_upsampler=False`, like the original WonderZoom service and GEN3C's own GUI server.
  `download_checkpoints.sh` therefore does not fetch the guardrail models (Cosmos-Guardrail1,
  Llama-Guard-3-8B).
- The NVIDIA Open Model License states that its rights terminate if safety guardrails are bypassed
  or disabled. Read
  [THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md#nvidia-open-model-license-and-the-gen3c-guardrail)
  before using or redistributing results.
- To run Gen3C with the guardrail, download the two guardrail models and set
  `services.gen3c.disable_guardrail: false` in `config/services.local.yaml`; see
  [docs/INSTALL.md](../../docs/INSTALL.md#gen3c-guardrail). The guardrail path has not been tested
  end to end with WonderZoom.

## Smoke tests

```bash
# Imports only (no weights), inside external/GEN3C:
(cd external/GEN3C && CUDA_HOME=$CONDA_PREFIX PYTHONPATH=. python ../../services/workers/gen3c_worker.py --dry-import)
# Full worker test: 121 frames at 1280x704 plus an mp4, timings and peak memory:
python tests/smoke_workers.py --service gen3c --steps 18 --suspend-resume --check-errors
```
