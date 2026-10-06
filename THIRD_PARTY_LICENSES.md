# Third-party licenses

WonderZoom builds on code and models released under different licenses. Several of them **do not allow
commercial use**. One of them, the NVIDIA Open Model License, also has a condition about safety guardrails that
applies to how WonderZoom runs Gen3C. This file summarizes the terms. The license texts in this repository, in the
upstream repositories and on the model pages are authoritative. Read them before you use or redistribute the code,
the weights or the generated scenes.

## WonderZoom's own code

The authors have not chosen a license for WonderZoom's own code yet, and the repository has no `LICENSE` file for
it. Until they add one, no license is granted for that code. <!-- TODO(authors): choose the license of WonderZoom's own code and add a root LICENSE file. It must be compatible with the CC BY-NC-SA 4.0 terms of the LucidDreamer-derived files and the Inria Gaussian-Splatting License of the 3DGS-derived files listed below, and must not claim a more permissive license for them. -->

Whatever license the authors choose, it cannot change the terms of the third-party code below. In particular:
- the files derived from LucidDreamer stay under CC BY-NC-SA 4.0, which requires attribution, non-commercial use
  and that adaptations are shared under the same license;
- the 3DGS-derived files stay under the Inria Gaussian-Splatting License (non-commercial research and evaluation).

## Summary of restrictions

| Restriction | Components |
|---|---|
| **Non-commercial / research only** | 3DGS rasterizer, simple-knn and the 3DGS-derived code (Inria Gaussian-Splatting License); files derived from LucidDreamer (CC BY-NC-SA 4.0); GeometryCrafter code and weights (academic, research and education only); Stable Diffusion 3 Medium (Stability AI Non-Commercial Research Community License, gated); Qwen2.5-VL-3B-Instruct (Qwen Research License) |
| **Share-alike** | files derived from LucidDreamer (CC BY-NC-SA 4.0): adaptations must be shared under the same license |
| **Stability AI Community License** (research and non-commercial use; commercial use only with registration and below US$1M annual revenue) | Stable Video Diffusion img2vid-xt (also the base of GeometryCrafter) |
| **Use-based restrictions** | Stable Diffusion 2 inpainting (CreativeML Open RAIL++-M) |
| **Guardrail condition** | Gen3C-Cosmos-7B and the Cosmos tokenizer (NVIDIA Open Model License); see [below](#nvidia-open-model-license-and-the-gen3c-guardrail) |
| **Paid API, provider terms** | OpenAI GPT-4o (optional; only with `OPENAI_API_KEY`) |

**Net effect: both the full generation pipeline and the render-only viewer are for non-commercial use only.** The
render-only viewer uses none of the models, but it uses the 3DGS rasterizer, simple-knn, the 3DGS-derived code and
the LucidDreamer-derived `arguments_in.py`. The generation pipeline additionally needs GeometryCrafter, SD3 Medium
and Qwen2.5-VL-3B-Instruct, which are all non-commercial.

## Code in this repository

| Component | Path | Upstream | License |
|---|---|---|---|
| Differentiable Gaussian rasterizer (depth variant) | `submodules/depth-diff-gaussian-rasterization-min/` | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Gaussian-Splatting License (Inria, MPII): non-commercial research and evaluation; `LICENSE.md` in that directory |
| simple-knn | `submodules/simple-knn/` | graphdeco-inria | Gaussian-Splatting License; `LICENSE.md` in that directory |
| 3DGS-derived files (Inria header) | `gaussian_renderer/`, `scene/gaussian_model.py`, `scene/cameras.py`, `scene/dataset_readers.py`, `scene/colmap_loader.py`, `utils/{loss,general,graphics,camera,system,image}.py` | graphdeco-inria/gaussian-splatting, through LucidDreamer and WonderWorld | Gaussian-Splatting License. The headers refer to "the LICENSE.md file"; a copy is in [`LICENSES/Gaussian-Splatting-LICENSE.md`](LICENSES/Gaussian-Splatting-LICENSE.md) (identical to the rasterizer's `LICENSE.md`). See also the LucidDreamer row. |
| LucidDreamer code | copied unchanged: `utils/trajectory.py`, `utils/depth.py`; adapted: `arguments_in.py` (from LucidDreamer's `arguments.py`), `scene/__init__.py`, `scene/dataset_readers.py` | [luciddreamer-cvlab/LucidDreamer](https://github.com/luciddreamer-cvlab/LucidDreamer) @ `76ed990f`, Copyright (c) 2023 Computer Vision Lab, Seoul National University | **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike 4.0 International); [`LICENSES/LucidDreamer-CC-BY-NC-SA-4.0.txt`](LICENSES/LucidDreamer-CC-BY-NC-SA-4.0.txt), <https://creativecommons.org/licenses/by-nc-sa/4.0/>. Details [below](#luciddreamer-cc-by-nc-sa-40). |
| Spherical-harmonics helpers | `utils/sh.py` (via LucidDreamer, unchanged) | PlenOctree | BSD-2-Clause (in the file header) |
| GeometryCrafter (trimmed, `geo_infer.py` added) | `GeometryCrafter/` | [TencentARC/GeometryCrafter](https://github.com/TencentARC/GeometryCrafter) @ `03ba645e` | GeometryCrafter license: academic, research and education use only; `GeometryCrafter/LICENSE`, `GeometryCrafter/NOTICE` (includes the Stability AI Community License of SVD) |
| MoGe (trimmed) | `MoGe/` | [microsoft/MoGe](https://github.com/microsoft/MoGe) @ `72fdee98` | MIT; its DINOv2 code is Apache-2.0 (Meta). Both texts: `MoGe/LICENSE` |
| MoGe snapshot inside GeometryCrafter (trimmed) | `GeometryCrafter/third_party/moge/` | microsoft/MoGe @ `dd158c05` (the commit of GeometryCrafter's `third_party/moge` submodule) | MIT, DINOv2 code Apache-2.0: [`GeometryCrafter/third_party/moge/LICENSE`](GeometryCrafter/third_party/moge/LICENSE). The "LICENSE file in the root directory of this source tree" named by the DINOv2 headers in this snapshot is that file, **not** `GeometryCrafter/LICENSE`. |
| RepViT-SAM | `RepViT/sam/` | [THU-MIG/RepViT](https://github.com/THU-MIG/RepViT) | Apache-2.0; `RepViT/LICENSE` |
| Marigold pipeline (modified) | `marigold_lcm/` | [prs-eth/Marigold](https://github.com/prs-eth/Marigold), through WonderWorld; parts adapted from the Marigold pipelines of [huggingface/diffusers](https://github.com/huggingface/diffusers) | Apache-2.0; [`marigold_lcm/LICENSE`](marigold_lcm/LICENSE), and [`marigold_lcm/NOTICE`](marigold_lcm/NOTICE) for the attributions and the modifications |
| Web viewer | `splat-main/` | [antimatter15/splat](https://github.com/antimatter15/splat), via [haoyi-duan/splat](https://github.com/haoyi-duan/splat) | MIT; `splat-main/LICENSE` |
| Socket.IO client 4.5.0 | `splat-main/vendor/socket.io-4.5.0.min.js` | [socketio/socket.io](https://github.com/socketio/socket.io) | MIT |
| stb_image_write | `submodules/depth-diff-gaussian-rasterization-min/third_party/stbi_image_write.h` | [nothings/stb](https://github.com/nothings/stb) | public domain / MIT |
| Chain-of-Zoom wrapper | `third_party/chain_of_zoom/wonderzoom_coz.py` | built on [bryanswkim/Chain-of-Zoom](https://github.com/bryanswkim/Chain-of-Zoom) @ `42deeda0` | MIT, Copyright (c) 2025 Bryan Sangwoo Kim; [`third_party/chain_of_zoom/LICENSE`](third_party/chain_of_zoom/LICENSE) |
| Step1X-Edit wrapper | `third_party/step1x_edit/simple_step1x.py` | adapted from [stepfun-ai/Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit) | Apache-2.0 |
| INR-Harmonization patch and wrapper | `third_party/inr_harmonization/` | [WindVChen/INR-Harmonization](https://github.com/WindVChen/INR-Harmonization) | Apache-2.0 |

### LucidDreamer (CC BY-NC-SA 4.0)

WonderZoom contains code from LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes, by the
Computer Vision Lab, Seoul National University (<https://github.com/luciddreamer-cvlab/LucidDreamer>). LucidDreamer
is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License; the full
text is in [`LICENSES/LucidDreamer-CC-BY-NC-SA-4.0.txt`](LICENSES/LucidDreamer-CC-BY-NC-SA-4.0.txt) and at
<https://creativecommons.org/licenses/by-nc-sa/4.0/>. The material is provided as-is, without warranties.

| File | Origin | Changes |
|---|---|---|
| `utils/trajectory.py` | LucidDreamer `utils/trajectory.py` | none |
| `utils/depth.py` | LucidDreamer `utils/depth.py` (its `colorize` helper comes from [ZoeDepth](https://github.com/isl-org/ZoeDepth), MIT) | none |
| `arguments_in.py` | LucidDreamer `arguments.py` | adapted by the WonderWorld and WonderZoom authors (`GSParams`, `CameraParams`) |
| `scene/__init__.py` | LucidDreamer `scene/__init__.py` | adapted by the WonderWorld and WonderZoom authors |
| `scene/dataset_readers.py` | LucidDreamer's version of the 3DGS file (Inria header) | adapted by the WonderWorld and WonderZoom authors |

The original file headers are kept unchanged. Some of them say "All rights reserved" or ask for permission
requests to the LucidDreamer authors; the LucidDreamer repository distributes these files under the CC BY-NC-SA 4.0
license above. Several 3DGS-derived files (for example `utils/camera.py`, `scene/cameras.py`,
`scene/colmap_loader.py`) also came to WonderZoom through LucidDreamer's and WonderWorld's copies. The Inria
Gaussian-Splatting License applies to them; any changes LucidDreamer made to them are covered by CC BY-NC-SA 4.0.

## Code fetched by the install scripts (not in this repository)

`scripts/setup_third_party.sh` clones these at the commits in `third_party/pins.env`; the install scripts build or
pip-install the rest.

| Component | Where | License |
|---|---|---|
| [GEN3C](https://github.com/nv-tlabs/GEN3C) | `external/GEN3C` | Apache-2.0 |
| [NVIDIA apex](https://github.com/NVIDIA/apex) | `external/apex` (built into `wz-gen3c`) | BSD-3-Clause |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 1.12 | `wz-gen3c` | Apache-2.0 |
| [Chain-of-Zoom](https://github.com/bryanswkim/Chain-of-Zoom), including its SR LoRA and VAE checkpoints | `external/Chain-of-Zoom` | MIT |
| [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit) | `external/Step1X-Edit` | Apache-2.0 |
| [flash-attention](https://github.com/Dao-AILab/flash-attention) 2.7.4.post1 | `wz-step1x` | BSD-3-Clause |
| [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) (GroundingDINO, segment_anything) | `external/Grounded-Segment-Anything` (optional) | Apache-2.0 |
| [INR-Harmonization](https://github.com/WindVChen/INR-Harmonization) | `external/INR-Harmonization` (optional) | Apache-2.0 |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) 0.7.8 | `wz-main` | BSD-3-Clause |
| [GLM](https://github.com/g-truc/glm) | `submodules/depth-diff-gaussian-rasterization-min/third_party/glm` | Happy Bunny License or MIT |
| [utils3d](https://github.com/EasternJournalist/utils3d) | `wz-main` | MIT |

All other Python dependencies are installed from PyPI under their own licenses (`requirements/*.txt` and the
upstream requirement files).

## Model weights (downloaded by `scripts/download_checkpoints.sh`)

Licenses as stated on the model pages at the revisions pinned in `third_party/pins.env`.

| Model | Used for | Group | License | Notes |
|---|---|---|---|---|
| [OneFormer ADE20k Swin-L](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) | sky masks | `--core` | MIT | |
| [OneFormer demo metadata](https://huggingface.co/datasets/shi-labs/oneformer_demo) | ADE20k class list for OneFormer | `--core` | no license stated on the dataset page | small JSON file |
| [Marigold normals v0-1](https://huggingface.co/prs-eth/marigold-normals-v0-1) | normals | `--core` | Apache-2.0 | |
| [GeometryCrafter](https://huggingface.co/TencentARC/GeometryCrafter) | video depth for camera moves | `--core` | GeometryCrafter license | **academic / research / education only** |
| [Stable Video Diffusion img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) (image encoder, VAE) | GeometryCrafter pipeline | `--core` | Stability AI Community License (July 5, 2024) | research and non-commercial use; commercial use only with registration and below US$1M annual revenue |
| [MoGe ViT-L](https://huggingface.co/Ruicheng/moge-vitl) | single-image geometry | `--core` | Apache-2.0 (model page) | |
| [RepViT-SAM](https://github.com/THU-MIG/RepViT) | segments for depth alignment | `--core` | Apache-2.0 | |
| [Gen3C-Cosmos-7B](https://huggingface.co/nvidia/GEN3C-Cosmos-7B) | camera moves, HQ views | `--gen3c` | NVIDIA Open Model License | guardrail condition, see below |
| [Cosmos-Tokenize1-CV8x8x8-720p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p) | Gen3C video tokenizer | `--gen3c` | NVIDIA Open Model License | guardrail condition, see below |
| [T5-11B](https://huggingface.co/google-t5/t5-11b) | Gen3C text encoder | `--gen3c` | Apache-2.0 | |
| [Stable Diffusion 3 Medium (diffusers)](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | Chain-of-Zoom SR backbone | `--coz` | **Stability AI Non-Commercial Research Community License** (December 6, 2023) | **gated** (accept the license on the model page); **non-commercial use only**. The more permissive Community License belongs to the single-file repository `stabilityai/stable-diffusion-3-medium`, not to the diffusers repository WonderZoom uses. |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Chain-of-Zoom prompts | `--coz` | Qwen Research License | **non-commercial** |
| [Step1X-Edit v1.0](https://huggingface.co/stepfun-ai/Step1X-Edit) | object insertion | `--step1x` | Apache-2.0 | |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Step1X-Edit conditioner | `--step1x` | Apache-2.0 | |
| [GroundingDINO SwinT-OGC](https://github.com/IDEA-Research/GroundingDINO) | object segmentation | `--objects` | Apache-2.0 | |
| [BERT base uncased](https://huggingface.co/bert-base-uncased) | GroundingDINO text encoder | `--objects` | Apache-2.0 | |
| [SAM ViT-H](https://github.com/facebookresearch/segment-anything) | object segmentation | `--objects` | Apache-2.0 | |
| [Stable Diffusion 2 inpainting](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting) (community mirror) | background plates | `--objects` | CreativeML Open RAIL++-M | use-based restrictions |
| INR-Harmonization `Resolution_RAW_iHarmony4.pth` | harmonization | `--objects` | released with the Apache-2.0 [INR-Harmonization](https://github.com/WindVChen/INR-Harmonization) repository | hosted on Google Drive |
| [Released WonderZoom scenes](https://huggingface.co/datasets/TmaKiss/WonderZoom) | render-only viewer | `--scenes` | see the dataset page <!-- TODO(authors): scene license --> | generated with the models above |

Not downloaded by default: the Gen3C guardrail models [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)
(NVIDIA Open Model License, gated) and [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
(Llama 3.1 Community License, gated). They are needed only with `services.gen3c.disable_guardrail: false`.

Optional: OpenAI GPT-4o, used only for object-insertion prompts when `OPENAI_API_KEY` is set. It is a paid API
under OpenAI's terms.

## NVIDIA Open Model License and the Gen3C guardrail

The Gen3C-Cosmos-7B and Cosmos tokenizer weights are released under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
The Gen3C model card states:

> If you bypass, disable, reduce the efficacy of, or circumvent any technical limitation, safety guardrail or
> associated safety guardrail hyperparameter, encryption, security, digital rights management, or authentication
> mechanism contained in the Model, your rights under NVIDIA Open Model License Agreement will automatically
> terminate.

**By default, WonderZoom runs Gen3C with the guardrail disabled.** The default `services.gen3c.disable_guardrail: true`
(with the prompt upsampler off) reproduces the paper runs. It matches the original WonderZoom service, GEN3C's GUI
server and the low-memory example in GEN3C's README, which pass `--disable_guardrail`. The guardrail models are
therefore not downloaded. Read the license and decide whether your use is covered before you run Gen3C through
WonderZoom or redistribute its outputs.

To run with the guardrail, set `services.gen3c.disable_guardrail: false` (for example in
`config/services.local.yaml`), then download Cosmos-Guardrail1 and Llama-Guard-3-8B into the Gen3C checkpoint
directory, as described in [docs/INSTALL.md](docs/INSTALL.md#gen3c-guardrail). The guardrail checks the text prompt
and every generated video. It can reject a request, and it blurs faces in the generated frames, so results can
differ from the paper's. `services.gen3c.offload_guardrail_models` (default `true`) loads the guardrail models for
each check and frees them afterwards. See [third_party/gen3c/README.md](third_party/gen3c/README.md) for how the
worker runs Gen3C.
