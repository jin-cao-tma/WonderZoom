# Installation

WonderZoom's generation pipeline uses four conda environments. Gen3C, Chain-of-Zoom and Step1X-Edit pin
incompatible versions of torch, transformers and diffusers, so each one runs as a worker process inside
its own environment. The main process (`run.py`) talks to the workers over pipes; you never activate the
worker environments yourself.

| Environment | Runs | Key versions | Compiles | Install script |
|---|---|---|---|---|
| `wz-main` | `run.py` (generation and `--view`), the in-process models (MoGe, GeometryCrafter, OneFormer, Marigold, RepViT-SAM; optional GroundedSAM, SD2 inpainting, INR), all scripts and tests | Python 3.10, torch 2.4.0+cu124, torchvision 0.19.0, PyTorch3D 0.7.8, transformers 4.37.2, diffusers 0.31.0 | PyTorch3D, 3DGS rasterizer, simple-knn (+ GroundingDINO with `--objects`) | `scripts/install_env_main.sh` |
| `wz-gen3c` | `services/workers/gen3c_worker.py` (camera moves, HQ views) | Python 3.10, torch 2.6.0, transformer-engine 1.12.0, apex, megatron-core 0.10.0, transformers 4.49.0, diffusers 0.32.2 | transformer-engine, apex | `scripts/install_env_gen3c.sh` |
| `wz-coz` | `services/workers/coz_worker.py` (zoom-in super-resolution) | Python 3.10, torch 2.4.1 (cu121 wheel), diffusers 0.32.1, transformers 4.49.0, peft 0.15.2 | nothing | `scripts/install_env_coz.sh` |
| `wz-step1x` | `services/workers/step1x_worker.py` (object insertion only) | Python 3.10, torch 2.7.1 (cu126 wheel), transformers 4.51.3, diffusers 0.34.0, flash-attn 2.7.4.post1 (prebuilt wheel) | nothing | `scripts/install_env_step1x.sh` |

Viewing a released or saved scene (`python run.py --view`) needs only `wz-main` and the scene file: no checkpoints
and no other environment.

There is no top-level `requirements.txt`. A plain `pip install -r` cannot build this stack: `wz-main` needs a
pinned torch build, PyTorch3D and the 3DGS CUDA extensions compiled from source, and `utils3d` from a pinned git
commit. Use the install scripts below. The pinned pip lists they install are `requirements/main.txt`,
`requirements/main-objects.txt` (object insertion) and `requirements/step1x.txt`; the Gen3C and Chain-of-Zoom
environments use the upstream requirement files of their pinned clones.

## Prerequisites

- Linux x86_64, with `conda` (Miniconda or Miniforge), `git` and `curl` on `PATH`.
- An NVIDIA driver that supports CUDA 12.4. `wz-step1x` needs CUDA 12.6, because the PyPI torch 2.7.1 wheel is a cu126
  build. No system CUDA toolkit or compiler is needed: `wz-main` and `wz-gen3c` bring GCC 12.4 and the CUDA 12.4
  toolkit from conda-forge.
- A host C compiler at run time for `wz-step1x`: Triton JIT-compiles the liger RMSNorm kernel.
- Disk and RAM: see [HARDWARE.md](HARDWARE.md). About 168 GB of checkpoints, plus the environments.

The install scripts work without a visible GPU (for example on a login node), but they then skip the GPU kernel
tests. Run `scripts/check_install.py` on a GPU node afterwards.

## One-shot install

```bash
bash scripts/install_all.sh            # wz-main, wz-gen3c, wz-coz
bash scripts/install_all.sh --objects  # + object-insertion stack in wz-main, and wz-step1x
```

| Option | Effect |
|---|---|
| `--objects` | also run `install_objects_optional.sh` on `wz-main` and build `wz-step1x` |
| `--prefix-root DIR` | create the environments as `DIR/wz-main`, `DIR/wz-gen3c`, ... instead of named conda envs (useful for a fast local disk) |
| `--only main,coz` / `--skip gen3c` | build a subset |
| `--keep-going` | continue with the next environment after a failure |
| `--dry-run` | show what each script would do; read-only checks only |

Logs go to `logs/install/<env>.log`. Every step is idempotent: re-running after a failure skips what is already
done. When the environments are built, `install_all.sh` runs `scripts/check_install.py` on them with
`--no-checkpoints`, because the checkpoints are downloaded in the next step, and with `--no-gpu` when no GPU is
visible. It then prints the matching `download_checkpoints.sh` command and the full `check_install.py` command to
run once the checkpoints are in place.

Build time: a clean build of one environment takes roughly 30-60 min on 16 CPUs. The script headers estimate
several hours on a 2-CPU machine. transformer-engine, apex and PyTorch3D dominate.

## Build settings

These environment variables are passed through by `install_all.sh` and every per-environment script:

| Variable | Default | Notes |
|---|---|---|
| `MAX_JOBS` | number of CPUs, at most 8 | Parallel compile jobs. Each nvcc job can use 4-6 GB of RAM, so lower it on machines with little memory. |
| `TORCH_CUDA_ARCH_LIST` | `8.0;8.6;8.9;9.0` | GPU architectures compiled into PyTorch3D, the rasterizer, simple-knn, GroundingDINO and apex. A single entry such as `8.9` builds several times faster, but the result then only runs on that GPU family. |
| `NVTE_CUDA_ARCHS` | transformer-engine's default (`70;80;89;90`) | Architectures for the transformer-engine build in `wz-gen3c`. |
| `PIP_CONFIG_FILE` | unset | Set it to `/dev/null` to ignore a `pip.conf` that adds unreachable extra indexes. |
| `WZ_TRACE` | `0` | `1` prints every command (bash xtrace). |

Architectures in the default list:

| Arch | GPUs |
|---|---|
| `8.0` | A100 |
| `8.6` | RTX 30xx, A6000 |
| `8.9` | RTX 40xx, L40S, RTX 6000 Ada |
| `9.0` | H100 |

To find your GPU's value, run `nvidia-smi --query-gpu=name,compute_cap --format=csv`.

## Per-environment details

Each script accepts `--name NAME` or `--prefix DIR` (default name: the environment name above) and `--dry-run`.
As its last step, each script registers the environment's interpreter in `config/services.local.yaml`; see
[Registering environments](#registering-environments).

### `wz-main` (`scripts/install_env_main.sh [--objects] [--force-rebuild]`)

1. Creates the conda environment from `envs/wz-main.yml`: Python 3.10, pip 25.0, cmake, ninja, GCC 12.4,
   and the CUDA 12.4 nvcc and toolkit from conda-forge. `CUDA_HOME` is set to the environment prefix during the build.
2. Installs torch 2.4.0 and torchvision 0.19.0 from the cu124 index, then `requirements/main.txt` with torch held fixed.
3. Builds PyTorch3D at `PYTORCH3D_COMMIT` (v0.7.8). This step takes 45-90 min with `MAX_JOBS=2`.
4. Fetches the GLM headers (`setup_third_party.sh glm`), then builds `submodules/depth-diff-gaussian-rasterization-min`
   and `submodules/simple-knn`.
5. Installs `RepViT/sam` in editable mode.
6. With `--objects`, runs `scripts/install_objects_optional.sh`. That script installs `requirements/main-objects.txt`
   (openai, supervision, albumentations, adamp, gdown, ...), GroundingDINO (it compiles `groundingdino._C`) and
   segment_anything from Grounded-Segment-Anything at `GSAM_COMMIT`, and INR-Harmonization at `INR_COMMIT` with
   WonderZoom's patch and wrapper.
7. Runs a strict import and version check against `requirements/main.txt`, then registers the interpreter.

`--force-rebuild` rebuilds PyTorch3D and the CUDA extensions even when they already import. Use it after
changing `TORCH_CUDA_ARCH_LIST`.

Notes:
- `utils3d` must come from the pinned git commit in `requirements/main.txt`. The PyPI package `utils3d` is an
  unrelated project.
- Do not install the `moge` pip package into this environment. WonderZoom uses its vendored copies in `MoGe/` and
  `GeometryCrafter/third_party/moge`.

### `wz-gen3c` (`scripts/install_env_gen3c.sh [--force-rebuild]`)

This is the upstream GEN3C `INSTALL.md` recipe at the pinned commit, without MoGe.

1. Clones `external/GEN3C` and `external/apex` at their pins (`setup_third_party.sh gen3c apex`).
2. Creates the conda environment from `external/GEN3C/cosmos-predict1.yaml`, then installs
   `external/GEN3C/requirements.txt`.
3. Symlinks the CUDA headers of the `nvidia-*` wheels into the environment, which the transformer-engine build needs.
4. Builds `transformer-engine[pytorch]==1.12.0` from source, then apex with `--cpp_ext --cuda_ext`.
   apex with CUDA extensions is required even for inference, because Gen3C imports `amp_C` at import time.
5. Checks that `Gen3cPipeline` imports, then registers the interpreter.

MoGe is not installed: only `gen3c_persistent.py` and `gen3c_single_image.py` need it, and the worker uses neither.
Installing it would upgrade `huggingface-hub` and `numpy` and break `transformers==4.49.0`. See
[third_party/gen3c/README.md](../third_party/gen3c/README.md) for how the worker runs Gen3C.

### `wz-coz` (`scripts/install_env_coz.sh`)

1. Clones `external/Chain-of-Zoom` at `COZ_COMMIT` and copies `third_party/chain_of_zoom/wonderzoom_coz.py` into it.
   The SR LoRA and VAE checkpoints (`ckpt/SR_LoRA`, `ckpt/SR_VAE`, about 77 MB) come with the clone.
2. Creates a Python 3.10 environment and installs the upstream `external/Chain-of-Zoom/requirements.txt`.
3. Runs an import check, then registers the interpreter.

Keep the pin. Upstream `main` imports `utils.vaehook`, which clashes with WonderZoom's `utils/` package.
The model weights, SD3-Medium (gated) and Qwen2.5-VL-3B, come from `download_checkpoints.sh --coz`.

### `wz-step1x` (`scripts/install_env_step1x.sh`), object insertion only

1. Clones `external/Step1X-Edit` at `STEP1X_COMMIT` and copies `third_party/step1x_edit/simple_step1x.py` into it.
2. Creates a Python 3.10 environment and installs torch 2.7.1 and torchvision 0.22.1, then `requirements/step1x.txt`,
   then the flash-attn wheel in `FLASH_ATTN_WHEEL_URL` (`third_party/pins.env`).
3. Runs an import check, then registers the interpreter. The `liger_kernel` import needs a visible GPU, so the check
   is partly skipped on a node without one.

Keep the pin. Upstream `main` makes `Step1XParams.version` a required field, which `simple_step1x.py` does not pass.

## Registering environments

The install scripts record each interpreter in `config/services.local.yaml`, which is machine-specific and
gitignored. `run.py`, `run_server.sh`, `check_install.py` and the download script read it from there. To use
environments you built yourself, register them by hand:

```bash
python scripts/register_env.py main  /path/to/envs/wz-main/bin/python
python scripts/register_env.py gen3c /path/to/envs/wz-gen3c/bin/python
python scripts/register_env.py --show          # list registrations
python scripts/register_env.py --get coz       # print the effective interpreter
```

The environment variables `WZ_MAIN_PYTHON`, `WZ_GEN3C_PYTHON`, `WZ_COZ_PYTHON` and `WZ_STEP1X_PYTHON` override
the registrations. A service whose interpreter is not registered is disabled at start-up. For example, without
`wz-step1x` the server runs normally, but object insertion is unavailable.

## Third-party code

`scripts/setup_third_party.sh` clones each upstream repository into `external/` at the exact commit in
`third_party/pins.env`, verifies `HEAD`, and adds WonderZoom's files: `wonderzoom_coz.py`, `simple_step1x.py`, and
the INR patch and wrapper. The install scripts call it for you.

```bash
bash scripts/setup_third_party.sh --all           # gen3c apex coz step1x inr gsam glm
bash scripts/setup_third_party.sh --check         # verify the clones that are set up, change nothing
bash scripts/setup_third_party.sh --check --all   # verify every component; fails for any that is not set up
```

To keep the clones outside the repository, export `WZ_EXTERNAL_DIR=/abs/path` before running the install scripts
(`setup_third_party.sh` also accepts `--external-dir DIR`). Export the same `WZ_EXTERNAL_DIR` when you run `run.py`,
`run_server.sh` and `check_install.py`: it overrides `paths.external_dir` in `config/services.yaml`.

The script refuses to move a clone with local changes. `third_party/pins.env` is the single source of truth for every
git commit, Hugging Face revision and wheel URL. The scripts accept only these pins.

## Checkpoints

```bash
bash scripts/download_checkpoints.sh --core --gen3c --coz             # without object insertion (~111 GB)
bash scripts/download_checkpoints.sh --all                            # everything (~168 GB)
bash scripts/download_checkpoints.sh --core --gen3c --coz --dry-run   # list files, sizes and what is missing
```

Several of these models are for non-commercial use only (GeometryCrafter, SD3 Medium, Qwen2.5-VL-3B-Instruct);
see [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md#model-weights-downloaded-by-scriptsdownload_checkpointssh).

| Group | Contents | Size | Destination |
|---|---|---|---|
| `--core` (required for generation) | OneFormer ADE20k Swin-L, Marigold normals v0-1, GeometryCrafter + SVD-xt image encoder/VAE, MoGe ViT-L, RepViT-SAM | ~12 GB | HF cache; `checkpoints/repvit_sam.pt` |
| `--gen3c` (camera moves, HQ views) | Gen3C-Cosmos-7B, Cosmos-Tokenize1-CV8x8x8-720p, T5-11B (without `tf_model.h5`) | ~76 GB | `checkpoints/gen3c/` |
| `--coz` (zoom-in) | Stable Diffusion 3 Medium (**gated**), Qwen2.5-VL-3B-Instruct | ~23 GB | HF cache |
| `--step1x` (objects) | Step1X-Edit v1.0 (`step1x-edit-i1258.safetensors`, `vae.safetensors`), Qwen2.5-VL-7B-Instruct | ~42 GB | `checkpoints/step1x/`; HF cache |
| `--objects` (objects) | GroundingDINO SwinT-OGC, SAM ViT-H, BERT base, SD2 inpainting (`sd2-community` mirror), INR-Harmonization | ~6.8 GB | `checkpoints/objects/`; HF cache |
| `--scenes` (`run.py --view`) | released scenes from `TmaKiss/WonderZoom` | ~7.8 GB | `gaussian/` |

- **Locations.** Hub files go to the Hugging Face cache, `$HF_HOME/hub` (default `~/.cache/huggingface/hub`). Gen3C,
  Step1X-Edit and the non-Hub files go to `$WZ_CKPT_DIR` (default `checkpoints/`). Put both on a fast local disk, and
  export the same `HF_HOME` and `WZ_CKPT_DIR` when you run the server: `config/services.yaml` reads `WZ_CKPT_DIR`.
- **Pinned and verified.** Every repository is downloaded at the revision in `third_party/pins.env` with an exact file
  list. Non-Hub files are checked against `third_party/checksums.sha256`. Re-running is safe: present files are skipped
  (size check; `--verify` also re-hashes them).
- **Gated SD3.** Accept the license, the Stability AI Non-Commercial Research Community License (non-commercial
  use only), on
  [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers),
  then run `huggingface-cli login` or `export HF_TOKEN=hf_...`. The script checks access first and stops with
  instructions and exit code 3 when access is missing. `--skip-gated` downloads everything else.
- **INR-Harmonization** weights are only hosted on Google Drive and are fetched with `gdown` (installed by
  `--objects`). If that fails, the script prints manual download steps. Only harmonization needs this file.
- **Other options:** `--only NAME[,NAME]` re-fetches single items (names are listed by `--dry-run`),
  `--ckpt-dir DIR`, `--python PATH`. `HF_ENDPOINT` (mirrors) and `HF_HUB_ENABLE_HF_TRANSFER=1` are honoured.
- **Cache variables.** Unset `TRANSFORMERS_CACHE`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE` and `DIFFUSERS_CACHE`; set
  only `HF_HOME`. Libraries may otherwise look in a different cache from the one the script filled. The download script and
  `check_install.py` warn about these variables.
- **Offline after download.** The Gen3C worker always runs with `HF_HUB_OFFLINE=1`. Once the `--coz` files are in the
  cache, also set `services.coz.hf_hub_offline: true` in `config/services.yaml`; `services.step1x` has the same
  key. `services.coz.sd3_model`, `services.coz.vlm_model` and `services.step1x.qwen_model` accept either a
  Hugging Face id or a local directory.

## Gen3C guardrail

By default, the Gen3C worker runs with GEN3C's safety guardrail disabled (`services.gen3c.disable_guardrail: true`),
as in the paper runs, the original WonderZoom service and GEN3C's GUI server. The Gen3C-Cosmos-7B and Cosmos
tokenizer weights are under the NVIDIA Open Model License, which says that your rights under it terminate if you
disable the model's safety guardrails. Read
[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md#nvidia-open-model-license-and-the-gen3c-guardrail) before you
decide how to run it.

To run Gen3C with the guardrail:

1. Accept the licenses of the two gated guardrail models with the Hugging Face account whose token is on the
   machine: [nvidia/Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1) (NVIDIA Open Model
   License, about 7 GB) and [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
   (Llama 3.1 Community License, about 16 GB without `original/`).
2. Download them into the Gen3C checkpoint directory (`services.gen3c.checkpoint_dir`, default
   `checkpoints/gen3c`). GEN3C looks for them at `<checkpoint_dir>/nvidia/Cosmos-Guardrail1` and
   `<checkpoint_dir>/meta-llama/Llama-Guard-3-8B`. `download_checkpoints.sh` does not fetch or verify them; the
   revisions below were the current ones when this guide was written.

   ```bash
   CK=${WZ_CKPT_DIR:-checkpoints}/gen3c
   huggingface-cli download nvidia/Cosmos-Guardrail1 --revision d6d4bfa899a71454a700907664f3e88f503950cf \
       --local-dir "$CK/nvidia/Cosmos-Guardrail1"
   huggingface-cli download meta-llama/Llama-Guard-3-8B --revision 7327bd9f6efbbe6101dc6cc4736302b3cbb6e425 \
       --exclude "original/*" --local-dir "$CK/meta-llama/Llama-Guard-3-8B"
   ```

3. Turn the guardrail on in `config/services.local.yaml` and restart the server:

   ```yaml
   services:
     gen3c:
       disable_guardrail: false
       offload_guardrail_models: true   # default: load the guardrail models for each check, then free them
   ```

The worker log (`runs/<example_name>/<time>/logs/gen3c.log`) then reports `guardrail enabled`. If the guardrail
models are missing, the worker stops at start-up with a fatal error that names the missing directories.

With the guardrail on, GEN3C checks the text prompt (a keyword blocklist and Llama Guard 3) and every generated
video (a video content filter). A rejected prompt or video fails the request with
`Gen3C returned no video (blocked by the guardrail?)`, and the scene is rolled back. The guardrail also blurs faces in
the generated frames, so results can differ from the paper's. With `offload_guardrail_models: true`, the guardrail
models are loaded from disk for every check, which makes each Gen3C request slower. `false` keeps them on the GPU
next to Gen3C: they are not parked under the `exclusive` policy, so use it only when Gen3C has a GPU of its own with
room to spare. The guardrail path has not been tested end to end with WonderZoom.

## Checking the installation

```bash
python scripts/check_install.py                 # every registered environment (wz-main, wz-gen3c, wz-coz after a default install)
python scripts/check_install.py --objects       # the same, plus the object-insertion stack (after install_all.sh --objects)
python scripts/check_install.py --all           # all four; fails when wz-step1x is not installed (it is only built with --objects)
python scripts/check_install.py --env main --objects
python scripts/check_install.py --env main --no-checkpoints --no-gpu   # before the download, on a node without a GPU
HF_HUB_OFFLINE=1 python scripts/check_install.py --env main --load-models core   # also load the main models offline
```

Use `--all` only after `install_all.sh --objects`; without object insertion, plain `check_install.py` checks the
three environments you built.

`check_install.py` needs only the Python standard library. It re-runs itself inside each environment with that
environment's registered interpreter and checks:
- package versions against the pinned requirement files;
- CUDA availability and the GPU architectures compiled into every extension, with a tiny kernel run for each
  (PyTorch3D, rasterizer, simple-knn, GroundingDINO, apex, transformer-engine, flash-attn, liger);
- the pinned third-party clones;
- each worker's imports, by running `services/workers/<svc>_worker.py --dry-import` with exactly the environment the
  server gives it (`step1x` is skipped with `--no-gpu`);
- checkpoint presence in the checkpoints directory and the HF cache (no network);
- Hugging Face access to the gated SD3 model.

Other options: `--no-gpu`, `--no-checkpoints`, `--offline`, `--json FILE`, `--verbose`, `--timeout SECONDS`. The
exit status is 0 when nothing failed; warnings are allowed.

More tests, run with the `wz-main` interpreter:

```bash
python tests/smoke_workers.py --selftest                       # protocol, restarts, GPU arbiter; no GPU or models
python tests/smoke_workers.py --service coz --seed 123         # one real worker: start-up, one request, peak memory
python tests/smoke_workers.py --service gen3c --steps 18 --suspend-resume --check-errors
python tests/smoke_workers.py --service step1x --offload true
python run.py --dry_run --example_config config/more_examples/street.yaml   # configs and imports only
python tests/e2e_headless.py --scenario boot,crack_fix,delete,undo,save    # against a running server
```

Each smoke test writes `runs/_smoke/<svc>-<time>/report.json` with the start-up time, the request time and the
peak GPU memory. The peak is reported both by the worker and by `nvidia-smi`.

## Troubleshooting

**Gated SD3 (`--coz` download exits with code 3, or the Chain-of-Zoom worker fails at start-up with a 401 / gated-repo
error).** Accept the license on the model page with the same Hugging Face account whose token is on the machine.
Then run `huggingface-cli login`, or export `HF_TOKEN`, and re-run `bash scripts/download_checkpoints.sh --coz`. Check
with `python scripts/check_install.py --env coz`. Once the files are cached, `services.coz.hf_hub_offline: true` stops
the worker from contacting the Hub at all.

**Transformer Engine and `CUDA_HOME` (`wz-gen3c`).** Transformer Engine 1.12 finds NVRTC through `CUDA_HOME`, both
when it is built and when it is imported. The install script sets `CUDA_HOME=$CONDA_PREFIX`. At run time,
`services.gen3c.cuda_home: auto` picks the first of these:
1. `<env>/targets/x86_64-linux` from the conda CUDA packages (fast);
2. an inherited `CUDA_HOME` with an NVRTC of the same major version;
3. the environment prefix.

Symptoms of a wrong value are NVRTC or JIT errors when the Gen3C worker starts, or an import that takes minutes:
TE globs `CUDA_HOME` recursively, which is slow over a whole environment on NFS. Keep `auto`, or set an explicit
toolkit directory that contains `lib*/libnvrtc.so*`. `check_install.py --env gen3c` reports the `CUDA_HOME` it would
use. When you run Gen3C code by hand inside `wz-gen3c`, export `CUDA_HOME=$CONDA_PREFIX`.

**Architecture mismatch (`no kernel image is available for execution on the device`, or `check_install.py` reports
`archs ...: no code for this GPU (sm_XX)`).** The extensions were built for other architectures. Rebuild with your
GPU's architecture included:

```bash
TORCH_CUDA_ARCH_LIST="8.9" bash scripts/install_env_main.sh --force-rebuild            # PyTorch3D, rasterizer, simple-knn
TORCH_CUDA_ARCH_LIST="8.9" bash scripts/install_objects_optional.sh --force-rebuild    # GroundingDINO
TORCH_CUDA_ARCH_LIST="8.9" bash scripts/install_env_gen3c.sh --force-rebuild           # transformer-engine, apex
```

Architectures outside 8.0-9.0 are untested.

**GPU out of memory.** On a single GPU, make sure the `exclusive` policy is active: the server log prints
`policy exclusive` for the main GPU. Force it with `--gpu_policy exclusive` or `WZ_GPU_POLICY=exclusive`, and keep
other processes off the GPU. Further steps, from cheapest to slowest:
1. `services.step1x.offload: true`, then `quantized: true` (FP8 cast of the DiT).
2. `geometrycrafter.low_memory_usage: true` in the scene config. This is automatic under `exclusive`.
3. Gen3C: `offload_network`, `offload_tokenizer`, `offload_text_encoder_model: true`. These reload the weights from
   disk for every request, which is slow.
4. A smaller `geometrycrafter.decode_chunk_size` (default 8). This may change results slightly.

Host-RAM OOM (the process is killed): loading Gen3C needs at least 96 GB of host RAM, and the `exclusive` policy
parks every idle model in RAM; see [HARDWARE.md](HARDWARE.md). `run_server.sh` already sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Slow cold start on NFS.** The first start reads tens of GB of weights. Measured on an L40S node:
- Gen3C worker: 669-676 s even with the checkpoints on a local disk (unpickling the legacy T5-11B checkpoint
  dominates), more than 60 min from slow NFS, and 108-122 s with a warm page cache;
- Chain-of-Zoom worker: 366 s from NFS;
- Step1X-Edit worker: 753 s cold from NFS, 231 s with a warm page cache.

Put `HF_HOME`, `WZ_CKPT_DIR` and the environments (`install_all.sh --prefix-root`) on a local NVMe disk if you can.
A worker that exceeds its start-up timeout is killed. The timeouts are `services.<svc>.init_timeout_s`: gen3c 3600 s,
coz 2400 s, step1x 1800 s. Raise them for very slow storage. The server keeps working while the workers load; the
status line shows `Waiting for <service> to finish loading...` when a request needs a worker that is not ready yet.

**The first request waits although its worker is ready (`exclusive` policy).** Under the `exclusive` policy, the
workers that share a GPU start one after another, and each is parked as soon as it is ready. A request that arrives
during this start-up can wait for the remaining workers to load, even when the worker it needs is already ready:
in the release test, the first camera move waited about 190 s for Chain-of-Zoom and Step1X-Edit to load, although
Gen3C was ready. Later requests do not wait. To avoid the delay, wait until the server log shows
`[services] <svc>: ready after N s` for every enabled worker before the first request, or turn off the workers you
do not need (`services.step1x.enabled: false` without object insertion).

**Models are not found, or are downloaded again, although the checkpoints are in place (`TRANSFORMERS_CACHE`,
`HF_HUB_CACHE`).** `TRANSFORMERS_CACHE` and `HF_HUB_CACHE` (or the older `HUGGINGFACE_HUB_CACHE`) override
`HF_HOME`. When one of them is set, for example by a shell profile, transformers or huggingface_hub look in that
directory instead of `$HF_HOME/hub`, the cache that `download_checkpoints.sh` filled. Check with
`env | grep -E 'HF_|TRANSFORMERS_CACHE'`, then unset them in the shell that runs the download script and the
server (`unset TRANSFORMERS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE`), or point them at the same cache.
`check_install.py` warns when `TRANSFORMERS_CACHE` is set.

**Results differ after a server restart.** Gen3C is not bit-deterministic across restarts, even with the same
config, the same start-up seed (`services.gen3c.seed`) and the same sequence of requests, because some of the CUDA
kernels it uses are nondeterministic. In the release test, the first camera moves of two server runs differed by
up to 92/255 in single pixels, and the scenes built from them differed by a few points out of about one million.
Chain-of-Zoom with a fixed seed (`services.coz.seed`) is deterministic within one server session: in the release
test, a zoom-in, an undo and the same zoom-in again gave identical results. With the default
`services.coz.seed: null`, every call draws a new seed, so a zoom-in repeated after an undo gives a different result.

**`pip.conf` with an extra index.** A global or user `pip.conf` that adds an unreachable `extra-index-url` makes pip
hang or fail during the install. Run the install with `PIP_CONFIG_FILE=/dev/null bash scripts/install_all.sh ...`.

**Worker logs.** Each worker's stderr goes to `runs/<example_name>/<time>/logs/<svc>.log`. A worker that fails at
start-up (for example because of a missing checkpoint) reports a fatal error that lists the missing files. It is
not restarted, and requests that need it fail with `<service> service failed to start; see <log>` in the UI.
