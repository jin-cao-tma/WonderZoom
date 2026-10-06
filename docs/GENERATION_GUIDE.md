# Generation guide

This guide assumes that the environments are installed and the checkpoints are downloaded
([INSTALL.md](INSTALL.md)).

- [Starting a session](#starting-a-session)
- [Workflows](#workflows): camera move, zoom-in, object insertion, HQ auxiliary views, crack fixing, deleting,
  undo and save, viewing saved scenes, exporting `.splat`
- [Your own images](#your-own-images)
- [Config reference](#config-reference): `run.py` flags, scene config keys, example scenes
- [config/services.yaml and GPU policies](#configservicesyaml-and-gpu-policies)
- [Optional features and OPENAI_API_KEY](#optional-features-and-openai_api_key)

## Starting a session

```bash
bash scripts/run_server.sh --example_config config/more_examples/street.yaml
# on your laptop:
ssh -L 7747:localhost:7747 <server>        # then open http://localhost:7747/
```

`scripts/run_server.sh` does the following, then starts `run.py`:
- uses the registered `wz-main` interpreter (no `conda activate` needed);
- keeps your current directory, so you can start it from anywhere;
- sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `PYTHONUNBUFFERED=1`;
- clears `LD_LIBRARY_PATH` (set `WZ_KEEP_LD_LIBRARY_PATH=1` to keep it);
- passes every argument through to `run.py`, and adds `--services_config <repo>/config/services.yaml` unless you
  give one.

Relative `--image`, `--example_config` and `--services_config` paths are resolved against your current directory
first, then against the repository root. `run.py` then changes into the repository root itself.

On start-up, the server does the following in order:
1. Opens the web UI at `http://<host>:<port>/` (default `127.0.0.1:7747`) with status `loading`.
2. Loads the main-process models.
3. Starts the model workers in the background. Under the `exclusive` policy they load one after another, and each
   is parked in host RAM as soon as it is ready.
4. Builds the initial scene from the input image.

If start-up or the initial scene fails, the server exits with an error. The status line shows `idle` when the
server accepts requests. A request that needs a worker that is still loading waits for it
(`Waiting for <service> to finish loading...`).

Stop the server with Ctrl+C (or `kill <pid>`, i.e. SIGTERM); it stops the model workers before it exits.

The page (`splat-main/index_gen.html`) shows:
- the live preview, with the key table next to it;
- the object prompt box at the top;
- three video panels: **Rough Video** (the current scene rendered along the trajectory, holes included),
  **Output Video** (the generated video) and **Concatenated Video**;
- in the bottom bar, the scene statistics and the session directory.

With `--view` the page shows only the live view, the view keys and the bottom bar (see
[Viewing saved scenes](#viewing-saved-scenes)).

Rejected requests (busy, wrong number of trajectory points, service disabled) are explained in the message line.

Every run writes a session directory:

```
runs/<example_name>/<YYYYmmdd-HHMMSS>/
├── config.yaml     # the merged scene config of this session
├── scenes/         # saved scenes (X): <example_name>_000.pth, _001.pth, ...
├── logs/           # worker logs: gen3c.log, coz.log, step1x.log
├── services/       # per-request worker outputs: services/<svc>/<NNNN>_<HHMMSS>/
├── cache/, frames/ # intermediate images and videos of the last jobs
```

The main process changes into this directory once start-up is done, so concurrent sessions do not overwrite each
other's files.

## Workflows

All generation requests run one at a time. While a job runs, the status line shows `busy [<job>]`, and other
scene-changing keys are rejected.

Two keys select a camera trajectory:
- **H** adds the current view as a trajectory point;
- **J** clears the trajectory.

**R** decides between the two generation modes by the focal length. Above the initial focal length (after
pressing **V**), R starts a **zoom-in**. At the initial focal length (**B** returns to it), R starts a
**camera move**.

### Camera move (Gen3C)

1. Make sure you are at the initial focal length (press **B** until it stops changing).
2. Navigate with **W/A/S/D** (rotate), the arrow keys (move) and **N/M** (down/up).
3. Optional: press **H** at intermediate views to add waypoints.
4. At the target view, press **R**.

The trajectory starts at the nearest already generated pose, passes through the H waypoints and ends at the
current view. It is sampled into 121 frames (`interpolate_cameras_RT`). The current scene is rendered along it,
and the holes are masked. Gen3C fills them with `gen3c_steps_move` (18) diffusion steps. GeometryCrafter
estimates the video depth, which is aligned to the rendered depth of the scene. New Gaussians are added only
where the scene is empty, and only the new content is trained.

### Zoom-in (Chain-of-Zoom), with optional object insertion

1. At the start view, press **H** once. Exactly one point is allowed; press **J** to clear extra points.
2. Press **V** repeatedly to zoom in (the focal length grows by 5% per press). You can also re-aim with the
   movement keys.
3. Press **R**.

The server interpolates the focal length and pose over 49 frames (`interpolate_cameras_K`). It renders three
keyframes and super-resolves the two finer ones with Chain-of-Zoom, each from the previous scale. The depth of the
new content is registered to the existing scene; by default this includes a short online fine-tune of MoGe,
`num_finetune_depth_model_steps` (100). The new scale is added as Gaussians that fade in as you zoom.

**Rewrite and overwrite modes (Q).** In the default **rewrite** mode, the zoomed region is re-generated and merged
into the scene. **Overwrite** mode trains only the newly added points, and re-generates objects inserted earlier
when the zoom sees them. Q toggles the mode for the following zooms; the message line confirms the new state.

**Object insertion.** This needs the optional object stack; see
[Optional features](#optional-features-and-openai_api_key).
1. Type an object into the box at the top, for example `a ladybug`.
2. Press Enter or **Set**. The box shows the current prompt.
3. Do a zoom-in.

At the end of the zoom:
1. Step1X-Edit inserts the object into the finest keyframe. The edit prompt comes from GPT-4o, or from
   `object_edit_prompt_template` without it.
2. GroundedSAM segments the object.
3. The object is harmonized with INR-Harmonization (`use_harmol`, if installed).
4. SD2 inpainting fills the background behind the object.
5. The object is lifted to 3D as its own label, at the new scale.

Without GPT-4o, the default template `'{object} is on the ground'` only works when the zoomed view shows ground or a
similar surface: aim the zoom at one, or set `object_edit_prompt_template` in the scene config to fit the view (for
example `'{object} is on the window ledge'`). If GroundedSAM finds no object in the edited keyframe, the zoom-in
fails with `No mask found for object '...'`, the scene is rolled back to its state before the zoom and the prompt
is kept for the next zoom-in.

The prompt applies to the next zoom only and is then cleared; submitting an empty prompt also clears it. Without
the object stack, the prompt box is disabled. A prompt sent anyway is answered with
`object insertion unavailable: <reasons>`, and the zoom runs without insertion.

### High-quality auxiliary views (Ctrl+Shift+Space)

Use this to fill the disocclusions that become visible when you orbit around a view.

1. Press **Space** for an orbit preview (22 frames, no scene change).
2. Press **Ctrl+Shift+Space**.

The server records a 121-frame orbit around the current view (`generate_orbit_cameras`, or the scene config's
`generate_orbit_cameras_code`). Gen3C fills the holes with `gen3c_steps_hq` (10) steps, and its output is used only
inside the holes. The scene is then refined on the new views. This needs the Gen3C service; the key is dimmed in
the UI when Gen3C is disabled.

### Crack fixing (Ctrl+Alt+Space)

This is a cheaper variant without Gen3C for small cracks between scales. The server records a 49-frame orbit,
inpaints the small holes with OpenCV (`cv2.INPAINT_TELEA`) and refines the scene on these frames.

### Background completion (P), object stack only

**P** inpaints the background behind the foreground objects of the current view and adds it to the scene. It needs
the object stack, plus either GPT-4o or the scene config's `foreground_words` (and optionally `background_prompt`).
The P row of the key table appears only when object insertion is available.

### Delete (C)

**C** deletes the Gaussians visible in the current view.

### Undo (Z) and save (X)

- **Undo is single-level.** It reverts the last scene-changing job: camera move, zoom-in, HQ views, crack fix,
  background completion or delete. The Gaussians, labels, generated poses and camera memory are all restored.
  A job that fails is rolled back automatically; there is then nothing left to undo, and the server reports
  `Nothing to undo`.
- **Save** writes `<session>/scenes/<example_name>_<NNN>.pth`, and the message line shows `Saved: <path>`. You can
  save several times per session; the files are numbered.

While a job runs, the preview pauses whenever the main models work on the scene. During Gen3C, Chain-of-Zoom and
Step1X-Edit calls, the preview keeps streaming under the `resident` policy and pauses under `exclusive`.

### Viewing saved scenes

Saved scenes use the same format as the released ones. Load one with `run.py --view`, and pass the session's
`config.yaml`: it records the generation resolution, which the viewer needs for the principal point, and the
scene's orbit code. View mode loads no generation model and starts no worker, so it needs only `wz-main`. Relative
paths are resolved against your current directory first, then against the repository root. Use a port other than
the generation server's if that is still running.

```bash
conda activate wz-main          # or call the interpreter printed by: python3 scripts/register_env.py --get main
python run.py --view --pth_path runs/street/<time>/scenes/street_000.pth \
    --example_config runs/street/<time>/config.yaml --port 7748
```

`bash scripts/run_server.sh --view ...` works as well. Then open `http://localhost:7748/` (forward port 7748 first
if the machine is remote: `ssh -N -L 7748:localhost:7748 <server>`). The page hides the generation controls; a
generation key is refused with `<action> ignored: view mode (--view): generation is off; ...`. The view keys
are listed in the [README](../README.md#quick-start-view-a-released-scene).

### Exporting a `.splat` file

`tools/export_splat.py` writes a saved scene as a `.splat` file for web viewers:

```bash
python tools/export_splat.py --scene runs/street/<time>/scenes/street_000.pth --out street.splat
python tools/export_splat.py --scene ... --out street.splat --mode fill
python tools/export_splat.py --scene ... --out ladybug.splat --mode sky --shift 5.1574e-3 -4.0808e-3 6.8847
```

| Mode | What it does | Needs |
|---|---|---|
| `plain` (default) | the scene as it is | nothing (no models) |
| `fill` | re-generates the background hidden behind inserted objects from the origin view, then exports | the main models, loaded like `run.py` (OneFormer, Marigold, RepViT-SAM, MoGe/GeometryCrafter) |
| `sky` | `fill` (skip it with `--no-fill`) plus a regenerated sky layer (skip it with `--no-sky`), shifted by `--shift X Y Z` | as `fill` |

`--config` defaults to the session's `config.yaml`, next to `scenes/`. The script header lists the shifts used for
the paper's web demos.

## Your own images

```bash
bash scripts/run_server.sh --image /abs/path/to/photo.jpg --name my_scene
```

`--image` merges `config/custom_template.yaml` over `config/base-config.yaml`. It sets `image_filepath`, and sets
`example_name` from `--name` (default: the file stem). The session goes to `runs/my_scene/<time>/`.

- The image is resized and center-cropped to `gen_W x gen_H` (1088x720 by default), so a landscape 3:2 photo works
  best. Transparent PNGs are converted to RGB.
- To keep tuned settings, copy `config/custom_template.yaml` to, for example, `config/my_scene.yaml`. Set
  `image_filepath` (relative to the repository root) and `example_name`, then start with
  `--example_config config/my_scene.yaml`.
- The settings most worth tuning per scene are `amount`, `use_scaling_pull` and `use_tilt` (see below),
  `init_focal_length`, and the orbit code `generate_orbit_cameras_code`. Copy one from `config/more_examples/`.
- Without GPT-4o, set `foreground_words` (and optionally `background_prompt`) if you want background completion (P)
  or `pull_foreground_depth_rewrite`.

About randomness: Chain-of-Zoom draws a new seed for every call from the main process's random stream, which is
seeded with `seed`, as in the paper runs. Gen3C uses its own random stream, seeded once at start-up. The same
config with the same sequence of requests therefore reuses the same seeds, while a different sequence gives
different results. To fix the seeds per request instead, see `services.coz.seed` and
`services.gen3c.request_seed` [below](#configservicesyaml-and-gpu-policies).

## Config reference

### `run.py` flags (`bash scripts/run_server.sh` passes them through)

| Flag | Default | Meaning |
|---|---|---|
| `--example_config PATH` | `config/more_examples/street.yaml` | scene config, merged over the base config (the scene config wins) |
| `--base_config PATH` | `config/base-config.yaml` | base config |
| `--image PATH` | none | generate from your own image (uses `config/custom_template.yaml` unless `--example_config` is given) |
| `--name NAME` | file stem | session name (`example_name`) |
| `--view` | off | view a saved scene only: `--pth_path`, else the config's `pth_path`, rendered at `orig_H` x `orig_W` (else `gen_H` x `gen_W`); loads no generation model, starts no worker, needs only `wz-main`; `--image` and `--name` are rejected |
| `--pth_path PATH` | config's `pth_path` | scene to view; implies `--view` |
| `--services_config PATH` | `config/services.yaml` | model services config |
| `--no_services` | off | do not start Gen3C, Chain-of-Zoom or Step1X-Edit: build, view and edit the initial scene only (for a saved scene use `--view`) |
| `--gpu_policy {auto,resident,exclusive}` | from `services.yaml` | override `gpu.policy` |
| `--main_gpu N` | from `services.yaml` | logical GPU index of the main process (`gpu.main_device`) |
| `--host ADDR` | `127.0.0.1` | bind address; `0.0.0.0` serves other machines |
| `--port N` | `7747` | port of the web UI and Socket.IO |
| `--stream_max_size N` | `256` (`1080` with `--view`) | longest edge (pixels) of the streamed preview frames |
| `--stream_quality N` | `20` (`80` with `--view`) | JPEG quality (1-100) of the preview frames |
| `--debug` | off | open a post-mortem debugger when a request fails: ipdb if installed (`pip install ipdb`), else pdb (default: log, roll back, continue) |
| `--dry_run` | off | resolve the configs, import every module and exit before loading any model (with `--view`: print the scene path, render size and stream settings) |

The defaults keep the preview light over SSH. On a fast connection, try `--stream_max_size 1088 --stream_quality 80`.
If the `--view` stream (1080 / 80) lags, lower it, e.g. `--stream_max_size 512 --stream_quality 40`.

### Scene config keys

`config/base-config.yaml` lists every key the code reads, with its default and a comment. The most relevant keys:

| Key | Default | Meaning |
|---|---|---|
| `image_filepath` | | input image, relative to the repository root |
| `example_name` | image stem | session name: `runs/<example_name>/<time>/` |
| `seed` | `1` | seed of the main process (also drives the per-call Chain-of-Zoom seeds) |
| `gen_H`, `gen_W` | `720`, `1088` | generation resolution; both must be multiples of 16 (Chain-of-Zoom) |
| `init_focal_length` | `1024` | focal length (px) of the input view; zoom-in is relative to it |
| `depth_shift`, `sky_hard_depth` | `0.001`, `0.02` | depth offset and sky depth of the initial scene |
| `initial_iterations` | `300` | Gaussian optimization steps for the initial scene |
| `num_finetune_depth_model_steps` | `100` | online MoGe fine-tuning steps per zoom-in (0 disables it) |
| `amount` | `2` | depth extent of inserted objects (larger = flatter) |
| `use_scaling_pull` | `true` | align new zoom-in depth to the scene per RepViT-SAM segment |
| `use_tilt` | `true` | fit a planar tilt of inserted objects to the scene depth |
| `pull_foreground_depth_rewrite` | `false` | pull GroundedSAM foreground objects to their median depth during rewrite zooms (object stack + `foreground_words` or GPT) |
| `geometrycrafter.decode_chunk_size` | `8` | GeometryCrafter VAE decode chunk |
| `geometrycrafter.low_memory_usage` | `null` | `null`: on under the `exclusive` policy, off otherwise (intermediates in host RAM; results match up to floating-point differences) |
| `generate_orbit_cameras_code` | `null` | Python source of `generate_orbit_cameras(...)` for the orbits (Space, Ctrl+Shift+Space, Ctrl+Alt+Space); **executed as code**, so only run configs you trust |
| `gen3c_prompt` | `'None'` | Gen3C text prompt; the paper runs sent the literal string `'None'` |
| `gen3c_prompt_hq` | `gen3c_prompt` | prompt for the HQ views (leave it unset rather than `null`) |
| `gen3c_steps_move`, `gen3c_steps_hq` | `18`, `10` | Gen3C diffusion steps for camera moves and HQ views |
| `use_harmol` | `true` | harmonize inserted objects (skipped with a warning if INR-Harmonization is not installed) |
| `object_edit_prompt_template` | `'{object} is on the ground'` | Step1X-Edit prompt without GPT-4o; `{object}` is the text typed in the UI |
| `foreground_words`, `background_prompt` | `[]`, `''` | foreground objects and background description for P and `pull_foreground_depth_rewrite` when GPT-4o is not used |
| `pth_path`, `orig_H`, `orig_W` | | used only by `run.py --view` (scene file and its render size; `--pth_path` overrides `pth_path`); generation ignores them |

### Example scenes (`config/more_examples/`)

| Config | Input image | Generation size | Released scene (`--view`) |
|---|---|---|---|
| `street.yaml` | `street.png` | 720x1088 | `gau_bird3_complete1080.pth` |
| `fish.yaml` | `coral.png` | 720x1088 | `gau_fish1_complete1080.pth` |
| `tree.yaml` | `Tree.png` | 720x1088 | `gau_beetle1_complete1080.pth` |
| `beach.yaml` | `beach.jpg` | 720x1088 | `gau_conch1_complete1080.pth` |
| `sunflower.yaml` | `sunf.png` | 720x1088 | `gau_ladybug1_complete1080.pth` |
| `wooden.yaml` | `wooden.jpg` | 720x1088 | `gau_lizard_complete1080.pth` |
| `tea_garden.yaml` | `tea_garden.jpg` | 720x1088 | `gau_butterfly_complete1080.pth` |
| `lego.yaml` | `deng.png` | 480x720 | `gau_lego_complete.pth` |
| `beach2.yaml` | `beach2.png` | 480x720 | `gau_conch_complete.pth` |
| `tea_garden2.yaml` | `tea_garden2.png` | 480x720 | `gau_butterfly_complete.pth` |
<!-- VERIFY: generation at 480x720 (lego, beach2, tea_garden2) has not been run end to end yet. -->

`sunflower.yaml` sets `foreground_words: [sunflower]` and `pull_foreground_depth_rewrite: True`, so it works without
GPT-4o (the object stack is still needed for GroundedSAM).

## `config/services.yaml` and GPU policies

`config/services.yaml` configures the model workers, GPU placement and checkpoint paths. Values are applied in
this order, later ones winning:
1. `config/services.yaml`;
2. `config/services.local.yaml`, which is machine-specific and gitignored. The install scripts write the
   interpreter paths there, and you can add your own keys.
3. environment variables:
   - `WZ_GPU_POLICY`: `gpu.policy`;
   - `WZ_CKPT_DIR`: `paths.checkpoints_dir`;
   - `WZ_EXTERNAL_DIR`: `paths.external_dir` (use the same value as for the install scripts);
   - `WZ_GEN3C_PYTHON`, `WZ_COZ_PYTHON`, `WZ_STEP1X_PYTHON`: the worker interpreters;
4. `run.py` flags: `--gpu_policy`, `--main_gpu`, `--no_services`.

Relative paths are relative to the repository root.

### Sections

| Section | Keys |
|---|---|
| `gpu` | `policy` (`auto`), `main_device` (0), `pause_render_during_services`, `render_pause_timeout_s`, `reserve_gb` (4), `main_resident_gb` (16), `restore_main_after_service` (false) |
| `paths` | `external_dir` (`$WZ_EXTERNAL_DIR` or `external`), `checkpoints_dir` (`$WZ_CKPT_DIR` or `checkpoints`), `runs_dir` (`runs`) |
| `services.gen3c` | `enabled`, `python`, `repo_dir`, `device`, `cuda_home` (`auto`), `hf_hub_offline` (true), timeouts, `resident_gb` (36), `checkpoint_dir`; paper settings (`guidance` 1.0, 1280x704, 24 fps, 121 frames, `seed` 1, `legacy_none_prompt`); `request_seed` (null); guardrail: `disable_guardrail` (true), `offload_guardrail_models` (true); memory: `cache_text_embeddings`, `park_text_encoder` (`auto`), `park_tokenizer`, `offload_network`, `offload_tokenizer`, `offload_text_encoder_model` |
| `services.coz` | `enabled`, `python`, `repo_dir`, `device`, `hf_hub_offline` (false), timeouts, `resident_gb` (28), `lora_path`, `vae_path`, `sd3_model`, `vlm_model`, `lora_rank`, `seed` (null), `pad_to_multiple_of_16` |
| `services.step1x` | `enabled`, `python`, `repo_dir`, `device`, `hf_hub_offline`, timeouts, `resident_gb` (42), `checkpoint_dir`, `qwen_model`, `offload` (`auto`), `quantized` (false) |
| `features` | `object_insertion`, `harmonization`, `gpt`: each `true`, `false` or `auto` |
| `objects` | `groundingdino_checkpoint`, `groundingdino_config`, `sam_checkpoint`, `inr_repo_dir`, `inr_checkpoint`, `inpaint_model` |
| `main_models` | `repvit_sam_checkpoint` |

Useful keys:
- `services.<svc>.enabled: false` turns a worker off. A worker whose interpreter is not registered is also off.
- `services.coz.seed`: an integer fixes the Chain-of-Zoom seed. The default `null` draws a new one per call, as in the paper.
- `services.gen3c.request_seed`: an integer reseeds Gen3C before every request.
- `services.gen3c.disable_guardrail`: `true` (default, as in the paper runs) runs Gen3C without GEN3C's safety
  guardrail. The NVIDIA Open Model License of the Gen3C weights says your rights under it terminate if you disable
  the model's safety guardrails; see
  [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md#nvidia-open-model-license-and-the-gen3c-guardrail).
  `false` runs the guardrail, which needs two extra downloads ([INSTALL.md](INSTALL.md#gen3c-guardrail)), can reject
  requests and blurs faces in the generated frames. `offload_guardrail_models: true` (default) loads the guardrail
  models for each check instead of keeping them on the GPU.
- `services.<svc>.init_timeout_s` and `request_timeout_s`: a worker that crashes or times out is killed. It is
  restarted on the next request; a worker that reported a fatal start-up error is not restarted.
- `services.coz.sd3_model`, `services.coz.vlm_model` and `services.step1x.qwen_model` accept a Hugging Face id or a
  local directory.

### GPU policies

The main process and the three workers are *tenants*, each on one GPU. The policy is decided per physical GPU:

| Policy | Behavior |
|---|---|
| `resident` | Every tenant keeps its weights on its GPU. Fastest; needs enough memory for all tenants of a GPU at once. |
| `exclusive` | Tenants that share a GPU take turns. Before a tenant runs, the others on that GPU are parked in host RAM (workers through a `suspend` request, the main models through `run.py`'s park function). The 3DGS scene always stays on the GPU. Workers start one after another and are parked right after loading. |
| `auto` (default) | `exclusive` on every GPU whose tenants' `resident_gb` sum exceeds its memory minus `gpu.reserve_gb`; `resident` otherwise. A GPU with a single tenant is always resident. |

Details:
- Under `exclusive`, the main models stay parked after a worker call until the main process needs them again.
  `gpu.restore_main_after_service: true` moves them back right away, at the cost of extra transfers.
- `gpu.pause_render_during_services` pauses the preview while a worker uses the main process's GPU.
- `services.step1x.offload: auto` keeps the paper placement (no offload) when Step1X-Edit's `resident_gb` fits.
  Under `resident`, it must fit next to the other resident tenants; under `exclusive`, in the GPU minus
  `gpu.reserve_gb`. Otherwise it offloads, moving Qwen2.5-VL-7B and the DiT to the GPU and back on every edit.
- `services.gen3c.park_text_encoder: auto` keeps T5-11B in host RAM between prompts under `exclusive`. T5 embeddings
  are cached per prompt.
- Device indices (`gpu.main_device`, `services.*.device`) are logical indices into the `CUDA_VISIBLE_DEVICES` that
  `run.py` was started with; all GPUs when it is unset.
- `auto` looks up GPU memory by `nvidia-smi` index. On nodes with mixed GPU models, set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` or choose the policy explicitly.

### Profiles

Put the machine-specific layout into `config/services.local.yaml`.

**A. One GPU with ≥46 GB** (L40S, RTX 6000 Ada, A6000 48 GB, A100/H100 80 GB). This is the default: everything on
GPU 0. The summed `resident_gb` estimates exceed one GPU, so `auto` resolves to `exclusive`, and GeometryCrafter
switches to low-memory mode. On 46-48 GB cards, Step1X-Edit also offloads. Plan for ≥192 GB of host RAM, or 256 GB
to be safe.

```yaml
gpu: {policy: auto, main_device: 0}
services: {gen3c: {device: 0}, coz: {device: 0}, step1x: {device: 0}}
```

**B. Two GPUs with ≥46 GB.** The main process gets its own GPU and is never parked; the three workers share GPU 1
(`exclusive`). On 46 GB cards, Step1X-Edit offloads. Plan for ≥192 GB of host RAM, or 256 GB to be safe.

```yaml
gpu: {policy: auto, main_device: 0}
services: {gen3c: {device: 1}, coz: {device: 1}, step1x: {device: 1}}
```

**C. Four GPUs with ≥48 GB.** Every tenant has its own GPU; nothing is swapped.

```yaml
gpu: {policy: resident, main_device: 0}
services: {gen3c: {device: 1}, coz: {device: 2}, step1x: {device: 3}}
```

Measured memory and timings: [HARDWARE.md](HARDWARE.md). Profiles A (one L40S 46 GB) and B (2x L40S 46 GB) were tested end to end: camera move, zoom-in, undo, HQ views,
object insertion, save. Profile C is untested.

## Optional features and OPENAI_API_KEY

The features are resolved at start-up. `run.py --dry_run` prints what is missing.

| Feature (`features.*`) | `auto` turns it on when | Without it |
|---|---|---|
| `object_insertion` | the step1x worker is enabled (`wz-step1x` registered), `groundingdino` and `segment_anything` are installed in `wz-main` (`install_all.sh --objects`), and `objects.groundingdino_checkpoint` and `objects.sam_checkpoint` exist (`download_checkpoints.sh --step1x --objects`) | the prompt box is disabled, P is hidden, and zoom-ins run without insertion |
| `harmonization` | `external/INR-Harmonization` with the wrapper (`setup_third_party.sh inr`), `objects.inr_checkpoint`, `adamp` and `albumentations` are present | inserted objects use the raw Step1X-Edit edit (`use_harmol` falls back, with one warning) |
| `gpt` | `OPENAI_API_KEY` is set (the `openai` package comes with `--objects`) | the config fallbacks below are used |

**`OPENAI_API_KEY` is optional.** GPT-4o is used only by the object features:

| Use | Without GPT-4o |
|---|---|
| Step1X-Edit prompt for an inserted object | `object_edit_prompt_template` (`'{object} is on the ground'`) |
| Foreground words and background description for background completion (P) | `foreground_words` / `background_prompt` from the scene config; P fails with an error if `foreground_words` is empty |
| Foreground words for `pull_foreground_depth_rewrite` | `foreground_words`; skipped with a warning if empty |

Camera moves, HQ views and crack fixing never call GPT-4o; a zoom-in calls it only for object insertion or
`pull_foreground_depth_rewrite`. Set `features.gpt: false` to keep GPT off even
when the key is set. For `pull_foreground_depth_rewrite`, a non-empty `foreground_words` takes precedence over GPT-4o.
For P, GPT-4o is used whenever it is available, and the config values are the fallback.
