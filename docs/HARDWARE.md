# Hardware

The end-to-end numbers below come from one node with **2x NVIDIA L40S (46,068 MiB each, sm_89)**, 16 CPUs and
755 GB of host RAM, in [profile B](#gpu-profiles): the main process on GPU 0, and Gen3C, Chain-of-Zoom and
Step1X-Edit taking turns on GPU 1 under the `exclusive` policy. The checkpoints and the HF cache were on a local
disk. The single-service numbers come from the smoke tests (`tests/smoke_workers.py`), with one worker alone on one
L40S.

## Summary

| | Viewing (`run.py --view`) | Generation |
|---|---|---|
| GPU | one NVIDIA GPU, compute capability 8.0 / 8.6 / 8.9 / 9.0 (default build); peak 5,258 MiB (`nvidia-smi`) for the largest released scene (`gau_bird3_complete1080.pth`, 18.4 M Gaussians) | profile B: two GPUs with ≥46 GB (tested); profile A: one GPU with ≥46 GB (tested end to end on one L40S 46 GB); profile C: four GPUs with ≥48 GB (untested); see [Profiles](#gpu-profiles) |
| Host RAM | about 1.3 GB RSS measured for the largest released scene | ≥192 GB recommended, 256 GB to be safe (idle models are parked in RAM); measured peak RSS of the server and its workers: 135 GB (profile B, with object insertion); loading Gen3C alone needs ≥96 GB |
| Disk | ~7.8 GB scenes + `wz-main` env (12 GB) | ~168 GB checkpoints (`download_checkpoints.sh --all`) + four envs (about 40 GB); a fast local disk is recommended |
| CPU | any | building the environments is CPU-bound (see [Build times](#build-times)) |

The default build compiles for compute capability 8.0, 8.6, 8.9 and 9.0 (`TORCH_CUDA_ARCH_LIST`); other GPUs need a
rebuild ([INSTALL.md](INSTALL.md#build-settings)) and are untested.

## GPU profiles

| Profile | Layout (`config/services.yaml`) | Policy | Status |
|---|---|---|---|
| A. one GPU ≥46 GB | everything on GPU 0 | `auto` → `exclusive`; Step1X-Edit offload on | **tested end to end** on one L40S 46 GB: camera move, zoom-in, undo, HQ views, object insertion, save. Start-up to idle 223-253 s, camera move 953-966 s, zoom-in 144-170 s, HQ views 373-377 s, zoom-in with object insertion 299-317 s, save < 3 s; GPU peak 39,971 MiB (`nvidia-smi`, during a camera move) |
| B. two GPUs ≥46 GB | main on 0; gen3c, coz, step1x on 1 | `auto` → `exclusive` on GPU 1 only; the main models are never parked; Step1X-Edit offload on with 46 GB | **tested end to end** on 2x L40S 46 GB: camera move, zoom-in, undo, HQ views, object insertion, save |
| C. four GPUs ≥48 GB | main 0, gen3c 1, coz 2, step1x 3 | `resident` | untested |

The `auto` policy decides with the per-tenant `resident_gb` estimates from `config/services.yaml`: main 16, gen3c
36, coz 28, step1x 42 (GiB). These are configuration estimates, not measurements. The profile B test set
`gpu.policy: exclusive`, which gives the same placement as `auto` there: GPU 1 `exclusive`, GPU 0 (one tenant)
`resident`.

In profile B on 46 GB cards, the main GPU peaked at 38,857 MiB during camera-move processing and at 32,983 MiB in an
object-insertion run. Keep other processes off that GPU.

## Single-service measurements (one L40S 46 GB)

Each worker ran alone on one L40S in the smoke tests, under the `exclusive` policy. The "end-to-end" values come
from the profile B runs, which followed earlier runs on the same machine, so most weights were in the page cache.
Peak allocated memory is `torch.cuda.max_memory_allocated()` inside the worker, in GiB. The `nvidia-smi` peak
includes the CUDA context and allocator overhead; it was sampled every 500 ms.

| Service | Start-up (load) | First request | Later requests | Peak allocated | `nvidia-smi` peak |
|---|---|---|---|---|---|
| Chain-of-Zoom, one dual-scale SR call at 1088x720 | 366 s from NFS; 31-75 s end to end | 13 s | 5 s | 31.7 GiB | 33,047 MiB |
| Step1X-Edit, `offload` on (default under `exclusive` on 46 GB), `size_level` 512 | 753 s cold from NFS; 231 s with a warm page cache; 83-117 s end to end | 332 s | 48 s per edit (88 s and 165 s end to end) | 23.9 GiB | 25,199 MiB |
| Step1X-Edit, `offload` off | 778 s cold from NFS | 74 s | 15 s per edit | 39.9 GiB (upstream figure: 42.5 GB) | 41,713 MiB |
| Gen3C, camera move (121 frames at 1280x704, 18 steps) | 669-676 s with the checkpoints on a local disk (unpickling the legacy T5-11B checkpoint dominates; more than 60 min from slow NFS); 108-122 s end to end | 529 s | 484-502 s | 34.2 GiB | 35,591 MiB (36,521 MiB end to end) |
| Gen3C, HQ views (121 frames, 10 steps) | | 263 s of denoising (end to end) | | as for camera moves | not measured separately |
| Main process (models + scene) | models 10-13 s and initial scene 12.7 s (end to end); about 60 s from launch to `idle` with a warm cache; about 30 min cold from NFS | | | not recorded | GPU 0 in profile B: 38,857 MiB during camera-move processing, 32,983 MiB in an object-insertion run; about 12 GB with `--no_services` |

The smoke tests require a parked (suspended) worker to keep less than 1.5 GiB allocated on the GPU. Chain-of-Zoom
and Step1X-Edit pass with 0.01 GiB, and Gen3C passes as well. Gen3C and Chain-of-Zoom give identical output before
and after a suspend/resume cycle.

Park and resume times in the end-to-end run (GPU 1):

| Worker | Park | Resume |
|---|---|---|
| Gen3C | 6.7-15.5 s | 1.7-9.1 s |
| Chain-of-Zoom | 17.6-29.0 s | 10.5-22.6 s |
| Step1X-Edit (`offload` on) | 0.2 s | 0.0 s |

## End-to-end timings (profile B, 2x L40S)

`street.yaml`, driven by `tests/e2e_headless.py`. Ranges cover two runs.

| Operation | Wall-clock | Notes |
|---|---|---|
| Server start until `idle` (`street.yaml`) | about 60 s with a warm cache; about 30 min cold from NFS | main models + initial scene; workers may still be loading |
| Server start until all workers are ready | about 5 min with a warm page cache: Gen3C 108-122 s, Chain-of-Zoom 31-75 s, Step1X-Edit 83-117 s, each followed by parking | exclusive policy: workers load one after another; cold start-up times are in the table above |
| Camera move (R at the base focal length) | 984-996 s | Gen3C 18 steps (T5 15 s + 483 s of denoising) + about 300 s on the main side (GeometryCrafter, depth alignment, training). The first move also waited about 190 s for Chain-of-Zoom and Step1X-Edit to finish loading ([INSTALL.md](INSTALL.md#troubleshooting)) |
| Zoom-in (H, V×n, R) | 100-149 s | two Chain-of-Zoom calls (4-28 s each) + depth registration + training; the first zoom-in after a camera move includes swapping Gen3C for Chain-of-Zoom (about 20 s) |
| Zoom-in with object insertion | 330 s | adds Step1X-Edit (88 s, offload on), GroundedSAM, SD2 inpainting, INR, and swapping Chain-of-Zoom for Step1X-Edit (18 s). A zoom-in whose view did not fit the edit prompt failed after 348 s and was rolled back |
| HQ auxiliary views (Ctrl+Shift+Space) | 363-433 s | Gen3C 10 steps (263 s of denoising, 337 s with the swap from Chain-of-Zoom) + refinement |
| Crack fixing (Ctrl+Alt+Space) | 54-64 s | no Gen3C (measured with `--no_services`) |
| Undo / save | undo 0.1 s; save 2.8-4.2 s | a saved scene was 272-428 MB |

Under the `exclusive` policy, every switch between the main models and a worker parks one and resumes the other,
which adds to each operation's time. A switch between two workers took 14-38 s in the end-to-end run. In profile A,
the main models are parked and resumed as well (not measured).

## Host RAM

- Loading Gen3C needs at least 96 GB of host RAM. The T5-11B checkpoint is a legacy pickle (about 45 GB) and the DiT
  is stored in fp32 (about 29 GB).
- Under the `exclusive` policy, every idle model is parked in host RAM: Gen3C, Chain-of-Zoom, Step1X-Edit and, on a
  single GPU, the main models. Plan for ≥192 GB, or 256 GB to be safe.
- Measured peak (profile B, sampled every 2 s): the process-tree RSS of `run.py` and its workers was 102 GB during
  camera moves, zoom-ins and HQ views, and 135 GB with object insertion. Per-process peaks: Step1X-Edit 44.4 GB
  (offloaded DiT and Qwen2.5-VL-7B in host RAM), Gen3C 47.3 GB while loading and about 34.7 GB when parked,
  Chain-of-Zoom 30.0 GB, main process 16.9 GB, Gen3C's inductor compile subprocesses about 10.6 GB. In profile A,
  the parked main models come on top (not measured).

## Disk

| Item | Size |
|---|---|
| `download_checkpoints.sh --core` | ~12 GB |
| `--gen3c` | ~76 GB |
| `--coz` | ~23 GB |
| `--step1x` | ~42 GB |
| `--objects` | ~6.8 GB |
| `--scenes` | ~7.8 GB |
| `--all` | ~168 GB |
| Gen3C guardrail models, only with `services.gen3c.disable_guardrail: false` ([INSTALL.md](INSTALL.md#gen3c-guardrail)) | ~23 GB (not in `--all`) |
| environments `wz-main`, `wz-gen3c`, `wz-coz`, `wz-step1x` | 12 GB (with the object stack), 13 GB, 7.0 GB, 7.5 GB: about 40 GB, plus the conda and pip caches |
| one generation session (`runs/<name>/<time>/`) | 1.2 GB for the end-to-end test session (camera move, zoom-ins, HQ views, object insertion and two saved scenes: 669 MB of scenes, 389 MB of worker outputs) |

`download_checkpoints.sh --all` fetched 167.8 GB in about 25 min (80-100 MB/s). Re-running it with every file in
place took 3 s.

Start-up time is dominated by reading the weights. On NFS, a cold Step1X-Edit start took 753 s, against 231 s with
a warm page cache. Put `HF_HOME`, `WZ_CKPT_DIR` and the environments on a local NVMe disk if you can.

## Build times

- A clean build takes roughly 30-60 min per environment on 16 CPUs (`MAX_JOBS=8`).
- A fresh build of all four environments on 16 CPUs with `TORCH_CUDA_ARCH_LIST="8.9;9.0"` succeeded; `wz-gen3c`
  took the longest (transformer-engine and apex).
- On 2 CPUs, the script headers estimate several hours in total. PyTorch3D alone takes 45-90 min with `MAX_JOBS=2`;
  transformer-engine and apex take the longest.
- `TORCH_CUDA_ARCH_LIST` with a single architecture builds several times faster.
- Each nvcc job can use 4-6 GB of RAM.

## Measuring on your machine

```bash
python tests/smoke_workers.py --service coz --seed 123          # wz-main interpreter
python tests/smoke_workers.py --service step1x
python tests/smoke_workers.py --service gen3c --steps 18 --suspend-resume
```

Each run writes `runs/_smoke/<svc>-<time>/report.json` with the following fields:
- `init_s`: start-up time;
- `call_s` and `second_call_s`: request times;
- `worker_stats.peak_allocated_gb` and `peak_reserved_gb`: peak memory seen by torch;
- `nvidia_smi_peak_mib`: per-GPU `nvidia-smi` peak.

At run time, the server log prints the duration of every worker request (`[services] <svc>: ... in N s`).
The worker's `mem` request (`svc.mem(name)` in `services/`) returns the same peak-memory numbers as the smoke tests.
