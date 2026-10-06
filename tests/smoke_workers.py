#!/usr/bin/env python
"""Smoke tests of the WonderZoom model workers (verification step V4).

Each run starts one worker through services.ServiceManager, exactly as run.py does, sends one
request on fixtures generated from example_images/street.png, and prints a JSON report (also
written to <out>/report.json) with the start-up time, the request time and the peak GPU memory
(worker statistics plus `nvidia-smi -lms 500`).

    python tests/smoke_workers.py --service coz --seed 123
    python tests/smoke_workers.py --service step1x --offload true
    python tests/smoke_workers.py --service gen3c --steps 18 --suspend-resume --check-errors
    python tests/smoke_workers.py --service gen3c --bad-checkpoint      # must fail with a fatal reply
    python tests/smoke_workers.py --selftest                            # no GPU and no models needed

Options:
    --suspend-resume  suspend the worker (allocated memory must drop below 1.5 GB), resume it and
                      repeat the request with the same seed (the output must be identical).
    --check-errors    gen3c: 122 frames must be rejected with a clear error and 120 frames are
                      padded to 121 (one more generation); step1x: a prompt with quotes and a
                      newline must work.
    --set KEY=VALUE   any services.yaml override, e.g. --set services.gen3c.offload_network=true
    --selftest        tests the JSON-lines protocol, WorkerClient (timeouts, crashes, restarts,
                      fatal start-up), the GPU arbiter and ServiceManager with fake workers.

Run it with the wz-main interpreter (only the standard library, numpy, Pillow and omegaconf are
needed); the workers use the interpreters registered in config/services.local.yaml. The exit
status is 0 when every check passed.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

import yaml  # installed with omegaconf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_IMAGE = os.path.join(ROOT, "example_images", "street.png")
STEP1X_PROMPT = "a red ball is on the ground"
STEP1X_TRICKY_PROMPT = 'a "red" ball\nis on the ground, isn\'t it?'
SUSPENDED_LIMIT_GB = 1.5


def log(message):
    print(message, file=sys.stderr, flush=True)


def str2bool(value):
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


# ---------------------------------------------------------------------------------------------
# GPU memory monitor
# ---------------------------------------------------------------------------------------------

class NvidiaSmiMonitor:
    """Peak memory.used per GPU from `nvidia-smi --query-gpu=... -lms 500`."""

    def __init__(self, path=None):
        self.path = path
        self.peak_mib = {}
        self.proc = None
        self.thread = None

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=index,timestamp,memory.used", "--format=csv,noheader,nounits",
                 "-lms", "500"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.proc = None
            return self
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        out = open(self.path, "w") if self.path else None
        try:
            for line in self.proc.stdout:
                if out:
                    out.write(line)
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        used = float(parts[2])
                    except ValueError:
                        continue
                    self.peak_mib[parts[0]] = max(self.peak_mib.get(parts[0], 0.0), used)
        finally:
            if out:
                out.close()

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=5)
        return dict(self.peak_mib)


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------

def make_fixtures(service, out_dir, image_path):
    import numpy as np
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    source = Image.open(image_path)
    rgb = source.convert("RGB")
    width, height = rgb.size
    fx = {"size": [width, height]}

    if service == "gen3c":
        # A camera move revealing a hole on the right side that grows over the 121 frames.
        root = os.path.join(out_dir, "gen3c")
        frames_dir = os.path.join(root, "frames")
        masks_dir = os.path.join(root, "masks")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        cond = os.path.join(root, "input.png")
        source.save(cond)  # RGBA, like frames/saved_frames/input.png
        base = np.asarray(rgb).copy()
        for i in range(121):
            hole_w = int(round(width * 0.25 * i / 120))
            frame = base.copy()
            mask = np.zeros((height, width), np.uint8)
            if hole_w > 0:
                frame[:, width - hole_w:] = 0
                mask[:, width - hole_w:] = 255
            Image.fromarray(frame).save(os.path.join(frames_dir, f"test_{i:08d}.png"), compress_level=1)
            Image.fromarray(mask).save(os.path.join(masks_dir, f"mask_{i:08d}.png"), compress_level=1)
        # 120-frame (padded to 121) and 122-frame (rejected) variants, as symlinks to the same files.
        variants = {}
        for count in (120, 122):
            for kind, src_dir in (("frames", frames_dir), ("masks", masks_dir)):
                dst_dir = os.path.join(root, f"{kind}_{count}")
                os.makedirs(dst_dir, exist_ok=True)
                names = sorted(os.listdir(src_dir))
                links = [(name, name) for name in names[:count]]
                if count > len(names):  # repeat the last file under the next index
                    prefix = names[-1].rsplit("_", 1)[0]
                    links += [(f"{prefix}_{i:08d}.png", names[-1]) for i in range(len(names), count)]
                for dst_name, src_name in links:
                    dst = os.path.join(dst_dir, dst_name)
                    if not os.path.lexists(dst):
                        os.symlink(os.path.join(src_dir, src_name), dst)
                variants[f"{kind}_{count}_dir"] = dst_dir
        fx.update(condition_image=cond, frames_dir=frames_dir, masks_dir=masks_dir, **variants)
    elif service == "coz":
        # img_0 and a 1.5x centre zoom of it, as render_zoomin_rough_video3 produces.
        root = os.path.join(out_dir, "coz")
        os.makedirs(root, exist_ok=True)
        prev_png = os.path.join(root, "img_0.png")
        cur_png = os.path.join(root, "img_1.png")
        rgb.save(prev_png)
        crop_w, crop_h = int(width / 1.5), int(height / 1.5)
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        rgb.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.BILINEAR).save(cur_png)
        fx.update(prev_png=prev_png, cur_png=cur_png)
    elif service == "step1x":
        root = os.path.join(out_dir, "step1x")
        os.makedirs(root, exist_ok=True)
        image_png = os.path.join(root, "current_image.png")
        rgb.save(image_png)
        fx.update(image_png=image_png)
    return fx


# ---------------------------------------------------------------------------------------------
# Output checks
# ---------------------------------------------------------------------------------------------

def _load_rgb(path):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def check_output(service, fx, result):
    import numpy as np
    from PIL import Image
    checks = {}
    if service == "gen3c":
        frames = sorted(f for f in os.listdir(result["frames_dir"]) if f.startswith("frame_"))
        checks["n_frames_121"] = len(frames) == 121 and result.get("n_frames") == 121
        sizes = {Image.open(os.path.join(result["frames_dir"], f)).size for f in (frames[0], frames[-1])} if frames else set()
        checks["frame_size_1280x704"] = sizes == {(1280, 704)}
        checks["video_written"] = os.path.isfile(result["video_path"]) and os.path.getsize(result["video_path"]) > 0
    else:
        out = result
        image = Image.open(out)
        checks["output_size_matches_input"] = list(image.size) == fx["size"]
        arr = np.asarray(image.convert("RGB"))
        checks["output_not_constant"] = bool(arr.std() > 0)
        reference = fx["cur_png"] if service == "coz" else fx["image_png"]
        ref = _load_rgb(reference)
        checks["output_differs_from_input"] = ref.shape != arr.shape or bool(np.any(ref != arr))
    return checks


def outputs_identical(service, first, second):
    import numpy as np
    if service == "gen3c":
        names = sorted(f for f in os.listdir(first["frames_dir"]) if f.startswith("frame_"))
        max_diff = 0
        for name in names:
            a = _load_rgb(os.path.join(first["frames_dir"], name)).astype(np.int16)
            b = _load_rgb(os.path.join(second["frames_dir"], name)).astype(np.int16)
            max_diff = max(max_diff, int(np.abs(a - b).max()))
        return max_diff == 0, max_diff
    a = _load_rgb(first).astype(np.int16)
    b = _load_rgb(second).astype(np.int16)
    max_diff = int(np.abs(a - b).max())
    return max_diff == 0, max_diff


# ---------------------------------------------------------------------------------------------
# Real worker smoke test
# ---------------------------------------------------------------------------------------------

def call_service(svc, service, fx, args, out_dir, tag, prompt=None):
    if service == "gen3c":
        return svc.gen3c.generate(fx["condition_image"], fx["frames_dir"], fx["masks_dir"],
                                  args.prompt if args.prompt is not None else "", args.steps or 18,
                                  out_dir=os.path.join(out_dir, f"gen3c_{tag}"), seed=args.seed)
    if service == "coz":
        return svc.coz.super_resolve_dual(fx["prev_png"], fx["cur_png"], os.path.join(out_dir, f"coz_{tag}.png"),
                                          prompt=args.prompt, seed=args.seed)
    return svc.step1x.edit(fx["image_png"], prompt or args.prompt or STEP1X_PROMPT,
                           os.path.join(out_dir, f"step1x_{tag}.png"),
                           seed=42 if args.seed is None else args.seed, num_steps=args.steps or 28)


def run_service_test(args):
    from omegaconf import OmegaConf
    from services import SERVICE_NAMES, ServiceError, ServiceManager, load_services_config

    service = args.service
    overrides = {"policy": args.policy}
    for other in SERVICE_NAMES:
        if other != service:
            overrides[f"services.{other}.enabled"] = False
    if args.python:
        overrides[f"services.{service}.python"] = os.path.abspath(args.python)
    if args.offload is not None:
        if service != "step1x":
            raise SystemExit("--offload only applies to --service step1x")
        overrides["services.step1x.offload"] = args.offload
    if args.bad_checkpoint:
        key = {"gen3c": "checkpoint_dir", "coz": "lora_path", "step1x": "checkpoint_dir"}[service]
        overrides[f"services.{service}.{key}"] = "/nonexistent/wonderzoom_smoke_checkpoint"
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    cfg = load_services_config(args.config, args.local_config, overrides=overrides)

    out_dir = os.path.abspath(args.out or os.path.join(cfg.paths.runs_dir, "_smoke",
                                                       f"{service}-{time.strftime('%Y%m%d-%H%M%S')}"))
    os.makedirs(out_dir, exist_ok=True)
    report = {"service": service, "session_dir": out_dir, "ok": False, "checks": {}, "errors": []}
    if args.suspend_resume and args.seed is None:
        args.seed = 1234  # the identity check needs a fixed seed
        report["seed_note"] = "no --seed given: using 1234 for the suspend/resume identity check"
    report["seed"] = args.seed

    fx = make_fixtures(service, os.path.join(out_dir, "fixtures"), args.image) if not args.bad_checkpoint else {}
    svc = ServiceManager(cfg, out_dir, log=log)
    report["config"] = {k: v for k, v in OmegaConf.to_container(cfg.services[service]).items()
                        if k not in ("env",)}
    if not svc.enabled(service):
        report["errors"].append(f"{service} is not enabled: register its interpreter (scripts/register_env.py) "
                                f"or pass --python")
        return report

    monitor = NvidiaSmiMonitor(os.path.join(out_dir, "nvidia_smi.csv")) if not args.no_nvidia_smi else None
    if monitor:
        monitor.start()
    try:
        t0 = time.time()
        states = svc.start([service])
        report["init_s"] = round(time.time() - t0, 1)
        report["state_after_start"] = states.get(service)
        client = svc.client(service) if svc.enabled(service) else None
        report["ready_info"] = client.ready_info if client else None

        if args.bad_checkpoint:
            failed = states.get(service) == "failed"
            message = (client.last_error or "").lower()
            report["checks"]["fatal_at_init"] = failed and ("not found" in message or "missing" in message)
            report["fatal_error"] = client.last_error
            try:
                client.request("ping")
                report["checks"]["requests_rejected_after_fatal"] = False
            except ServiceError as e:
                report["checks"]["requests_rejected_after_fatal"] = "failed to start" in str(e)
            report["checks"]["no_restart_after_fatal"] = client.starts == 1
            report["ok"] = all(report["checks"].values())
            return report

        if states.get(service) != "ready":
            report["errors"].append(f"start-up failed: {client.last_error if client else 'unknown'}")
            return report

        if svc.arbiter.is_exclusive(service):
            report["note"] = "exclusive policy: the worker was parked after start-up and resumed for the call"
        t0 = time.time()
        first = call_service(svc, service, fx, args, out_dir, "first")
        report["call_s"] = round(time.time() - t0, 1)
        report["worker_stats"] = client.last_stats
        report["output"] = first
        report["checks"].update(check_output(service, fx, first))

        if args.check_errors:
            if service == "gen3c":
                try:
                    svc.gen3c.generate(fx["condition_image"], fx["frames_122_dir"], fx["masks_122_dir"], "",
                                       args.steps or 18, out_dir=os.path.join(out_dir, "gen3c_122"))
                    report["checks"]["rejects_122_frames"] = False
                except ServiceError as e:
                    report["checks"]["rejects_122_frames"] = "121" in str(e)
                    report["error_122_frames"] = str(e)
                report["checks"]["alive_after_error"] = client.alive()
                t1 = time.time()
                padded = svc.gen3c.generate(fx["condition_image"], fx["frames_120_dir"], fx["masks_120_dir"], "",
                                            args.steps or 18, out_dir=os.path.join(out_dir, "gen3c_120"))
                report["padded_120_s"] = round(time.time() - t1, 1)
                report["checks"]["pads_120_frames_to_121"] = padded.get("n_frames") == 121
            elif service == "step1x":
                t1 = time.time()
                tricky = call_service(svc, service, fx, args, out_dir, "tricky_prompt", prompt=STEP1X_TRICKY_PROMPT)
                report["tricky_prompt_s"] = round(time.time() - t1, 1)
                report["checks"]["quote_newline_prompt"] = os.path.isfile(tricky)

        if args.suspend_resume:
            client.suspend()
            suspended = svc.mem(service)
            report["mem_suspended"] = suspended
            allocated = suspended.get("allocated_gb")
            report["checks"]["suspended_below_1.5GB"] = allocated is not None and allocated < SUSPENDED_LIMIT_GB
            t1 = time.time()
            client.resume()
            report["resume_s"] = round(time.time() - t1, 1)
            report["mem_resumed"] = svc.mem(service)
            t1 = time.time()
            second = call_service(svc, service, fx, args, out_dir, "after_resume")
            report["second_call_s"] = round(time.time() - t1, 1)
            identical, max_diff = outputs_identical(service, first, second)
            report["checks"]["identical_after_resume"] = identical
            report["max_abs_diff_after_resume"] = max_diff

        report["mem_final"] = svc.mem(service)
        report["ok"] = bool(report["checks"]) and all(report["checks"].values())
        return report
    except ServiceError as e:
        report["errors"].append(str(e))
        if e.remote_traceback:
            report["remote_traceback"] = e.remote_traceback[-4000:]
        return report
    finally:
        svc.shutdown()
        if monitor:
            report["nvidia_smi_peak_mib"] = monitor.stop()


# ---------------------------------------------------------------------------------------------
# Self-test of the protocol, WorkerClient and GpuArbiter (no GPU, no models)
# ---------------------------------------------------------------------------------------------

FAKE_WORKER = textwrap.dedent('''
    import os, sys, time
    import _wz_protocol as wzp

    class Fake:
        def __init__(self, cfg):
            if cfg.get("fail"):
                raise RuntimeError("boom at load")
            time.sleep(float(cfg.get("load_sleep", 0)))
            print("library noise on stdout")           # must not reach the protocol pipe
            os.write(1, b"raw fd-1 noise\\n")
            self.cfg = cfg
            self.suspended = 0
        def info(self):
            return {"fake": True}
        def suspend(self):
            if self.cfg.get("suspend_fail"):
                raise RuntimeError("simulated suspend failure")
            self.suspended += 1
        def resume(self):
            if self.cfg.get("resume_fail"):
                raise RuntimeError("simulated resume failure")
        def op_echo(self, **kwargs):
            print("echo called", kwargs)
            return kwargs
        def op_sleep(self, seconds):
            time.sleep(seconds)
            return "slept"
        def op_fail(self):
            raise ValueError("bad request")
        def op_crash(self):
            os._exit(7)
        def op_input(self):
            return input()
        def op_suspend_count(self):
            return self.suspended

    def dry_import(cfg):
        return {"fake": True}

    if __name__ == "__main__":
        wzp.run_worker("fake", Fake, dry_import)
''')


FAKE_SERVICE_WORKER = textwrap.dedent('''
    import json, os, shutil, time
    import _wz_protocol as wzp

    NAME = os.path.basename(__file__).split("_worker")[0]

    class Fake:
        def __init__(self, cfg):
            self.cfg = cfg
        def info(self):
            return {"fake": True}
        def op_cfg(self):
            return self.cfg
        def op_generate(self, condition_image, frames_dir, masks_dir, prompt="", out_dir=None, num_steps=None, seed=None):
            time.sleep(float(self.cfg.get("op_sleep", 0)))
            with open(os.path.join(out_dir, "args.json"), "w") as f:
                json.dump({"prompt": prompt, "num_steps": num_steps, "seed": seed}, f)
            frames = os.path.join(out_dir, "frames")
            os.makedirs(frames, exist_ok=True)
            for i in range(3):
                shutil.copy(condition_image, os.path.join(frames, f"frame_{i:08d}.png"))
            video = os.path.join(out_dir, "gen3c_video.mp4")
            with open(video, "wb") as f:
                f.write(b"not a real video")
            return {"video_path": video, "frames_dir": frames, "n_frames": 3, "num_steps": num_steps}
        def op_super_resolve_dual(self, prev_png, cur_png, out_png, prompt=None, seed=None):
            time.sleep(float(self.cfg.get("op_sleep", 0)))
            shutil.copy(cur_png, out_png)
            return {"out_png": out_png, "seed": seed, "prompt": "fake prompt"}
        def op_edit(self, image_png, prompt, out_png, seed=42, num_steps=28, cfg_guidance=6.0, size_level=512):
            with open(out_png + ".prompt.json", "w") as f:
                json.dump(prompt, f)
            shutil.copy(image_png, out_png)
            return {"out_png": out_png}

    if __name__ == "__main__":
        wzp.run_worker(NAME, Fake, lambda cfg: {})
''')


def manager_selftest(tmp, check):
    """ServiceManager + service clients + arbiter with fake gen3c/coz/step1x workers (exclusive GPU 0)."""
    import services.manager as manager_module
    from PIL import Image
    from services import ServiceError, ServiceManager, load_services_config
    from services.base import WORKERS_DIR

    workers = os.path.join(tmp, "workers")
    repo = os.path.join(tmp, "repo")
    os.makedirs(workers)
    os.makedirs(repo)
    shutil.copy(os.path.join(WORKERS_DIR, "_wz_protocol.py"), workers)
    for name in ("gen3c", "coz", "step1x"):
        with open(os.path.join(workers, f"{name}_worker.py"), "w") as f:
            f.write(FAKE_SERVICE_WORKER)
    image = os.path.join(tmp, "image.png")
    Image.new("RGB", (64, 48), (120, 30, 200)).save(image)
    frames = os.path.join(tmp, "frames")
    os.makedirs(frames)

    overrides = {"policy": "exclusive", "main_gpu": 0, "services.coz.op_sleep": 1.5}
    for name in ("gen3c", "coz", "step1x"):
        overrides[f"services.{name}.python"] = sys.executable
        overrides[f"services.{name}.repo_dir"] = repo
        overrides[f"services.{name}.device"] = 0
        overrides[f"services.{name}.init_timeout_s"] = 60
        overrides[f"services.{name}.request_timeout_s"] = 60
    cfg = load_services_config(overrides=overrides)

    events = []
    gate_at_park = []  # render_allowed when the main models were parked (must be False)
    original_dir = manager_module.WORKERS_DIR
    manager_module.WORKERS_DIR = workers
    svc = svc2 = None

    def park_main():
        gate_at_park.append(svc.render_allowed.is_set())
        events.append("park_main")

    try:
        svc = ServiceManager(cfg, os.path.join(tmp, "session"), log=log)
        svc.register_main_tenant(park_main, lambda: events.append("unpark_main"), device=0)
        states = svc.start()
        clients = {n: svc.client(n) for n in ("gen3c", "coz", "step1x")}
        check("mgr_start_all_ready", states == {"gen3c": "ready", "coz": "ready", "step1x": "ready"}, str(states))
        check("mgr_suspended_after_start", all(c.suspended for c in clients.values()))
        check("mgr_main_parked_for_startup", events == ["park_main"], str(events))
        check("mgr_render_paused_before_park", gate_at_park == [False], str(gate_at_park))
        check("mgr_wait_ready", svc.wait_ready("gen3c", timeout=1))

        flags = {n: clients[n].request("cfg") for n in clients}
        expected_offload = manager_module.auto_step1x_offload(
            svc.arbiter.device_memory_gb("step1x"), 42, True, svc.arbiter.reserve_gb)
        check("mgr_auto_flags_resolved", flags["gen3c"]["park_text_encoder"] is True
              and flags["step1x"]["offload"] is expected_offload
              and "python" not in flags["gen3c"] and flags["coz"]["service"] == "coz",
              f"step1x offload {flags['step1x']['offload']} (GPU {svc.arbiter.device_memory_gb('step1x')} GiB)")
        for c in clients.values():  # the 'cfg' requests above auto-resumed the parked workers
            c.suspend()

        # The paper-era drivers drew random.randint(1000, 9999) per Gen3C / CoZ / Step1X call
        # (CoZ: before its seed); the CoZ seed below must match that stream.
        rng = random.Random(1)
        rng.randint(1000, 9999)  # Gen3C marker
        rng.randint(1000, 9999)  # CoZ marker
        expected_coz_seed = rng.randint(1, 999999)
        random.seed(1)
        result = svc.gen3c.generate(image, frames, frames, None, 18)
        check("mgr_gen3c_generate", result["n_frames"] == 3 and os.path.isfile(result["video_path"])
              and set(result) == {"video_path", "frames_dir", "n_frames"}
              and result["frames_dir"].startswith(os.path.join(tmp, "session", "services", "gen3c")))
        with open(os.path.join(os.path.dirname(result["frames_dir"]), "args.json")) as f:
            sent = json.load(f)
        check("mgr_gen3c_none_prompt_is_literal_None", sent["prompt"] == "None" and sent["num_steps"] == 18, str(sent))
        check("mgr_gen3c_active", not clients["gen3c"].suspended and clients["coz"].suspended)

        paused = []
        out_png = os.path.join(tmp, "coz_out.png")
        thread = threading.Thread(target=lambda: svc.coz.super_resolve_dual(image, image, out_png))
        thread.start()
        time.sleep(0.8)
        paused.append(svc.render_paused())
        thread.join(timeout=30)
        check("mgr_render_paused_during_service", paused == [True] and not svc.render_paused())
        check("mgr_coz_output", os.path.isfile(out_png) and clients["gen3c"].suspended and not clients["coz"].suspended)
        check("mgr_coz_random_seed", isinstance(svc.coz.last_seed, int) and 1 <= svc.coz.last_seed <= 999999)
        check("mgr_coz_seed_matches_paper_stream", svc.coz.last_seed == expected_coz_seed,
              f"{svc.coz.last_seed} vs {expected_coz_seed}")

        prompt = 'a "red" ball\nis on the ground, isn\'t it? \\o/'
        edited = svc.step1x.edit(image, prompt, os.path.join(tmp, "edit.png"))
        with open(edited + ".prompt.json") as f:
            check("mgr_step1x_prompt_json_safe", json.load(f) == prompt)

        events.clear()
        with svc.lease("main_models"):
            check("mgr_main_lease", events == ["unpark_main"] and clients["step1x"].suspended)
        check("mgr_lease_unknown_name", _raises(ServiceError, lambda: svc.lease("nope")))

        stats = svc.mem("coz")
        check("mgr_mem", isinstance(stats, dict) and "suspended" in stats)
        sleeper = threading.Thread(target=lambda: clients["coz"].request("super_resolve_dual", {
            "prev_png": image, "cur_png": image, "out_png": os.path.join(tmp, "coz_busy.png")}))
        sleeper.start()
        time.sleep(0.3)
        t0 = time.time()
        stats = svc.mem("coz")
        check("mgr_mem_does_not_wait_for_request", "busy" in stats and time.time() - t0 < 0.5, str(stats))
        sleeper.join(timeout=30)
        status = svc.status()
        check("mgr_status", status["services"]["coz"]["state"] == "ready" and status["gpu"]["devices"]["0"]["policy"] == "exclusive")

        # A killed worker is restarted by the next call.
        os.killpg(clients["coz"].pid, 9)
        time.sleep(0.5)
        out2 = svc.coz.super_resolve_dual(image, image, os.path.join(tmp, "coz_out2.png"), seed=7)
        check("mgr_restart_after_kill", os.path.isfile(out2) and clients["coz"].starts == 2 and svc.coz.last_seed == 7)

        pids = [c.pid for c in clients.values()]
        svc.shutdown()
        time.sleep(0.5)
        check("mgr_shutdown", not any(_pid_alive(p) for p in pids) and all(c.state == "closed" for c in clients.values()))
        check("mgr_calls_after_shutdown_fail", _raises(ServiceError, lambda: svc.gen3c.generate(image, frames, frames, "", 1)))

        # start_async() + wait_ready() while holding lease('main_models') must not deadlock: the
        # start-up thread needs that GPU, so wait_ready starts the worker in the calling thread.
        svc2 = ServiceManager(cfg, os.path.join(tmp, "session2"), log=log)
        svc2.register_main_tenant(lambda: True, lambda: True, device=0)
        with svc2.lease("main_models"):
            starter = svc2.start_async()
            t0 = time.time()
            ok = svc2.wait_ready("step1x", timeout=30)
            waited = time.time() - t0
        check("mgr_wait_ready_inside_main_lease", ok and waited < 25, f"{ok} after {waited:.1f} s")
        starter.join(timeout=60)
        states2 = {n: svc2.client(n).state for n in ("gen3c", "coz", "step1x")}
        check("mgr_inline_start_not_repeated", states2 == {"gen3c": "ready", "coz": "ready", "step1x": "ready"}
              and svc2.client("step1x").starts == 1, str(states2))
        svc2.shutdown()
    finally:
        manager_module.WORKERS_DIR = original_dir
        for manager in (svc, svc2):
            if manager is not None:
                manager.shutdown()


def _raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return True
    return False


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Zombies count as dead.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def selftest():
    from services.base import WORKERS_DIR, ServiceError, WorkerClient, build_worker_env
    from services.gpu_arbiter import GpuArbiter

    results = {}

    def check(name, condition, detail=""):
        results[name] = bool(condition)
        log(f"[selftest] {'PASS' if condition else 'FAIL'} {name} {detail}")

    tmp = tempfile.mkdtemp(prefix="wz_selftest_")
    script = os.path.join(tmp, "fake_worker.py")
    with open(script, "w") as f:
        f.write(FAKE_WORKER)
    python = sys.executable
    env = build_worker_env(python, WORKERS_DIR, "0")
    log_path = os.path.join(tmp, "fake.log")

    def client(**cfg):
        timeouts = {"init_timeout": cfg.pop("init_timeout", 60), "request_timeout": cfg.pop("request_timeout", 60)}
        return WorkerClient("fake", python, script, tmp, worker_config=cfg, env=env, log_path=log_path,
                            log=log, **timeouts)

    # Dry import.
    p = subprocess.run([python, script, "--dry-import"], cwd=tmp, env=env, capture_output=True, text=True, timeout=60)
    msg = json.loads(p.stdout.strip().splitlines()[-1])
    check("dry_import", p.returncode == 0 and msg.get("event") == "dry-import" and msg["info"].get("fake"))

    c = client()
    info = c.start()
    check("ready_handshake", c.state == "ready" and info.get("fake") and "load_seconds" in info)
    payload = {"a": 1, "text": 'quote " and \\ backslash\nnewline', "nested": [1, 2.5, None]}
    check("echo_roundtrip", c.request("echo", payload) == payload)
    check("stdout_noise_redirected", "library noise on stdout" in open(log_path).read())
    try:
        c.request("input")
        check("input_gets_eof", False)
    except ServiceError as e:
        check("input_gets_eof", "EOFError" in str(e), str(e))
    try:
        c.request("fail")
        check("error_reply", False)
    except ServiceError as e:
        check("error_reply", "bad request" in str(e) and "ValueError" in (e.remote_traceback or ""))
    try:
        c.request("no_such_op")
        check("unknown_op", False)
    except ServiceError as e:
        check("unknown_op", "unknown op" in str(e))
    check("suspend", c.suspend() and c.suspended and c.request("ping")["suspended"])
    check("suspend_idempotent", c.suspend() is False)
    # A service op on a suspended worker resumes it first (safety net for stale bookkeeping).
    check("auto_resume_on_op", c.request("suspend_count") == 1 and not c.suspended)
    check("resume", c.suspend() and c.resume() and not c.suspended and c.resume() is False)
    check("stats_in_reply", isinstance(c.last_stats, dict) and "seconds" in c.last_stats)

    pid = c.pid
    try:
        c.request("sleep", {"seconds": 30}, timeout=2)
        check("request_timeout", False)
    except ServiceError as e:
        time.sleep(0.5)
        check("request_timeout", "timed out" in str(e) and c.state == "dead" and not _pid_alive(pid), str(e))
    check("lazy_restart_after_timeout", c.request("echo", {"x": 2}) == {"x": 2} and c.starts == 2 and c.pid != pid)

    pid = c.pid
    try:
        c.request("crash")
        check("crash_detected", False)
    except ServiceError as e:
        check("crash_detected", "exit code 7" in str(e) and c.state == "dead", str(e).splitlines()[0])
    check("lazy_restart_after_crash", c.request("echo", {"y": 3}) == {"y": 3} and c.starts == 3)
    c.stop(close=True)
    check("stop", c.state == "closed" and not c.alive())

    bad = client(fail=True)
    try:
        bad.start()
        check("fatal_at_load", False)
    except ServiceError as e:
        check("fatal_at_load", "boom at load" in str(e) and bad.state == "failed", str(e))
    try:
        bad.request("echo")
        check("no_restart_after_fatal", False)
    except ServiceError as e:
        check("no_restart_after_fatal", "failed to start" in str(e) and bad.starts == 1)

    # A failed suspend kills the worker (the arbiter then counts it as holding no GPU memory).
    sf = client(suspend_fail=True)
    sf.start()
    pid = sf.pid
    try:
        sf.suspend()
        check("suspend_failure_kills_worker", False)
    except ServiceError as e:
        time.sleep(0.5)
        check("suspend_failure_kills_worker", "simulated suspend failure" in str(e) and sf.state == "dead"
              and not sf.alive() and not _pid_alive(pid), str(e).splitlines()[0])
    check("restart_after_failed_suspend", sf.request("echo", {"w": 1}) == {"w": 1} and sf.starts == 2)
    sf.stop(close=True)

    # A failed automatic resume (service op on a suspended worker) is an error reply, not a dead pipe.
    rf = client(resume_fail=True)
    rf.start()
    rf.suspend()
    try:
        rf.request("echo", {"v": 1})
        check("auto_resume_failure_is_error_reply", False)
    except ServiceError as e:
        check("auto_resume_failure_is_error_reply", "simulated resume failure" in str(e) and rf.alive()
              and rf.state == "ready", str(e).splitlines()[0])
    try:
        rf.resume()
        check("resume_failure_kills_worker", False)
    except ServiceError:
        check("resume_failure_kills_worker", rf.state == "dead" and not rf.alive())
    rf.stop(close=True)

    # mem() never waits for a start-up.
    from types import SimpleNamespace
    from services.manager import ServiceManager
    starting = client(load_sleep=4)
    starter = threading.Thread(target=starting.start, daemon=True)
    starter.start()
    time.sleep(1.0)
    t0 = time.time()
    stats = ServiceManager.mem(SimpleNamespace(client=lambda name: starting), "fake")
    check("mem_does_not_wait_for_startup", stats.get("state") == "starting" and time.time() - t0 < 0.5, str(stats))
    starter.join(timeout=30)
    starting.stop(close=True)

    slow = client(load_sleep=30, init_timeout=2)
    t0 = time.time()
    try:
        slow.start()
        check("init_timeout", False)
    except ServiceError as e:
        check("init_timeout", "within 2 s" in str(e) and time.time() - t0 < 20 and not slow.alive(), str(e))

    # Parent-death watchdog: kill -9 an intermediate process that owns a worker.
    helper = textwrap.dedent(f'''
        import sys, time
        sys.path.insert(0, {ROOT!r})
        from services.base import WorkerClient, build_worker_env
        c = WorkerClient("fake", {python!r}, {script!r}, {tmp!r}, env=build_worker_env({python!r}, {WORKERS_DIR!r}, "0"),
                         log_path={log_path!r}, log=lambda m: None)
        c.start()
        print(c.pid, flush=True)
        time.sleep(600)
    ''')
    h = subprocess.Popen([python, "-c", helper], stdout=subprocess.PIPE, text=True)
    worker_pid = int(h.stdout.readline())
    h.kill()
    h.wait()
    deadline = time.time() + 15
    while time.time() < deadline and _pid_alive(worker_pid):
        time.sleep(0.5)
    check("worker_exits_when_parent_dies", not _pid_alive(worker_pid))

    # GPU arbiter with fake tenants on one exclusive GPU.
    events = []
    arb = GpuArbiter(policy="auto", reserve_gb=4, log=lambda m: None)
    arb.add_tenant("main_models", "0", "main", 16)
    arb.add_tenant("gen3c", "0", "worker", 36)
    arb.add_tenant("coz", "0", "worker", 28)
    arb.add_tenant("step1x", "1", "worker", 42)
    arb.resolve({"0": 45.0, "1": 80.0})
    check("auto_policy", arb.device_policy("0") == "exclusive" and arb.device_policy("1") == "resident")
    for name in ("main_models", "gen3c", "coz"):
        arb.set_callbacks(name, park_fn=lambda n=name: events.append(("park", n)),
                          unpark_fn=lambda n=name: events.append(("unpark", n)), parked=name != "main_models")
    with arb.lease("gen3c"):
        check("lease_parks_others", ("park", "main_models") in events and ("unpark", "gen3c") in events)
        check("render_paused", not arb.render_allowed.is_set())
        events.clear()
        with arb.lease("gen3c"):
            check("reentrant_noop", events == [])
        with arb.lease("main_models"):
            check("nested_switch", events == [("park", "gen3c"), ("unpark", "main_models")])
            check("render_allowed_for_main", arb.render_allowed.is_set())
        check("nested_restore", events[-2:] == [("park", "main_models"), ("unpark", "gen3c")])
    check("render_resumed", arb.render_allowed.is_set())
    events.clear()
    with arb.lease("step1x"):
        check("resident_gpu_noop", events == [])

    # The render gate is closed before anything moves, and the frame in progress is waited for.
    gate = GpuArbiter(policy="exclusive", log=lambda m: None)
    gate.add_tenant("main_models", "0", "main", 16)
    gate.add_tenant("gen3c", "0", "worker", 36)
    gate.resolve({"0": 45.0})
    moves = []
    frame_done = []
    gate.set_callbacks("main_models", park_fn=lambda: moves.append(("park_main", gate.render_allowed.is_set(), time.time())),
                       unpark_fn=lambda: moves.append(("unpark_main", gate.render_allowed.is_set(), time.time())), parked=False)
    gate.set_callbacks("gen3c", park_fn=lambda: moves.append(("park_gen3c", gate.render_allowed.is_set(), time.time())),
                       unpark_fn=lambda: moves.append(("unpark_gen3c", gate.render_allowed.is_set(), time.time())), parked=True)
    in_frame = threading.Event()

    def draw_one_frame():
        with gate.render_frame() as allowed:
            in_frame.set()
            time.sleep(0.6)
            frame_done.append((allowed, time.time()))

    renderer = threading.Thread(target=draw_one_frame)
    renderer.start()
    in_frame.wait(5)
    with gate.lease("gen3c"):
        with gate.render_frame() as allowed_inside:
            pass
        with gate.lease("main_models"):
            pass
    renderer.join(timeout=10)
    first_move = moves[0] if moves else ("none", True, 0.0)
    check("render_gate_closed_before_moves", all(not m[1] for m in moves if m[0] in ("park_main", "unpark_gen3c")),
          str([(m[0], m[1]) for m in moves]))
    check("render_waits_for_frame_in_progress", frame_done and frame_done[0][0] and first_move[2] >= frame_done[0][1],
          f"first move {first_move[0]} at +{first_move[2] - (frame_done[0][1] if frame_done else 0):.2f} s")
    check("render_frame_reports_pause", allowed_inside is False and gate.render_allowed.is_set())

    from services.manager import auto_step1x_offload
    check("step1x_auto_offload_rule",
          auto_step1x_offload(44.99, 42, exclusive=False, reserve_gb=4) is False      # L40S, step1x alone
          and auto_step1x_offload(44.99, 42, exclusive=True, reserve_gb=4) is True    # L40S shared, exclusive
          and auto_step1x_offload(47.99, 42, exclusive=True, reserve_gb=4) is False   # 48 GB, exclusive
          and auto_step1x_offload(79.6, 42, exclusive=False, reserve_gb=4, co_resident_gb=64) is True
          and auto_step1x_offload(79.6, 42, exclusive=False, reserve_gb=4) is False
          and auto_step1x_offload(39.6, 42, exclusive=False, reserve_gb=4) is True
          and auto_step1x_offload(None, 42, exclusive=False, reserve_gb=4) is True)

    order = []
    arb2_entered = threading.Event()

    def other_thread():
        with arb.lease("coz"):
            order.append("coz")
            arb2_entered.set()

    with arb.lease("main_models"):
        thread = threading.Thread(target=other_thread)
        thread.start()
        time.sleep(0.5)
        check("lease_blocks_other_thread", not arb2_entered.is_set())
        check("holds_lease", arb.holds_lease("coz") and arb.holds_lease("main_models") and not arb.holds_lease("step1x"))
        order.append("main")
    thread.join(timeout=10)
    check("lease_handover", order == ["main", "coz"])

    manager_selftest(tmp, check)

    ok = all(results.values())
    if ok:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"selftest": True, "ok": ok, "checks": results, "tmp_dir": None if ok else tmp}


# ---------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--service", choices=["gen3c", "coz", "step1x"])
    parser.add_argument("--selftest", action="store_true", help="test the protocol with a fake worker (no GPU)")
    parser.add_argument("--suspend-resume", action="store_true")
    parser.add_argument("--check-errors", action="store_true")
    parser.add_argument("--bad-checkpoint", action="store_true", help="point the weights to a missing path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="gen3c: 18 by default; step1x: 28 by default")
    parser.add_argument("--offload", type=str2bool, default=None, help="step1x only: true|false")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--policy", choices=["auto", "resident", "exclusive"], default=None)
    parser.add_argument("--python", default=None, help="interpreter of the service environment")
    parser.add_argument("--config", default="config/services.yaml")
    parser.add_argument("--local-config", default="config/services.local.yaml")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--out", default=None, help="output directory (default runs/_smoke/<service>-<time>)")
    parser.add_argument("--no-nvidia-smi", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra config override, e.g. --set services.gen3c.repo_dir=/path (repeatable)")
    args = parser.parse_args()

    if args.selftest:
        report = selftest()
    elif args.service:
        report = run_service_test(args)
        path = os.path.join(report["session_dir"], "report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log(f"report written to {path}")
    else:
        parser.error("--service or --selftest is required")
    print(json.dumps(report, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
