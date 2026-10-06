#!/usr/bin/env python
"""Headless end-to-end driver for the WonderZoom generation server (run.py).

It connects with a python-socketio client (run it with the wz-main interpreter) and replays what the
browser UI (splat-main/index_gen.html + main_stream.js) sends: a background thread emits 'render-pose'
at about 10 Hz, and every scenario step sends the same events as the corresponding key.

    python tests/e2e_headless.py --scenario boot,crack_fix,delete,undo,save
    python tests/e2e_headless.py --url http://127.0.0.1:7747 --scenario move,zoom,undo,zoom,hq_nvs,save \
        --coz_seed 123 --timeout 7200 --out runs/_e2e/street
    python tests/e2e_headless.py --url http://127.0.0.1:7748 --scenario frames      # run.py --view

Scenario steps (comma separated, run in order; 'boot' is implied unless the scenario starts with
'frames'):
    boot         wait until 'server-status' is idle after the initial scene build
    move         yaw +0.3 rad and 0.1 forward at the base focal length, then R ('gen', addToTrajectory)
    zoom         H at the base focal length, then R at fx = base * 1.05^14 (V pressed 14 times)
    object_zoom  'scene-prompt' --object, then zoom (insertion needs features.objects)
    undo / save / delete (C, current view) / crack_fix (Ctrl+Alt+Space) / hq_nvs (Ctrl+Shift+Space)
    orbit        Space (orbit preview, no scene change)
    reject_move  R at the base focal length must be refused (e.g. --no_services); the server stays alive
    reject_zoom  H + R at a zoomed focal length must be refused
    frames       wait for --min_frames 'frame' events (also works against run.py --view)

Camera math (invert4 / translate4 / rotate4 and the view matrix) is ported from main_stream.js.
The driver saves the received 'frame' JPEGs (throttled, plus one per step), the 'rough-video' /
'out-video' / 'concat-video' payloads, and <out>/report.json. It exits with 0 when every step passed,
1 when a step failed and 2 on a usage error.
"""
import argparse
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime

try:
    import socketio
except ImportError:  # pragma: no cover
    sys.exit("python-socketio is required: run this with the wz-main interpreter")


# ------------------------------------------------------------------------------------------------
# Camera math, ported 1:1 from splat-main/main_stream.js (column-major 4x4 matrices as flat lists)
# ------------------------------------------------------------------------------------------------
DEFAULT_VIEW_MATRIX = [-1, 0, 0, 0,
                       0, -1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1]
DEFAULT_FOCAL_LENGTH = 1024
ZOOM_STEP = 1.05            # one V key press
MAX_FOCAL = 9999999


def invert4(a):
    b00 = a[0] * a[5] - a[1] * a[4]
    b01 = a[0] * a[6] - a[2] * a[4]
    b02 = a[0] * a[7] - a[3] * a[4]
    b03 = a[1] * a[6] - a[2] * a[5]
    b04 = a[1] * a[7] - a[3] * a[5]
    b05 = a[2] * a[7] - a[3] * a[6]
    b06 = a[8] * a[13] - a[9] * a[12]
    b07 = a[8] * a[14] - a[10] * a[12]
    b08 = a[8] * a[15] - a[11] * a[12]
    b09 = a[9] * a[14] - a[10] * a[13]
    b10 = a[9] * a[15] - a[11] * a[13]
    b11 = a[10] * a[15] - a[11] * a[14]
    det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
    if not det:
        return None
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ]


def rotate4(a, rad, x, y, z):
    length = math.hypot(x, y, z)
    x /= length
    y /= length
    z /= length
    s = math.sin(rad)
    c = math.cos(rad)
    t = 1 - c
    b00 = x * x * t + c
    b01 = y * x * t + z * s
    b02 = z * x * t - y * s
    b10 = x * y * t - z * s
    b11 = y * y * t + c
    b12 = z * y * t + x * s
    b20 = x * z * t + y * s
    b21 = y * z * t - x * s
    b22 = z * z * t + c
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
    ] + list(a[12:16])


def translate4(a, x, y, z):
    return list(a[0:12]) + [
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ]


class Camera:
    """The browser camera state (yaw, pitch, movement, fx, fy) of main_stream.js."""

    def __init__(self, focal=DEFAULT_FOCAL_LENGTH):
        self.lock = threading.Lock()
        self.yaw = 0.0
        self.pitch = 0.0
        self.movement = [0.0, 0.0, 0.0]
        self.fx = float(focal)
        self.fy = float(focal)

    def view_matrix(self):
        """currentViewMatrix() of main_stream.js."""
        with self.lock:
            inv = invert4(DEFAULT_VIEW_MATRIX)
            inv = translate4(inv, *self.movement)
            inv = rotate4(inv, self.yaw, 0, 1, 0)    # yaw around the Y axis
            inv = rotate4(inv, self.pitch, 1, 0, 0)  # pitch around the X axis
            return invert4(inv)

    def turn(self, dyaw=0.0, dpitch=0.0):
        with self.lock:
            self.yaw += dyaw
            self.pitch = max(-math.pi / 2, min(math.pi / 2, self.pitch + dpitch))

    def move(self, dz=0.0, dx=0.0, dy=0.0):
        """Arrow keys (dz forward, dx right) and N/M (dy), converted with the current yaw as in JS."""
        with self.lock:
            forward = [math.sin(self.yaw) * dz, 0.0, math.cos(self.yaw) * dz]
            right = [math.sin(self.yaw + math.pi / 2) * dx, 0.0, math.cos(self.yaw + math.pi / 2) * dx]
            self.movement[0] += forward[0] + right[0]
            self.movement[1] += forward[1] + right[1] + dy
            self.movement[2] += forward[2] + right[2]

    def set_focal(self, fx, fy=None):
        with self.lock:
            self.fx = float(fx)
            self.fy = float(fx if fy is None else fy)

    def zoom_in(self, presses):
        """V pressed `presses` times: fx = min(fx * 1.05, 9999999), multiplied step by step as in JS."""
        with self.lock:
            for _ in range(presses):
                self.fx = min(self.fx * ZOOM_STEP, MAX_FOCAL)
                self.fy = min(self.fy * ZOOM_STEP, MAX_FOCAL)

    def pose(self):
        """The 'render-pose' / 'add-trajectory-point' payload."""
        vm = self.view_matrix()
        with self.lock:
            return {"viewMatrix": vm, "fx": self.fx, "fy": self.fy}

    def state(self):
        with self.lock:
            return {"yaw": self.yaw, "pitch": self.pitch, "movement": list(self.movement),
                    "fx": self.fx, "fy": self.fy}


# ------------------------------------------------------------------------------------------------
# Socket.IO client
# ------------------------------------------------------------------------------------------------
class StepFailed(Exception):
    pass


class Driver:
    SCENE_JOBS = ("move", "zoom", "hq_nvs", "crack_fix", "delete", "complete_background")

    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out)
        self.frames_dir = os.path.join(self.out, "frames")
        self.videos_dir = os.path.join(self.out, "videos")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        self.t0 = time.time()
        self.deadline = self.t0 + args.timeout
        self.camera = Camera()
        self.base_focal = None

        self.cond = threading.Condition()
        self.events = []              # (t, event, summary) of every received event, in order
        self.server_config = None
        self.last_status = None
        self.num_points = None
        self.labels = None
        self.frame_count = 0
        self.last_frame = None
        self.last_frame_saved_at = 0.0
        self.saved_frames = []
        self.video_count = 0
        self.connected = False
        self.connect_count = 0
        self.stop = threading.Event()
        self.points_before = []       # num_points before each scene-changing job (for undo)
        self.steps = []
        self.saved_pth = []

        self.sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=1,
                                   reconnection_delay_max=5, logger=False, engineio_logger=False)
        self._register_handlers()

    # ---- bookkeeping ----

    def log(self, message):
        print(f"[e2e {time.time() - self.t0:8.1f}s] {message}", flush=True)

    def _record(self, event, summary=None):
        with self.cond:
            self.events.append((time.time() - self.t0, event, summary))
            self.cond.notify_all()

    def cursor(self):
        with self.cond:
            return len(self.events)

    def remaining(self, cap=None):
        left = self.deadline - time.time()
        if cap is not None:
            left = min(left, cap)
        return max(0.0, left)

    def wait_for(self, predicate, timeout, what):
        """Wait until predicate(events) returns a non-None value; StepFailed on timeout."""
        end = time.time() + timeout
        with self.cond:
            while True:
                value = predicate(self.events)
                if value is not None:
                    return value
                left = end - time.time()
                if left <= 0:
                    raise StepFailed(f"timed out after {timeout:.0f} s waiting for {what}")
                self.cond.wait(min(left, 1.0))

    # ---- handlers ----

    def _register_handlers(self):
        sio = self.sio

        @sio.event
        def connect():
            self.connected = True
            self.connect_count += 1
            self.log(f"connected (#{self.connect_count}, sid {sio.get_sid()})")
            self._record("connect")

        @sio.event
        def disconnect(*_):
            self.connected = False
            self.log("disconnected")
            self._record("disconnect")

        @sio.on("server-config")
        def on_server_config(data=None):
            self.server_config = data
            focal = (data or {}).get("init_focal_length")
            if focal and self.base_focal is None:
                self.base_focal = float(focal)
                self.camera.set_focal(self.base_focal)
            self._record("server-config", data)

        @sio.on("server-status")
        def on_server_status(data=None):
            data = dict(data or {})
            self.last_status = data
            self.log(f"status: {data.get('state')} [{data.get('job')}] {data.get('message', '')}")
            self._record("server-status", data)

        @sio.on("server-state")
        def on_server_state(message=""):
            if not str(message).startswith(("Collecting frames", "Orbit preview:")):
                self.log(f"server-state: {message}")
            self._record("server-state", str(message))

        @sio.on("scene-stats")
        def on_scene_stats(data=None):
            data = dict(data or {})
            if data.get("num_points") is not None:
                self.num_points = int(data["num_points"])
            if data.get("labels") is not None:
                self.labels = list(data["labels"])
            self.log(f"scene-stats: {data}")
            self._record("scene-stats", data)

        @sio.on("scene-prompt")
        def on_scene_prompt(data=None):  # None is sent without arguments
            self._record("scene-prompt", data)

        @sio.on("frame")
        def on_frame(data=b""):
            self.frame_count += 1
            self.last_frame = bytes(data)
            now = time.time()
            if self.frame_count == 1 or now - self.last_frame_saved_at >= self.args.frame_interval:
                self.last_frame_saved_at = now
                self._save_frame(f"frame_{self.frame_count:06d}.jpg")
            with self.cond:
                self.cond.notify_all()

        for event in ("rough-video", "out-video", "concat-video"):
            sio.on(event, self._video_handler(event))

    def _video_handler(self, event):
        def handler(data=b""):
            self.video_count += 1
            path = os.path.join(self.videos_dir, f"{self.video_count:03d}_{event}.mp4")
            with open(path, "wb") as f:
                f.write(bytes(data))
            self._record(event, {"path": path, "bytes": len(data)})
        return handler

    def _save_frame(self, name):
        if self.last_frame is None:
            return None
        path = os.path.join(self.frames_dir, name)
        with open(path, "wb") as f:
            f.write(self.last_frame)
        self.saved_frames.append(path)
        return path

    # ---- pose streaming ----

    def _pose_loop(self):
        period = 1.0 / max(self.args.pose_hz, 0.1)
        while not self.stop.is_set():
            if self.connected:
                try:
                    self.sio.emit("render-pose", self.camera.pose())
                except Exception:  # disconnected between the check and the emit
                    pass
            self.stop.wait(period)

    def emit(self, event, *data):
        if not self.connected:
            self.wait_for(lambda ev: True if self.connected else None, self.remaining(120), "a connection")
        self.log(f"emit {event}" + (f" {json.dumps(data[0])[:200]}" if data and event != "render-pose" else ""))
        self.sio.emit(event, *data)

    # ---- connection ----

    def connect(self):
        end = time.time() + self.remaining(self.args.connect_timeout)
        last = None
        while time.time() < end:
            try:
                self.sio.connect(self.args.url, transports=["websocket", "polling"], wait_timeout=10)
                break
            except Exception as e:  # server not up yet
                last = e
                time.sleep(2)
        else:
            raise StepFailed(f"could not connect to {self.args.url}: {last}")
        threading.Thread(target=self._pose_loop, daemon=True).start()

    # ---- job helpers ----

    @staticmethod
    def _rejection(summary):
        text = str(summary)
        return " ignored: " in text or text.startswith("object insertion unavailable")

    def run_job(self, job, send, accept_timeout=60.0, expect_reject=False):
        """Send a request and wait for its job to finish.

        Returns the final 'server-status' payload. The job is accepted when a busy status for `job`
        arrives; a 'server-state' '<action> ignored: ...' reply means it was refused."""
        start = self.cursor()
        send()

        def accepted(events):
            for _, event, summary in events[start:]:
                if event == "server-state" and self._rejection(summary):
                    return ("rejected", summary)
                if event == "server-status" and summary.get("job") == job and summary.get("state") in ("busy", "error"):
                    return ("accepted", summary)
            return None

        kind, info = self.wait_for(accepted, self.remaining(accept_timeout), f"'{job}' to be accepted")
        if expect_reject:
            if kind != "rejected":
                raise StepFailed(f"'{job}' was accepted but should have been refused")
            return {"rejected": info}
        if kind == "rejected":
            raise StepFailed(f"'{job}' refused by the server: {info}")

        def finished(events):
            seen_busy = False
            for _, event, summary in events[start:]:
                if event == "server-status":
                    if summary.get("state") == "busy" and summary.get("job") == job:
                        seen_busy = True
                    elif summary.get("state") == "error" and summary.get("job") in (job, None):
                        return summary
                    elif summary.get("state") == "idle" and seen_busy:
                        return summary
                if event == "disconnect" and not self.args.allow_reconnect:
                    return {"state": "error", "job": job, "message": "disconnected from the server"}
            return None

        final = self.wait_for(finished, self.remaining(), f"'{job}' to finish")
        if final.get("state") != "idle":
            raise StepFailed(f"'{job}' failed: {final.get('message')}")
        return final

    def scene_stats_after(self, start, job):
        """The 'scene-stats' of `job` received after event index `start` (None if none)."""
        with self.cond:
            for _, event, summary in self.events[start:]:
                if event == "scene-stats" and summary.get("last_job") == job:
                    return summary
        return None

    def scene_job(self, job, send, expect_change=None, **kwargs):
        """A scene-changing job: remembers the point count before it (for undo) and checks the change."""
        before = self.num_points
        start = self.cursor()
        final = self.run_job(job, send, **kwargs)
        stats = self.scene_stats_after(start, job)
        after = stats.get("num_points") if stats else None
        self.points_before.append((job, before))
        result = {"num_points_before": before, "num_points_after": after, "message": final.get("message"),
                  "seconds": stats.get("seconds") if stats else None}
        if expect_change == "increase" and before is not None and after is not None and after <= before:
            raise StepFailed(f"{job}: num_points did not increase ({before} -> {after})")
        if expect_change == "decrease" and before is not None and after is not None and after >= before:
            raise StepFailed(f"{job}: num_points did not decrease ({before} -> {after})")
        if after is None:
            result["warning"] = "no scene-stats received for this job"
        return result

    def _gen3c_requests(self):
        """Per-request Gen3C output directories of the server session (<session>/services/gen3c/*)."""
        session = (self.last_status or {}).get("session_dir")
        if not self.args.session_check or not session:
            return None
        root = os.path.join(session, "services", "gen3c")
        return set(os.listdir(root)) if os.path.isdir(root) else set()

    def _check_gen3c_frames(self, before, result):
        """A Gen3C job must have written 121 frames into a new request directory."""
        after = self._gen3c_requests()
        if before is None or after is None:
            return
        root = os.path.join(self.last_status["session_dir"], "services", "gen3c")
        counts = {}
        for name in sorted(after - before):
            frames = os.path.join(root, name, "frames")
            counts[name] = len([f for f in os.listdir(frames) if f.endswith(".png")]) if os.path.isdir(frames) else 0
        result["gen3c_frames"] = counts
        if not counts:
            raise StepFailed("no new Gen3C request directory under " + root)
        if any(n != 121 for n in counts.values()):
            raise StepFailed(f"Gen3C frame counts {counts}, expected 121")

    def require_base_focal(self):
        if self.base_focal is None:
            self.wait_for(lambda ev: True if self.base_focal is not None else None, self.remaining(60),
                          "'server-config'")
        return self.base_focal

    def settle(self, seconds=0.4):
        """Let a few 'render-pose' updates reach the server (the browser streams them continuously)."""
        time.sleep(seconds)

    # ---- steps ----

    def step_boot(self):
        def ready(events):
            status = self.last_status
            if status is None:
                return None
            if status.get("state") in ("idle", "error"):
                return status
            return None

        status = self.wait_for(ready, self.remaining(), "the initial scene (server-status idle)")
        if status.get("state") == "error" and status.get("job") in ("startup", "initial_scene", None):
            raise StepFailed(f"server reported an error at start-up: {status.get('message')}")
        self.require_base_focal()
        result = {"status": status, "server_config": self.server_config, "num_points": self.num_points,
                  "seconds_since_start": round(time.time() - self.t0, 1)}
        # Frames must arrive once the scene exists. Under the exclusive GPU policy the preview pauses
        # while a worker loads on the main GPU, so a late first frame is only a warning here; the
        # report checks that frames arrived at all.
        n0 = self.frame_count
        try:
            self.wait_for(lambda ev: True if self.frame_count > n0 else None, self.remaining(120), "a 'frame'")
        except StepFailed as e:
            result["warning"] = str(e)
        return result

    def step_frames(self):
        n0 = self.frame_count
        need = self.args.min_frames
        self.wait_for(lambda ev: True if self.frame_count - n0 >= need else None,
                      self.remaining(self.args.frames_timeout), f"{need} 'frame' events")
        return {"frames": self.frame_count - n0}

    def step_move(self):
        base = self.require_base_focal()
        self.camera.set_focal(base)
        self.camera.turn(dyaw=self.args.move_yaw)
        self.camera.move(dz=self.args.move_forward)
        self.emit("clear-trajectory")
        self.settle()
        pose = self.camera.pose()
        gen3c_before = self._gen3c_requests()
        result = self.scene_job("move", lambda: self.emit("gen", dict(pose, addToTrajectory=True)),
                                expect_change="increase")
        result["camera"] = self.camera.state()
        self._check_gen3c_frames(gen3c_before, result)
        return result

    def _zoom(self, job="zoom", expect_reject=False):
        base = self.require_base_focal()
        self.emit("clear-trajectory")
        self.camera.set_focal(base)
        self.settle()
        start = self.cursor()
        self.emit("add-trajectory-point", self.camera.pose())
        self.wait_for(lambda ev: next((s for _, e, s in ev[start:] if e == "server-state"
                                       and (str(s).startswith("Trajectory point") or self._rejection(s))), None),
                      self.remaining(60), "the H trajectory point")
        self.camera.zoom_in(self.args.zoom_presses)
        self.settle()
        payload = dict(self.camera.pose(), addToTrajectory=True)
        if self.args.coz_seed is not None:
            payload["cozSeed"] = int(self.args.coz_seed)
        if expect_reject:
            return self.run_job("zoom", lambda: self.emit("gen", payload), expect_reject=True)
        result = self.scene_job("zoom", lambda: self.emit("gen", payload), expect_change="increase")
        result["camera"] = self.camera.state()
        if self.args.session_check and self.last_status and self.last_status.get("session_dir"):
            cache = os.path.join(self.last_status["session_dir"], "cache")
            result["coz_outputs"] = {name: _png_size(os.path.join(cache, name))
                                     for name in ("coz_output.png", "coz_output2.png")}
            gen_w, gen_h = (self.server_config or {}).get("gen_W"), (self.server_config or {}).get("gen_H")
            for name, size in result["coz_outputs"].items():
                if size is None:
                    raise StepFailed(f"zoom: {cache}/{name} missing")
                if gen_w and gen_h and tuple(size) != (gen_w, gen_h):
                    raise StepFailed(f"zoom: {name} is {size[0]}x{size[1]}, expected {gen_w}x{gen_h}")
        return result

    def step_zoom(self):
        return self._zoom()

    def step_object_zoom(self):
        obj = self.args.object
        if self.args.object_pitch:
            # Aim the zoom-in, e.g. at the ground for the default '{object} is on the ground' prompt.
            self.camera.turn(dpitch=self.args.object_pitch - self.camera.state()["pitch"])
        start = self.cursor()
        self.emit("scene-prompt", obj)
        time.sleep(1.0)
        with self.cond:
            refused = next((s for _, e, s in self.events[start:]
                            if e == "server-state" and str(s).startswith("object insertion unavailable")), None)
        features = (self.server_config or {}).get("features") or {}
        result = self._zoom()
        result["object"] = obj
        result["object_insertion"] = "unavailable" if refused else "requested"
        if refused:
            result["refusal"] = refused
            if features.get("objects"):
                raise StepFailed(f"server advertises features.objects but refused the prompt: {refused}")
        elif self.labels is not None and obj not in self.labels:
            raise StepFailed(f"object '{obj}' not in the scene labels {self.labels}")
        result["labels"] = self.labels
        return result

    def step_undo(self):
        if not self.points_before:
            raise StepFailed("undo: no scene-changing step before it in this scenario")
        job, before = self.points_before[-1]
        start = self.cursor()
        final = self.run_job("undo", lambda: self.emit("undo"))
        stats = self.scene_stats_after(start, "undo")
        after = stats.get("num_points") if stats else None
        self.points_before.pop()
        if final.get("message") != "Undo done":
            raise StepFailed(f"undo: unexpected result '{final.get('message')}'")
        if before is None or after is None:
            raise StepFailed(f"undo: cannot verify the point count (before {job}: {before}, after undo: {after})")
        if after != before:
            raise StepFailed(f"undo of {job} did not restore the point count ({before} expected, got {after})")
        return {"undone": job, "num_points": after}

    def step_save(self):
        final = self.run_job("save", lambda: self.emit("save"))
        message = final.get("message") or ""
        m = re.match(r"Saved: (.+)$", message)
        if not m:
            raise StepFailed(f"save: unexpected message '{message}'")
        path = m.group(1).strip()
        result = {"path": path}
        if os.path.isfile(path):
            result["bytes"] = os.path.getsize(path)
            parts = os.path.normpath(path).split(os.sep)
            if len(parts) < 4 or parts[-2] != "scenes":
                raise StepFailed(f"save: {path} is not under <runs>/<name>/<time>/scenes/")
        elif self.args.session_check:
            raise StepFailed(f"save: {path} does not exist")
        else:
            result["warning"] = "the file is not visible from this machine"
        self.saved_pth.append(path)
        return result

    def step_delete(self):
        vm = self.camera.view_matrix()
        return self.scene_job("delete", lambda: self.emit("delete", vm), expect_change="decrease")

    def step_crack_fix(self):
        return self.scene_job("crack_fix", lambda: self.emit("fix-small-cracks"), expect_change="increase")

    def step_hq_nvs(self):
        gen3c_before = self._gen3c_requests()
        result = self.scene_job("hq_nvs", lambda: self.emit("generate-nvs-hq"), expect_change="increase")
        self._check_gen3c_frames(gen3c_before, result)
        return result

    def step_orbit(self):
        n0 = self.frame_count
        self.emit("generate-nvs")
        self.wait_for(lambda ev: True if self.frame_count - n0 >= 22 else None, self.remaining(120),
                      "the orbit preview frames")
        return {"frames": self.frame_count - n0}

    def step_reject_move(self):
        base = self.require_base_focal()
        self.camera.set_focal(base)
        self.settle()
        payload = dict(self.camera.pose(), addToTrajectory=True)
        info = self.run_job("move", lambda: self.emit("gen", payload), accept_timeout=30, expect_reject=True)
        self._check_alive()
        return info

    def step_reject_zoom(self):
        info = self._zoom(expect_reject=True)
        self._check_alive()
        return info

    def _check_alive(self):
        """The server keeps streaming frames and stays idle after a refused request."""
        n0 = self.frame_count
        self.wait_for(lambda ev: True if self.frame_count > n0 else None, self.remaining(30),
                      "a 'frame' after the refused request")
        state = (self.last_status or {}).get("state")
        if state not in ("idle", None):
            raise StepFailed(f"server state is '{state}' after the refused request")

    # ---- driver ----

    def run(self, steps):
        ok = True
        error = None
        try:
            self.connect()
        except StepFailed as e:
            ok, error = False, str(e)
            steps = []
        for index, name in enumerate(steps):
            fn = getattr(self, f"step_{name}", None)
            t_step = time.time()
            self.log(f"=== step {index + 1}/{len(steps)}: {name}")
            record = {"step": name, "index": index}
            try:
                record["result"] = fn()
                record["ok"] = True
            except StepFailed as e:
                record["ok"] = False
                record["error"] = str(e)
            except Exception as e:  # driver bug: report it like a failure
                record["ok"] = False
                record["error"] = f"{type(e).__name__}: {e}"
                record["traceback"] = traceback.format_exc()
            record["seconds"] = round(time.time() - t_step, 1)
            # The step's frame: wait for frames rendered after the job (a quick job such as undo
            # finishes between two frames).
            n0 = self.frame_count
            try:
                self.wait_for(lambda ev: True if self.frame_count >= n0 + 2 else None, self.remaining(3.0),
                              "a frame after the step")
            except StepFailed:
                pass
            record["frame"] = self._save_frame(f"step{index + 1:02d}_{name}.jpg")
            record["frames_total"] = self.frame_count
            self.steps.append(record)
            self.log(f"=== step {name}: {'OK' if record['ok'] else 'FAILED: ' + record['error']} "
                     f"({record['seconds']} s)")
            if not record["ok"]:
                ok = False
                if not self.args.keep_going:
                    break
        self.stop.set()
        try:
            self.sio.disconnect()
        except Exception:
            pass
        if ok and self.steps and self.frame_count == 0:
            ok, error = False, "no 'frame' event was received"
        server_log = None
        if self.args.server_log:
            server_log = _scan_server_log(self.args.server_log)
            if ok and server_log.get("tracebacks"):
                ok, error = False, f"{server_log['tracebacks']} Traceback(s) in {self.args.server_log}"
        report = {
            "ok": ok and all(s["ok"] for s in self.steps),
            "error": error,
            "url": self.args.url,
            "scenario": steps,
            "started": datetime.fromtimestamp(self.t0).isoformat(timespec="seconds"),
            "seconds": round(time.time() - self.t0, 1),
            "server_config": self.server_config,
            "last_status": self.last_status,
            "num_points": self.num_points,
            "labels": self.labels,
            "frames_received": self.frame_count,
            "frames_saved": len(self.saved_frames),
            "videos_saved": self.video_count,
            "saved_pth": self.saved_pth,
            "coz_seed": self.args.coz_seed,
            "connects": self.connect_count,
            "server_log": server_log,
            "steps": self.steps,
            "events": [{"t": round(t, 2), "event": e, "data": s} for t, e, s in self.events
                       if e not in ("server-state",) or not str(s).startswith(("Collecting frames", "Orbit preview:"))],
        }
        path = os.path.join(self.out, "report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=1, default=str)
        self.log(f"report: {path}  ({'PASS' if report['ok'] else 'FAIL'})")
        return report


def _scan_server_log(path):
    """Count 'Traceback' blocks in the server log and keep the exception line of each."""
    try:
        with open(path, errors="replace") as f:
            text = f.read().replace("\r", "\n")
    except OSError as e:
        return {"error": str(e)}
    lines = text.split("\n")
    errors = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            tail = next((lines[j] for j in range(i + 1, min(i + 200, len(lines)))
                         if lines[j] and not lines[j].startswith((" ", "\t", "Traceback"))), "")
            errors.append(tail.strip())
    return {"path": path, "tracebacks": len(errors), "errors": errors[:20]}


def _png_size(path):
    """(width, height) of a PNG from its header, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    except OSError:
        return None


STEPS = ("boot", "frames", "move", "zoom", "object_zoom", "undo", "save", "delete", "crack_fix", "hq_nvs",
         "orbit", "reject_move", "reject_zoom")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7747", help="server URL")
    parser.add_argument("--scenario", default="boot,crack_fix,delete,undo,save",
                        help="comma-separated steps: " + ", ".join(STEPS))
    parser.add_argument("--out", default=None, help="output directory (default runs/_e2e/<time>)")
    parser.add_argument("--timeout", type=float, default=3600, help="overall time budget in seconds")
    parser.add_argument("--connect_timeout", type=float, default=600, help="how long to retry connecting")
    parser.add_argument("--coz_seed", type=int, default=None,
                        help="Chain-of-Zoom seed sent with every zoom request ('gen' cozSeed)")
    parser.add_argument("--object", default="a ladybug", help="object of the object_zoom step")
    parser.add_argument("--object_pitch", type=float, default=0.0,
                        help="camera pitch (rad, W/S keys) of the object_zoom start view; negative looks down")
    parser.add_argument("--move_yaw", type=float, default=0.3, help="yaw of the move step (rad)")
    parser.add_argument("--move_forward", type=float, default=0.1, help="forward movement of the move step")
    parser.add_argument("--zoom_presses", type=int, default=14, help="V presses between H and R of a zoom")
    parser.add_argument("--pose_hz", type=float, default=10.0, help="'render-pose' rate")
    parser.add_argument("--frame_interval", type=float, default=5.0, help="seconds between saved frames")
    parser.add_argument("--min_frames", type=int, default=3, help="frames the 'frames' step waits for")
    parser.add_argument("--frames_timeout", type=float, default=300, help="timeout of the 'frames' step")
    parser.add_argument("--no_session_check", dest="session_check", action="store_false",
                        help="do not inspect the server's session directory (server on another machine)")
    parser.add_argument("--allow_reconnect", action="store_true",
                        help="keep waiting for a job when the connection drops (the client reconnects)")
    parser.add_argument("--keep_going", action="store_true", help="run the remaining steps after a failure")
    parser.add_argument("--server_log", default=None,
                        help="server log file: the run fails when it contains a Traceback")
    args = parser.parse_args(argv)

    steps = [s.strip() for s in args.scenario.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s) {unknown}; choose from {', '.join(STEPS)}")
    if steps and steps[0] not in ("boot", "frames"):
        steps.insert(0, "boot")
    if args.out is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.out = os.path.join(root, "runs", "_e2e", datetime.now().strftime("%Y%m%d-%H%M%S"))
    report = Driver(args).run(steps)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
