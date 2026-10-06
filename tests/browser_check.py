#!/usr/bin/env python
"""Real-browser check of the WonderZoom web UI (headless Chromium driven by Playwright).

Part 'gen' opens the generation page that run.py serves at '/' (splat-main/index_gen.html) and drives it
with real key presses: frames on the canvas, the idle status, W/A/S/D and the arrow keys, V/B (focal),
H/J/R (trajectory; R at the base focal length must be refused without Gen3C), Ctrl+Alt+Space (crack
fix), Z (undo), X (save, then the saved file is checked) and the object prompt box (keys must not fire
while typing and must work again after Enter). If the page connects while the server is still loading
the models, H is pressed first: the refusal must leave the trajectory count at 0.
Part 'render' (alias 'view') opens the same page served by 'run.py --view' at <render_url>/ and checks
the viewer: title 'WonderZoom Viewer', status idle, only the W/A/S/D, arrow, V/B and Space rows shown
(no prompt box, no videos), frames, W/A/S/D, the arrows, V/B, generation keys refused with a
'view mode' reply (H, R, X: no trajectory point, no image change, no saved file) and Space (orbit).
It then opens splat-main/index_gen.html as a file:// page with ?server=<render_url>.

Playwright is not part of the release envs. Install it into a separate venv:
    python3 -m venv ~/wz-pw && ~/wz-pw/bin/pip install playwright
    ~/wz-pw/bin/python -m playwright install chromium          # or pass --chromium /path/to/chrome

Run it from the repository root, with the server on the GPU you choose:
    export CUDA_VISIBLE_DEVICES=0
    bash scripts/run_server.sh --example_config config/more_examples/street.yaml --no_services --port 7760 &
    ~/wz-pw/bin/python tests/browser_check.py --gen_url http://127.0.0.1:7760 --start_render_server
or let the driver start and stop both servers (logs go to <out>/):
    ~/wz-pw/bin/python tests/browser_check.py --start_gen_server --start_render_server
The view server is 'scripts/run_server.sh --view --example_config <cfg> --port <render port>'; it shows
--pth_path, else the scene saved by X in part 'gen', else the pth_path of the config. It uses
--render_python, else $WZ_MAIN_PYTHON, else the interpreter registered for 'main' (scripts/register_env.py).
--phases render checks only the viewer.

Outputs: <out>/NN_<step>.png screenshots and <out>/report.json. Exit code 0 when every check passed,
1 when a check failed, 2 on a usage error.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Records every text written into #server-connect / #status-line (also several writes in one task)
# and counts the frames drawn on any canvas.
INIT_SCRIPT = r"""
(() => {
  window.__wzLog = [];
  window.__wzDraws = 0;
  const draw = CanvasRenderingContext2D.prototype.drawImage;
  CanvasRenderingContext2D.prototype.drawImage = function (...a) {
    window.__wzDraws++;
    return draw.apply(this, a);
  };
  const watch = () => {
    for (const id of ['server-connect', 'status-line']) {
      const el = document.getElementById(id);
      if (!el) continue;
      window.__wzLog.push([Date.now(), id, el.innerText]);
      new MutationObserver((records) => {
        for (const r of records) {
          if (!r.addedNodes.length) continue;
          const text = Array.from(r.addedNodes).map((n) => n.textContent).join('');
          window.__wzLog.push([Date.now(), id, text]);
        }
      }).observe(el, { childList: true });
    }
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', watch);
  else watch();
})();
"""

# Downsampled grey levels of the canvas plus simple statistics.
CANVAS_JS = r"""
() => {
  const c = document.getElementById('canvas');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const step = 4, px = [];
  let sum = 0, sum2 = 0, lit = 0;
  for (let y = 0; y < c.height; y += step) {
    for (let x = 0; x < c.width; x += step) {
      const i = (y * c.width + x) * 4;
      const g = (d[i] + d[i + 1] + d[i + 2]) / 3;
      px.push(g); sum += g; sum2 += g * g; if (g > 8) lit++;
    }
  }
  const n = px.length, mean = sum / n;
  return { w: c.width, h: c.height, px: px, mean: mean,
           std: Math.sqrt(Math.max(0, sum2 / n - mean * mean)), lit: lit / n, draws: window.__wzDraws };
}
"""

# main_stream.js keeps its state in top-level bindings of a classic script.
CAMERA_JS = r"""
() => ({ fx: active_camera.fx, fy: active_camera.fy, yaw: yaw, pitch: pitch, movement: movement.slice(),
         count: trajectoryPointCount, base: baseFocalLength,
         focal_text: (document.getElementById('focal-x') || {}).innerText || '',
         active: document.activeElement ? (document.activeElement.id || document.activeElement.tagName) : null })
"""


class Checker:
    def __init__(self, out):
        self.out = out
        self.checks = []
        self.shots = []
        self.n_shot = 0

    def check(self, name, ok, detail=None):
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  -> {detail}"), flush=True)
        return bool(ok)

    def shot(self, page, step, canvas_only=False):
        self.n_shot += 1
        path = os.path.join(self.out, f"{self.n_shot:02d}_{step}.png")
        try:
            if canvas_only:
                page.locator('#canvas').screenshot(path=path)
            else:
                page.screenshot(path=path, full_page=True)
            self.shots.append(path)
        except Exception as e:  # a screenshot must not end the run
            print(f"screenshot {path} failed: {e}")
        return path

    @property
    def failed(self):
        return [c["name"] for c in self.checks if not c["ok"]]


# ------------------------------------------------------------------------------------------------
# Page helpers
# ------------------------------------------------------------------------------------------------
def canvas(page):
    return page.evaluate(CANVAS_JS)


def camera(page):
    return page.evaluate(CAMERA_JS)


def diff(a, b):
    if a["w"] != b["w"] or a["h"] != b["h"]:
        return float("inf")
    return sum(abs(x - y) for x, y in zip(a["px"], b["px"])) / max(1, len(a["px"]))


def log_len(page):
    return page.evaluate("() => window.__wzLog.length")


def log_since(page, since, el_id=None):
    entries = page.evaluate("(i) => window.__wzLog.slice(i)", since)
    return [text for _, eid, text in entries if el_id is None or eid == el_id]


def wait_log(page, el_id, pattern, since, timeout):
    """First text written to el_id after log index `since` that matches pattern, else None."""
    rx = re.compile(pattern)
    deadline = time.time() + timeout
    while True:
        for text in log_since(page, since, el_id):
            if rx.search(text):
                return text
        if time.time() > deadline:
            return None
        time.sleep(0.1)


def wait_frames(page, n=2, settle=1.0, timeout=20):
    """Wait `settle` seconds and until n more frames were drawn."""
    start = page.evaluate("() => window.__wzDraws")
    t0 = time.time()
    time.sleep(settle)
    while page.evaluate("() => window.__wzDraws") < start + n:
        if time.time() - t0 > timeout:
            return False
        time.sleep(0.05)
    return True


def hold(page, key, seconds):
    page.keyboard.down(key)
    time.sleep(seconds)
    page.keyboard.up(key)


def scene_points(page):
    text = page.evaluate("() => (document.getElementById('scene-stats') || {}).innerText || ''")
    m = re.search(r"([\d,]+) points", text)
    return (int(m.group(1).replace(",", "")) if m else None), text


def wait_title(page, title, timeout=10):
    """run.py sets the title once server-config arrives: poll until it is `title`; returns the last title."""
    deadline = time.time() + timeout
    while True:
        current = page.title()
        if current == title or time.time() > deadline:
            return current
        time.sleep(0.2)


def new_page(browser, url):
    page = browser.new_page(viewport={"width": 1500, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

    def on_console(m):
        if m.type == "error" and not (m.location or {}).get("url", "").endswith("favicon.ico"):
            errors.append(f"console: {m.text}")

    page.on("console", on_console)
    page.add_init_script(INIT_SCRIPT)
    page.goto(url)
    return page, errors


def check_keys_change_image(page, ck, prefix, keys, hold_s):
    """Each key (held) must change the streamed image; a pause without keys must not."""
    time.sleep(0.5)
    a = canvas(page)
    wait_frames(page, n=3, settle=1.0)
    b = canvas(page)
    still = diff(a, b)
    ck.check(f"{prefix}: image is stable without keys", still < 1.5, f"mean abs diff {still:.2f}")
    for key in keys:
        before = canvas(page)
        cam0 = camera(page)
        hold(page, key, hold_s.get(key, 0.6))
        wait_frames(page, n=3, settle=1.0)
        after = canvas(page)
        cam1 = camera(page)
        d = diff(before, after)
        moved = (cam1["yaw"], cam1["pitch"], cam1["movement"]) != (cam0["yaw"], cam0["pitch"], cam0["movement"])
        ck.check(f"{prefix}: {key} changes the image", d > 2.0 and moved,
                 f"mean abs diff {d:.2f}, yaw {cam0['yaw']:.3f}->{cam1['yaw']:.3f}, "
                 f"pitch {cam0['pitch']:.3f}->{cam1['pitch']:.3f}, "
                 f"movement {[round(v, 3) for v in cam1['movement']]}")
        ck.shot(page, f"{prefix}_{key}", canvas_only=True)


def shown_focal(cam):
    """The number shown in the #focal-x element."""
    try:
        return float(cam["focal_text"].split()[-1])
    except (ValueError, IndexError):
        return float("nan")


def check_focal(page, ck, prefix, presses=3):
    cam0 = camera(page)
    base = cam0["base"]
    before = canvas(page)
    for _ in range(presses):
        page.keyboard.press("v")
    wait_frames(page, n=3, settle=1.0)
    cam1 = camera(page)
    zoomed = canvas(page)
    want = base * 1.05 ** presses
    ck.check(f"{prefix}: V x{presses} zooms in", abs(cam1["fx"] - want) < 1e-6 * want and
             abs(shown_focal(cam1) - cam1["fx"]) <= 0.05,  # shown with 1 decimal
             f"fx {cam0['fx']} -> {cam1['fx']:.4f} (want {want:.4f}), shown '{cam1['focal_text']}'")
    d = diff(before, zoomed)
    ck.check(f"{prefix}: zoomed image differs", d > 2.0, f"mean abs diff {d:.2f}")
    ck.shot(page, f"{prefix}_zoom_in", canvas_only=True)
    for _ in range(presses + 2):  # two extra presses: B never goes below the base
        page.keyboard.press("b")
    wait_frames(page, n=3, settle=1.0)
    cam2 = camera(page)
    back = canvas(page)
    ck.check(f"{prefix}: B returns exactly to the base focal length",
             cam2["fx"] == base and cam2["fy"] == base and shown_focal(cam2) == base,
             f"fx {cam2['fx']} (base {base}), shown '{cam2['focal_text']}'")
    d2 = diff(before, back)
    ck.check(f"{prefix}: image after V/B matches the image before", d2 < 1.5, f"mean abs diff {d2:.2f}")


# ------------------------------------------------------------------------------------------------
# Generation page (run.py)
# ------------------------------------------------------------------------------------------------
def wait_status(page, since, state, job=None, timeout=60):
    pattern = "^" + re.escape(state) + (r" \[" + re.escape(job) + r"\]" if job else "")
    return wait_log(page, "status-line", pattern, since, timeout)


def run_gen(browser, args, ck):
    page, errors = new_page(browser, args.gen_url.rstrip("/") + "/")
    ck.check("gen: page title", page.title() == "WonderZoom Generation", page.title())
    connected = wait_log(page, "server-connect", r"^Connected to server", 0, 30)
    ck.check("gen: socket connected", connected is not None, connected)
    state = page.evaluate("() => document.body.getAttribute('data-server-state')")
    if state is None:
        time.sleep(1.0)
        state = page.evaluate("() => document.body.getAttribute('data-server-state')")

    # H while the models load: run.py refuses it, so the count must stay 0 and no 'added' text appears.
    if state == "loading":
        ck.shot(page, "gen_loading")
        idx = log_len(page)
        page.keyboard.press("h")
        reply = wait_log(page, "server-connect", r"ignored|Trajectory point \d+ added", idx, 15)
        cam = camera(page)
        texts = log_since(page, idx, "server-connect")
        if reply and "ignored" in reply:
            ck.check("gen(loading): refused H leaves the count at 0 and shows the refusal",
                     cam["count"] == 0 and not any("Trajectory point" in t for t in texts),
                     f"reply '{reply}', count {cam['count']}, texts {texts}")
        else:
            # kf_gen exists once the models are loaded, so a point sent during the initial scene build is
            # accepted; the count must then be the server's.
            m = re.search(r"Trajectory point (\d+) added", reply or "")
            ck.check("gen(loading): H answered (accepted while the initial scene was built)",
                     m is not None and cam["count"] == int(m.group(1)), f"reply '{reply}', count {cam['count']}")
        ck.shot(page, "gen_loading_h")
    else:
        print(f"note: server state was '{state}' when the page opened; the refused-H check needs 'loading'")

    idle = wait_log(page, "status-line", r"^idle", 0, args.boot_timeout)
    ck.check("gen: status line shows idle", idle is not None, idle)
    if idle is None:
        ck.shot(page, "gen_not_idle")
        return None
    ok_frames = wait_frames(page, n=5, settle=1.5, timeout=60)
    c = canvas(page)
    ck.check("gen: frames are drawn on the canvas", ok_frames and c["draws"] >= 5, f"{c['draws']} frames drawn")
    ck.check("gen: canvas is not blank", c["std"] > 5 and 5 < c["mean"] < 250 and c["lit"] > 0.5,
             f"mean {c['mean']:.1f}, std {c['std']:.1f}, lit {c['lit']:.2f}, canvas {c['w']}x{c['h']}")
    status_text = page.inner_text("#status-line")
    ck.check("gen: status-line data-state is idle",
             page.get_attribute("#status-line", "data-state") == "idle", status_text)
    points0, stats0 = scene_points(page)
    ck.check("gen: scene stats shown", points0 is not None and points0 > 0, stats0)
    ck.shot(page, "gen_idle")

    # Movement keys, in back-and-forth pairs so that the camera ends near the start view.
    keys = ["w", "s", "a", "d", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]
    check_keys_change_image(page, ck, "gen", keys, {"w": 0.8, "s": 0.8, "a": 0.6, "d": 0.6,
                                                    "ArrowUp": 0.4, "ArrowDown": 0.4,
                                                    "ArrowLeft": 0.4, "ArrowRight": 0.4})
    check_focal(page, ck, "gen")

    # Trajectory: J (clear), H (point 1), R at the base focal length (refused without Gen3C: the point
    # stays), H (point 2), J (cleared). Count and text must follow the server's replies.
    idx = log_len(page)
    page.keyboard.press("j")
    r = wait_log(page, "server-connect", r"Trajectory cleared", idx, 10)
    ck.check("gen: J clears the trajectory", r is not None and camera(page)["count"] == 0, r)
    idx = log_len(page)
    page.keyboard.press("h")
    r = wait_log(page, "server-connect", r"Trajectory point \d+ added|ignored", idx, 10)
    ck.check("gen: H adds point 1 (server reply)",
             r == "Trajectory point 1 added. Press R to generate video." and camera(page)["count"] == 1,
             f"'{r}', count {camera(page)['count']}")
    ck.shot(page, "gen_h")
    cam = camera(page)
    if cam["fx"] != cam["base"]:
        ck.check("gen: R is pressed at the base focal length", False, cam)
    gen3c_on = "feature-off" not in (page.get_attribute("tr[data-feature='gen3c']", "class") or "")
    if gen3c_on:
        # R would start a real camera move (minutes of Gen3C): tests/e2e_headless.py covers that.
        print("note: the server has Gen3C; R is not pressed (run the server with --no_services)")
    else:
        idx = log_len(page)
        page.keyboard.press("r")
        r = wait_log(page, "server-connect", r"ignored|Gen3C", idx, 15)
        time.sleep(1.0)
        texts = log_since(page, idx)
        ck.check("gen: R at the base fx shows 'Gen3C service not enabled'",
                 r is not None and "Gen3C service not enabled" in r, r)
        cam = camera(page)
        st = page.get_attribute("#status-line", "data-state")
        ck.check("gen: refused R keeps the point and the idle state", cam["count"] == 1 and st == "idle",
                 f"count {cam['count']}, state {st}, texts {texts}")
        ck.shot(page, "gen_r_refused")
        idx = log_len(page)
        page.keyboard.press("h")
        r = wait_log(page, "server-connect", r"Trajectory point \d+ added|ignored", idx, 10)
        ck.check("gen: second H is point 2 (the server kept point 1)",
                 r is not None and r.startswith("Trajectory point 2 added") and camera(page)["count"] == 2, r)
        idx = log_len(page)
        page.keyboard.press("j")
        r = wait_log(page, "server-connect", r"Trajectory cleared", idx, 10)
        ck.check("gen: J resets the count to 0", r is not None and camera(page)["count"] == 0,
                 f"'{r}', count {camera(page)['count']}")

    # Crack fix (Ctrl+Alt+Space): busy [crack_fix], then idle.
    idx = log_len(page)
    page.keyboard.press("Control+Alt+Space")
    busy = wait_status(page, idx, "busy", "crack_fix", 30)
    ck.check("gen: Ctrl+Alt+Space starts the crack fix", busy is not None, busy)
    time.sleep(3)
    ck.shot(page, "gen_crack_fix_busy")
    t0 = time.time()
    done = wait_status(page, idx, "idle", None, args.job_timeout) if busy else None
    points1, stats1 = scene_points(page)
    ck.check("gen: crack fix finishes (idle)", done is not None and "last job: crack_fix" in stats1,
             f"{done!r} after {time.time() - t0:.0f} s; stats '{stats1}'")
    wait_frames(page, n=3, settle=1.0)
    ck.shot(page, "gen_crack_fix_done")

    # Undo (Z): the point count returns to the one before the crack fix.
    idx = log_len(page)
    page.keyboard.press("z")
    done = wait_status(page, idx, "idle", None, 120)
    points2, stats2 = scene_points(page)
    ck.check("gen: Z undoes the crack fix", done is not None and "Undo done" in done and points2 == points0,
             f"{done!r}; points {points0} -> {points1} -> {points2}")
    wait_frames(page, n=3, settle=1.0)
    ck.shot(page, "gen_undo")

    # Save (X): 'Saved: <path>' and a torch zip archive on disk.
    idx = log_len(page)
    page.keyboard.press("x")
    saved = wait_log(page, "server-connect", r"^Saved: ", idx, 300)
    done = wait_status(page, idx, "idle", None, 60)
    pth = saved[len("Saved: "):].strip() if saved else None
    if pth and not os.path.isabs(pth):
        pth = os.path.join(REPO, pth)
    size = os.path.getsize(pth) if pth and os.path.isfile(pth) else 0
    ck.check("gen: X saves the scene", saved is not None and done is not None and size > 1e6
             and zipfile.is_zipfile(pth), f"{pth} ({size / 1e6:.1f} MB), status {done!r}")
    ck.shot(page, "gen_saved")

    # Object prompt: typing must not drive the camera; Enter blurs the box and the keys work again.
    box = page.locator("#prompt-box")
    forced = box.is_disabled()
    if forced:
        ck.check("gen: prompt box disabled when object insertion is off",
                 "disabled" in (box.get_attribute("placeholder") or ""), box.get_attribute("placeholder"))
        # Without Step1X the box is disabled; enable it to test the typing and blur logic of main_stream.js.
        page.evaluate("() => { document.getElementById('prompt-box').disabled = false; }")
    cam0 = camera(page)
    box.click()
    page.keyboard.type("a vase dbw", delay=60)
    page.keyboard.down("a")
    time.sleep(0.4)
    page.keyboard.up("a")
    time.sleep(0.3)
    cam1 = camera(page)
    ck.check("gen: keys typed into the prompt box do not move or zoom",
             cam1["active"] == "prompt-box" and (cam1["fx"], cam1["yaw"], cam1["pitch"], cam1["movement"]) ==
             (cam0["fx"], cam0["yaw"], cam0["pitch"], cam0["movement"]),
             f"active {cam1['active']}, fx {cam0['fx']}->{cam1['fx']}, yaw {cam0['yaw']:.4f}->{cam1['yaw']:.4f}")
    ck.shot(page, "gen_prompt_typed")
    idx = log_len(page)
    page.keyboard.press("Enter")
    time.sleep(0.3)
    cam2 = camera(page)
    # With insertion off the server answers 'object insertion unavailable: ...'; otherwise the page itself
    # confirms the prompt.
    want = r"^object insertion unavailable" if forced else r'^Object "'
    reply = wait_log(page, "server-connect", want, idx, 10)
    ck.check("gen: Enter sends the prompt and blurs the box", cam2["active"] == "BODY" and reply is not None,
             f"active {cam2['active']}, reply '{reply}'")
    page.keyboard.press("v")
    time.sleep(0.2)
    cam3 = camera(page)
    page.keyboard.press("b")
    before = canvas(page)
    hold(page, "a", 0.5)
    wait_frames(page, n=3, settle=1.0)
    after = canvas(page)
    cam4 = camera(page)
    ck.check("gen: V and A work again after Enter", cam3["fx"] > cam2["fx"] and cam4["yaw"] != cam2["yaw"]
             and diff(before, after) > 2.0,
             f"fx {cam2['fx']}->{cam3['fx']:.2f}, yaw {cam2['yaw']:.4f}->{cam4['yaw']:.4f}, "
             f"image diff {diff(before, after):.2f}")
    hold(page, "d", 0.5)
    if forced:
        page.evaluate("() => { document.getElementById('prompt-box').disabled = true; }")
        ck.check("gen: server cleared the refused prompt", box.input_value() == "", repr(box.input_value()))
    wait_frames(page, n=3, settle=1.0)
    ck.shot(page, "gen_after_prompt")

    ck.check("gen: no JavaScript errors", not errors, errors)
    page.close()
    return pth


# ------------------------------------------------------------------------------------------------
# Viewer (run.py --view): the same page, with server-config view_only
# ------------------------------------------------------------------------------------------------
# Visibility of the key rows of the controls table, by their <kbd> texts (e.g. 'WASD', 'CtrlAltSpace').
KEY_ROWS_JS = r"""
() => Array.from(document.querySelectorAll('.controls-panel tr')).map((tr) => ({
  keys: Array.from(tr.querySelectorAll('kbd')).map((k) => k.textContent.trim()).join(''),
  visible: tr.getClientRects().length > 0 }))
"""
VIEW_ROWS = ["WASD", "\u2191\u2193\u2190\u2192", "VB", "Space"]
GEN_ROWS = ["H", "R", "Z", "X", "C", "J", "Q"]


def visible(page, selector):
    return page.evaluate("(s) => { const e = document.querySelector(s); return !!e && e.getClientRects().length > 0; }",
                         selector)


def files_since(t0, dirs):
    """Files under dirs (runs/ recursively, the others top-level only) modified at or after t0."""
    found = []
    for d, recursive in dirs:
        if not os.path.isdir(d):
            continue
        walk = os.walk(d) if recursive else [(d, [], os.listdir(d))]
        for root, _, names in walk:
            for n in names:
                path = os.path.join(root, n)
                try:
                    if os.path.isfile(path) and os.path.getmtime(path) >= t0:
                        found.append(path)
                except OSError:
                    pass
    return found


def press_refused(page, key, settle=1.0):
    """Press key; returns (first 'view mode' reply or None, server-connect texts, image diff, camera after)."""
    idx = log_len(page)
    before = canvas(page)
    page.keyboard.press(key)
    reply = wait_log(page, "server-connect", r"view mode", idx, 10)
    wait_frames(page, n=3, settle=settle)
    return reply, log_since(page, idx, "server-connect"), diff(before, canvas(page)), camera(page)


def run_render(browser, args, ck, pth=None):
    page, errors = new_page(browser, args.render_url.rstrip("/") + "/")
    connected = wait_log(page, "server-connect", r"^Connected to server", 0, 60)
    ck.check("view: socket connected (page served by run.py --view)", connected is not None, connected)
    title = wait_title(page, "WonderZoom Viewer")
    ck.check("view: page title", title == "WonderZoom Viewer", title)
    idle = wait_log(page, "status-line", r"^idle", 0, 30)
    ck.check("view: status line shows idle with 'Viewing <file>'", idle is not None and "Viewing " in idle, idle)
    ok_frames = wait_frames(page, n=5, settle=1.5, timeout=120)
    c = canvas(page)
    ck.check("view: frames are drawn on the canvas", ok_frames and c["draws"] >= 5, f"{c['draws']} frames drawn")
    ck.check("view: canvas is not blank", c["std"] > 5 and 5 < c["mean"] < 250 and c["lit"] > 0.5,
             f"mean {c['mean']:.1f}, std {c['std']:.1f}, lit {c['lit']:.2f}, canvas {c['w']}x{c['h']}")
    ck.shot(page, "view_loaded")

    # Only the viewing controls: no prompt box, no videos, no generation key rows.
    hidden = {s: not visible(page, s) for s in ("#prompt-box", "#send-button", ".video-area")}
    ck.check("view: prompt box and video area are hidden", all(hidden.values()), hidden)
    rows = page.evaluate(KEY_ROWS_JS)
    shown = [r["keys"] for r in rows if r["visible"]]
    ck.check("view: W/A/S/D, arrow, V/B and Space rows are shown",
             all(k in shown for k in VIEW_ROWS), f"shown rows {shown}")
    ck.check("view: H/R/Z/X/C/J/Q rows are hidden", not any(k in shown for k in GEN_ROWS), f"shown rows {shown}")

    check_keys_change_image(page, ck, "view", ["w", "s", "a", "d", "ArrowUp", "ArrowDown"],
                            {"w": 0.8, "s": 0.8, "a": 0.6, "d": 0.6, "ArrowUp": 0.4, "ArrowDown": 0.4})
    check_focal(page, ck, "view")

    # Generation keys: the server refuses them ('<action> ignored: view mode ...'); nothing may change.
    for key, name in (("h", "H"), ("r", "R")):
        reply, texts, d, cam = press_refused(page, key)
        ck.check(f"view: {name} is refused (view mode) and changes nothing",
                 reply is not None and cam["count"] == 0 and not any("Trajectory point" in t for t in texts)
                 and d < 1.5, f"reply '{reply}', count {cam['count']}, image diff {d:.2f}, texts {texts}")
        ck.shot(page, f"view_{key}_refused")
    t0 = time.time() - 1.0
    reply, texts, d, cam = press_refused(page, "x", settle=2.0)
    scene_dir = os.path.dirname(os.path.abspath(pth)) if pth else None
    new = files_since(t0, [(os.path.join(REPO, "runs"), True), (REPO, False)] +
                      ([(scene_dir, False)] if scene_dir else []))
    new = [f for f in new if not f.endswith(".log")]
    ck.check("view: X is refused (view mode) and saves no file",
             reply is not None and not any(t.startswith("Saved") for t in texts) and not new and d < 1.5,
             f"reply '{reply}', new files {new}, image diff {d:.2f}")
    ck.shot(page, "view_x_refused")

    # Space: orbit preview. The server sends 'Orbit preview: i/N' for every orbit frame and
    # 'Orbit preview finished' at the end; the view moves, and afterwards it is back at the start view.
    idx = log_len(page)
    a = canvas(page)
    page.keyboard.press("Space")
    first = wait_log(page, "server-connect", r"^Orbit preview: \d+/\d+", idx, 30)
    diffs = []
    t_end = time.time() + 1.0
    while time.time() < t_end:
        diffs.append(diff(a, canvas(page)))
        time.sleep(0.1)
    ck.shot(page, "view_orbit")
    ck.check("view: Space starts the orbit (progress text, the view moves)",
             first is not None and max(diffs) > 2.0, f"'{first}', max image diff {max(diffs):.2f}")
    deadline = time.time() + 120
    n_prev, quiet_since = -1, time.time()
    while time.time() < deadline:  # finished: no progress message for 2 s
        n = sum(t.startswith("Orbit preview") for t in log_since(page, idx, "server-connect"))
        if n != n_prev:
            n_prev, quiet_since = n, time.time()
        elif time.time() - quiet_since > 2.0:
            break
        time.sleep(0.2)
    wait_frames(page, n=3, settle=1.0)
    end = canvas(page)
    ck.check("view: orbit ends and the view returns to the start view",
             time.time() < deadline and diff(a, end) < 1.5,
             f"{n_prev} progress messages, image diff to the start view {diff(a, end):.2f}")
    wait_frames(page, n=3, settle=1.0)
    ck.shot(page, "view_after_orbit")
    ck.check("view: no JavaScript errors", not errors, errors)
    page.close()

    # The same page as a local file:// copy with ?server= (one client at a time: the first page is closed).
    url = "file://" + os.path.join(REPO, "splat-main", "index_gen.html") + "?server=" + args.render_url
    page, errors = new_page(browser, url)
    connected = wait_log(page, "server-connect", r"^Connected to server", 0, 60)
    ck.check("view(file://): socket connected via ?server=", connected is not None, connected)
    title = wait_title(page, "WonderZoom Viewer")
    ok_frames = wait_frames(page, n=5, settle=1.5, timeout=60)
    c = canvas(page)
    ck.check("view(file://): viewer title and frames drawn",
             title == "WonderZoom Viewer" and ok_frames and c["draws"] >= 5 and c["std"] > 5,
             f"title '{title}', {c['draws']} frames drawn, std {c['std']:.1f}")
    ck.shot(page, "view_file_page")
    ck.check("view(file://): no JavaScript errors", not errors, errors)
    page.close()


# ------------------------------------------------------------------------------------------------
# Servers
# ------------------------------------------------------------------------------------------------
def http_ok(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_http(url, proc, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if http_ok(url):
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def server_env():
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if env.get("WZ_KEEP_LD_LIBRARY_PATH", "0") != "1":
        env.pop("LD_LIBRARY_PATH", None)  # as scripts/run_server.sh
    return env


def start(cmd, log_path, env=None):
    print("starting: " + " ".join(cmd) + f"  (log {log_path})", flush=True)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=dict(server_env(), **(env or {})),
                            start_new_session=True)


def stop(proc, name):
    if proc is None or proc.poll() is not None:
        return
    print(f"stopping {name} (pid {proc.pid})", flush=True)
    for sig, wait in ((signal.SIGTERM, 90), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def config_pth(example_config):
    """pth_path of the example config (relative paths: repository root), or None."""
    path = example_config if os.path.isabs(example_config) else os.path.join(REPO, example_config)
    try:
        with open(path) as f:
            m = re.search(r"^pth_path:\s*['\"]?([^'\"#\s]+)", f.read(), re.M)
    except OSError:
        return None
    return os.path.normpath(os.path.join(REPO, m.group(1))) if m else None


def port_of(url):
    m = re.search(r":(\d+)/?$", url)
    if not m:
        sys.exit(f"URL needs an explicit port: {url}")
    return m.group(1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--gen_url", default="http://127.0.0.1:7760", help="generation server (run.py)")
    p.add_argument("--render_url", "--view_url", dest="render_url", default="http://127.0.0.1:7761",
                   help="view server (run.py --view)")
    p.add_argument("--phases", default="gen,render", help="comma-separated: gen, render (or view)")
    p.add_argument("--example_config", default="config/more_examples/street.yaml")
    p.add_argument("--pth_path", default=None,
                   help="scene for the render phase (default: the one saved by X, else the config's pth_path)")
    p.add_argument("--start_gen_server", action="store_true",
                   help="start 'scripts/run_server.sh --no_services' on the --gen_url port and stop it at the end")
    p.add_argument("--start_render_server", "--start_view_server", dest="start_render_server", action="store_true",
                   help="start 'scripts/run_server.sh --view' on the --render_url port and stop it at the end")
    p.add_argument("--render_python", default=None,
                   help="interpreter of the view server (WZ_MAIN_PYTHON for scripts/run_server.sh)")
    p.add_argument("--chromium", default=None, help="Chromium/Chrome executable (default: Playwright's)")
    p.add_argument("--headed", action="store_true", help="show the browser window")
    p.add_argument("--boot_timeout", type=float, default=1800, help="seconds to wait for a server to be ready")
    p.add_argument("--job_timeout", type=float, default=900, help="seconds to wait for the crack fix")
    p.add_argument("--out", default=None, help="output directory (default runs/_browser/<time>)")
    args = p.parse_args()

    phases = [{"view": "render"}.get(s.strip(), s.strip()) for s in args.phases.split(",") if s.strip()]
    if not phases or set(phases) - {"gen", "render"}:
        p.print_usage()
        return 2
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is missing: pip install playwright (in a separate venv), then "
              "python -m playwright install chromium")
        return 2

    out = os.path.abspath(args.out or os.path.join(REPO, "runs", "_browser", datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    ck = Checker(out)
    procs = {}
    report = {"gen_url": args.gen_url, "render_url": args.render_url, "phases": phases,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    t_start = time.time()
    try:
        with sync_playwright() as pw:
            launch = {"headless": not args.headed, "args": ["--no-sandbox"]}
            if args.chromium:
                launch["executable_path"] = args.chromium
            browser = pw.chromium.launch(**launch)
            report["browser"] = f"chromium {browser.version}"
            pth = args.pth_path
            if "gen" in phases:
                if args.start_gen_server:
                    procs["gen"] = start(["bash", "scripts/run_server.sh", "--example_config", args.example_config,
                                          "--no_services", "--port", port_of(args.gen_url)],
                                         os.path.join(out, "server_gen.log"))
                ready = wait_http(args.gen_url.rstrip("/") + "/", procs.get("gen"), args.boot_timeout)
                if ck.check("gen: server answers HTTP", ready, args.gen_url):
                    saved = run_gen(browser, args, ck)
                    pth = pth or saved
                report["saved_pth"] = pth
            if "render" in phases:
                if args.start_render_server:
                    # As the gen server; without a .pth run.py --view shows the pth_path of the config.
                    procs["render"] = start(["bash", "scripts/run_server.sh", "--view", "--example_config",
                                             args.example_config, "--port", port_of(args.render_url)] +
                                            (["--pth_path", pth] if pth else []),
                                            os.path.join(out, "server_render.log"),
                                            {"WZ_MAIN_PYTHON": args.render_python} if args.render_python else None)
                ready = wait_http(args.render_url.rstrip("/") + "/", procs.get("render"), args.boot_timeout)
                if ck.check("view: server answers HTTP", ready, args.render_url):
                    run_render(browser, args, ck, pth or config_pth(args.example_config))
            browser.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        ck.check("driver ran without an exception", False, repr(e))
    finally:
        for name in list(procs)[::-1]:
            stop(procs[name], name)

    report.update(seconds=round(time.time() - t_start, 1), passed=not ck.failed, failed=ck.failed,
                  checks=ck.checks, screenshots=ck.shots)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n{len(ck.checks) - len(ck.failed)}/{len(ck.checks)} checks passed; report: {out}/report.json")
    return 0 if not ck.failed else 1


if __name__ == "__main__":
    sys.exit(main())
