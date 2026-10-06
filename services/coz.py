"""Client API of the Chain-of-Zoom service (one super-resolution step per zoom keyframe).

Used as svc.coz.super_resolve_dual(...) on a services.ServiceManager.
"""
import os
import random
import time

from .base import ServiceError, draw_legacy_request_marker, require_file

NAME = "coz"


class CozService:
    def __init__(self, manager):
        self._manager = manager
        self.last_seed = None
        self.last_prompt = None

    @property
    def enabled(self):
        return self._manager.enabled(NAME)

    def super_resolve_dual(self, prev_png, cur_png, out_png, prompt=None, seed=None):
        """Super-resolve cur_png (a zoom of prev_png) and write the result to out_png.

        The VLM prompt is computed from both images unless `prompt` is given; the SR network only
        sees cur_png, at its native resolution.

        Args:
            seed: None uses services.coz.seed when set, otherwise random.randint(1, 999999) drawn
                from this process's `random` module (seeded by run.py with config['seed']). As in
                the paper-era drivers, every Gen3C, CoZ and Step1X call first draws one
                random.randint(1000, 9999) request marker from the same stream, so the same
                config and the same sequence of calls give the paper run's seeds.

        Returns:
            The absolute path of out_png.

        Raises:
            ServiceError when the service is disabled or the request fails.
        """
        client = self._manager.client(NAME)
        prev_png = require_file(prev_png, "Chain-of-Zoom previous image", NAME)
        cur_png = require_file(cur_png, "Chain-of-Zoom current image", NAME)
        out_png = os.path.abspath(out_png)
        draw_legacy_request_marker()  # the paper-era driver drew its marker id before the seed
        if seed is None:
            seed = self._manager.service_cfg(NAME).get("seed")
        if seed is None:
            seed = random.randint(1, 999999)
        # A stale file from an earlier zoom must never be mistaken for this result.
        if os.path.exists(out_png) and out_png not in (prev_png, cur_png):
            os.remove(out_png)
        args = {"prev_png": prev_png, "cur_png": cur_png, "out_png": out_png,
                "prompt": None if prompt is None else str(prompt), "seed": int(seed)}
        t0 = time.time()
        with self._manager.lease(NAME):
            result = client.request("super_resolve_dual", args)
        result = result or {}
        if not os.path.isfile(out_png):
            raise ServiceError(f"coz: the worker reported success but {out_png} is missing",
                               service=NAME, op="super_resolve_dual")
        self.last_seed = result.get("seed", seed)
        self.last_prompt = result.get("prompt")
        self._manager.log(f"[services] coz: {os.path.basename(out_png)} in {time.time() - t0:.0f} s "
                          f"(seed {self.last_seed}, prompt {self.last_prompt!r})")
        return out_png
