"""Client API of the Step1X-Edit service (object insertion by image editing).

Used as svc.step1x.edit(...) on a services.ServiceManager.
"""
import os
import time

from .base import ServiceError, draw_legacy_request_marker, require_file

NAME = "step1x"


class Step1XService:
    def __init__(self, manager):
        self._manager = manager

    @property
    def enabled(self):
        return self._manager.enabled(NAME)

    def edit(self, image_png, prompt, out_png, seed=42, num_steps=28, cfg_guidance=6.0, size_level=512):
        """Edit image_png according to prompt and write the result (input size) to out_png.

        The prompt is sent as JSON, so quotes and newlines are safe.

        Returns:
            The absolute path of out_png.

        Raises:
            ServiceError when the service is disabled or the request fails.
        """
        client = self._manager.client(NAME)
        image_png = require_file(image_png, "Step1X-Edit input image", NAME)
        out_png = os.path.abspath(out_png)
        if prompt is None or not str(prompt).strip():
            raise ServiceError("step1x: the edit prompt is empty", service=NAME, op="edit")
        # A stale file from an earlier edit must never be mistaken for this result.
        if os.path.exists(out_png) and out_png != image_png:
            os.remove(out_png)
        args = {"image_png": image_png, "prompt": str(prompt), "out_png": out_png, "seed": int(seed),
                "num_steps": int(num_steps), "cfg_guidance": float(cfg_guidance), "size_level": int(size_level)}
        draw_legacy_request_marker()  # keeps run.py's `random` stream (CoZ seeds) as in the paper run
        t0 = time.time()
        self._manager.log(f"[services] step1x: edit {os.path.basename(image_png)} with {args['prompt']!r}")
        with self._manager.lease(NAME):
            client.request("edit", args)
        if not os.path.isfile(out_png):
            raise ServiceError(f"step1x: the worker reported success but {out_png} is missing",
                               service=NAME, op="edit")
        self._manager.log(f"[services] step1x: {os.path.basename(out_png)} in {time.time() - t0:.0f} s")
        return out_png
