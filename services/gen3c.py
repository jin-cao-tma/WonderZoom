"""Client API of the Gen3C service (novel-view video for camera moves and high-quality views).

Used as svc.gen3c.generate(...) on a services.ServiceManager.
"""
import os
import time

from .base import ServiceError, draw_legacy_request_marker, require_dir, require_file

NAME = "gen3c"


class Gen3cService:
    def __init__(self, manager):
        self._manager = manager

    @property
    def enabled(self):
        return self._manager.enabled(NAME)

    def generate(self, condition_image, frames_dir, masks_dir, prompt, num_steps, out_dir=None, seed=None):
        """Generate the 121-frame Gen3C video for a camera move.

        Args:
            condition_image: PNG of the input view (frames/saved_frames/input.png).
            frames_dir: directory with the warped renderings (holes are black): 121 frames, or
                fewer, which the worker pads to 121 by repeating the last frame and mask, as the
                paper-era service did. More than 121 frames are rejected.
            masks_dir: directory with one hole mask (holes = 255) per frame.
            prompt: text prompt. None is sent as the literal string 'None', exactly like the
                paper-era service, whose REPL command was an f-string (HQ views pass None);
                services.gen3c.legacy_none_prompt: false sends '' instead.
            num_steps: diffusion steps (18 for camera moves, 10 for HQ views). None uses
                services.gen3c.num_steps (18). The paper-era service's own default was 3, which
                no paper call used; every paper call passes num_steps.
            out_dir: output directory; default <session>/services/gen3c/<request id>.
            seed: optional per-request seed (initial noise and condition augmentation); None keeps
                the process-global RNG (paper behaviour) unless services.gen3c.request_seed is set.

        Returns:
            dict(video_path=..., frames_dir=..., n_frames=int); frames are frame_%08d.png at 1280x704.

        Raises:
            ServiceError when the service is disabled, the inputs are invalid or generation fails.
        """
        client = self._manager.client(NAME)
        if prompt is None:
            prompt = "None" if self._manager.service_cfg(NAME).get("legacy_none_prompt", True) else ""
        args = {
            "condition_image": require_file(condition_image, "Gen3C condition image", NAME),
            "frames_dir": require_dir(frames_dir, "Gen3C frames directory", NAME),
            "masks_dir": require_dir(masks_dir, "Gen3C masks directory", NAME),
            "prompt": str(prompt),
            "num_steps": None if num_steps is None else int(num_steps),
            "out_dir": os.path.abspath(out_dir) if out_dir else self._manager.new_request_dir(NAME),
            "seed": None if seed is None else int(seed),
        }
        draw_legacy_request_marker()  # keeps run.py's `random` stream (CoZ seeds) as in the paper run
        t0 = time.time()
        self._manager.log(f"[services] gen3c: generate ({args['num_steps'] or 'default'} steps, "
                          f"prompt {args['prompt']!r}) -> {args['out_dir']}")
        with self._manager.lease(NAME):
            result = client.request("generate", args)
        result = result or {}
        video_path = result.get("video_path")
        out_frames = result.get("frames_dir")
        n_frames = int(result.get("n_frames") or 0)
        if not video_path or not os.path.isfile(video_path):
            raise ServiceError(f"gen3c: the worker reported success but the video is missing ({video_path})",
                               service=NAME, op="generate")
        if not out_frames or not os.path.isdir(out_frames) or n_frames <= 0:
            raise ServiceError(f"gen3c: the worker reported success but no frames were written ({out_frames})",
                               service=NAME, op="generate")
        self._manager.log(f"[services] gen3c: {n_frames} frames in {time.time() - t0:.0f} s")
        return {"video_path": video_path, "frames_dir": out_frames, "n_frames": n_frames}
