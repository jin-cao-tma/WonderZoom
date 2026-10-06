"""ServiceManager: GPU placement, start-up and lifetime of the WonderZoom model workers.

    from services import ServiceManager, ServiceError, load_services_config
    cfg = load_services_config(overrides=dict(policy=args.gpu_policy, main_gpu=args.main_gpu))
    svc = ServiceManager(cfg, session_dir, log=print)
    svc.register_main_tenant(park_fn, unpark_fn, device=main_gpu)   # exclusive mode, before start()
    svc.start()                     # or svc.start_async() + svc.wait_ready('gen3c')
    out = svc.gen3c.generate(...)   # svc.coz.super_resolve_dual(...), svc.step1x.edit(...)
    with svc.lease('main_models'):  # around heavy main-model work (no-op under the resident policy)
        ...
    with svc.render_frame() as allowed:   # render thread, around each frame
        if allowed:
            ...
    svc.shutdown()

wait_ready() may be called while holding a lease: when the start-up thread would need that GPU,
the worker is started in the calling thread instead of deadlocking.
"""
import atexit
import contextlib
import os
import subprocess
import threading
import time
import traceback

from omegaconf import OmegaConf

from .base import (WORKERS_DIR, ServiceError, WorkerClient, auto_cuda_home, build_worker_env,
                   parent_visible_devices, physical_device)
from .config import SERVICE_NAMES
from .coz import CozService
from .gen3c import Gen3cService
from .gpu_arbiter import MAIN_TENANT, GpuArbiter
from .step1x import Step1XService

# Resident GPU memory estimates in GiB, used by gpu.policy=auto when resident_gb is not configured.
DEFAULT_RESIDENT_GB = {MAIN_TENANT: 16.0, "gen3c": 36.0, "coz": 28.0, "step1x": 42.0}


def auto_step1x_offload(total_gb, need_gb, exclusive, reserve_gb, co_resident_gb=0.0):
    """Resolve services.step1x.offload: auto.

    The paper-era service never offloaded (offload=False). Offloading moves Qwen2.5-VL-7B and the
    12B DiT host->GPU->host on every edit, so it is only turned on when step1x's resident
    footprint (need_gb) does not fit next to what else stays on its GPU:
      resident policy:  the other tenants of the GPU stay loaded (co_resident_gb);
      exclusive policy: they are parked, but keep their CUDA contexts and leftovers (reserve_gb).
    Unknown GPU memory offloads (always fits).
    """
    if total_gb is None:
        return True
    taken = float(reserve_gb) if exclusive else float(co_resident_gb or 0.0)
    return total_gb - taken < float(need_gb)

# services.<svc> keys that only configure the client side (everything else goes to the worker).
_CLIENT_KEYS = {"enabled", "python", "device", "init_timeout_s", "request_timeout_s", "park_timeout_s",
                "resident_gb", "hf_hub_offline", "cuda_home", "env"}


def query_gpu_memory_gb(timeout=30.0):
    """Total memory of every GPU in GiB, keyed by nvidia-smi index and UUID ({} if unavailable).

    nvidia-smi numbers GPUs in PCI bus order; CUDA does too when CUDA_DEVICE_ORDER=PCI_BUS_ID,
    which is also the usual order on machines with identical GPUs.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    totals = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gb = float(parts[2]) / 1024.0
        except ValueError:
            continue
        totals[parts[0]] = gb
        totals[parts[1]] = gb
    return totals


class ServiceManager:
    def __init__(self, cfg, session_dir=None, log=print):
        self.cfg = cfg
        self._log = log or (lambda *args, **kwargs: None)
        paths = cfg.get("paths") or {}
        if session_dir is None:
            session_dir = os.path.join(paths.get("runs_dir") or "runs", "_services", time.strftime("%Y%m%d-%H%M%S"))
        self.session_dir = os.path.abspath(session_dir)
        self.logs_dir = os.path.join(self.session_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        self._counter_lock = threading.Lock()
        self._request_counter = 0
        self._closed = False

        gpu = cfg.get("gpu") or {}
        self._visible = parent_visible_devices()
        self.arbiter = GpuArbiter(
            policy=gpu.get("policy", "auto"),
            reserve_gb=gpu.get("reserve_gb", 4),
            pause_render_during_services=gpu.get("pause_render_during_services", True),
            restore_main_after_service=gpu.get("restore_main_after_service", False),
            render_wait_s=gpu.get("render_pause_timeout_s", 10),
            log=self._log)
        # Event set while the render thread may render (cleared while a worker leases the main GPU).
        self.render_allowed = self.arbiter.render_allowed

        self.main_device = int(gpu.get("main_device") or 0)
        self.main_token = physical_device(self.main_device, self._visible)
        self._main_resident_gb = gpu.get("main_resident_gb") or DEFAULT_RESIDENT_GB[MAIN_TENANT]
        self.arbiter.add_tenant(MAIN_TENANT, self.main_token, "main", self._main_resident_gb)

        self._service_cfgs = {}
        self._tokens = {}
        self._problems = {}
        self._clients = {}
        self._start_events = {name: threading.Event() for name in SERVICE_NAMES}
        self._start_requested = dict.fromkeys(SERVICE_NAMES, False)
        # Set by whichever thread runs a requested start-up (start-up thread or wait_ready).
        self._start_claimed = dict.fromkeys(SERVICE_NAMES, False)
        self._claim_lock = threading.Lock()

        for name in SERVICE_NAMES:
            if not self.enabled(name):
                continue
            scfg = self.service_cfg(name)
            try:
                token = physical_device(scfg.get("device") or 0, self._visible)
            except ServiceError as e:
                self._problems[name] = f"{name}: {e}"
                self.log(f"[services] {self._problems[name]}")
                continue
            self._tokens[name] = token
            self.arbiter.add_tenant(name, token, "worker", scfg.get("resident_gb") or DEFAULT_RESIDENT_GB[name])

        self._gpu_totals = query_gpu_memory_gb() if self._tokens else {}
        self.arbiter.resolve(self._gpu_totals)

        for name in self._tokens:
            client = self._build_client(name)
            self._clients[name] = client
            self.arbiter.set_callbacks(name, park_fn=client.suspend, unpark_fn=client.resume, parked=True)

        self.gen3c = Gen3cService(self)
        self.coz = CozService(self)
        self.step1x = Step1XService(self)
        atexit.register(self._atexit)

    # ---- configuration ----

    def log(self, message):
        self._log(message)

    def service_cfg(self, name):
        """services.<name> as a plain dict ({} when absent)."""
        if name not in self._service_cfgs:
            services = self.cfg.get("services") or {}
            section = services.get(name) if name in services else None
            if section is None:
                self._service_cfgs[name] = {}
            elif OmegaConf.is_config(section):
                self._service_cfgs[name] = OmegaConf.to_container(section, resolve=True)
            else:
                self._service_cfgs[name] = dict(section)
        return self._service_cfgs[name]

    def enabled(self, name):
        """True when services.<name>.enabled is true and an interpreter is registered."""
        if name not in SERVICE_NAMES:
            return False
        scfg = self.service_cfg(name)
        return bool(scfg.get("enabled", False) and scfg.get("python"))

    def _resolved_flags(self, name, scfg):
        """Resolve the 'auto' worker settings once the GPU policy is known."""
        exclusive = self.arbiter.is_exclusive(name)
        total_gb = self.arbiter.device_memory_gb(name)
        flags = {}
        if name == "gen3c":
            park = scfg.get("park_text_encoder", "auto")
            flags["park_text_encoder"] = exclusive if park in (None, "auto") else bool(park)
        if name == "step1x":
            offload = scfg.get("offload", "auto")
            if offload in (None, "auto"):
                need_gb = scfg.get("resident_gb") or DEFAULT_RESIDENT_GB[name]
                offload = auto_step1x_offload(total_gb, need_gb, exclusive, self.arbiter.reserve_gb,
                                              self.arbiter.co_resident_gb(name))
            flags["offload"] = bool(offload)
        return flags

    def _worker_config(self, name, scfg):
        worker_cfg = {key: value for key, value in scfg.items() if key not in _CLIENT_KEYS}
        worker_cfg.update(self._resolved_flags(name, scfg))
        worker_cfg["service"] = name
        return worker_cfg

    def _build_client(self, name):
        scfg = self.service_cfg(name)
        python = scfg.get("python")
        repo_dir = scfg.get("repo_dir")
        cuda_home = scfg.get("cuda_home")
        if cuda_home == "auto":
            cuda_home = auto_cuda_home(python) if python else None
        env = build_worker_env(python, pythonpath=repo_dir or "", cuda_visible_devices=self._tokens[name],
                               cuda_home=cuda_home, hf_hub_offline=bool(scfg.get("hf_hub_offline", False)),
                               extra=scfg.get("env") or {})
        worker_cfg = self._worker_config(name, scfg)
        self.log(f"[services] {name}: GPU {self._tokens[name]} ({self.arbiter.tenant_policy(name)}), "
                 f"python {python}, repo {repo_dir}"
                 + "".join(f", {k}={worker_cfg[k]}" for k in ("park_text_encoder", "offload") if k in worker_cfg))
        return WorkerClient(
            name, python, os.path.join(WORKERS_DIR, f"{name}_worker.py"), repo_dir,
            worker_config=worker_cfg, env=env, log_path=os.path.join(self.logs_dir, f"{name}.log"),
            init_timeout=float(scfg.get("init_timeout_s", 1800)),
            request_timeout=float(scfg.get("request_timeout_s", 900)),
            park_timeout=float(scfg.get("park_timeout_s", 900)),
            log=self._log)

    def client(self, name):
        """The WorkerClient of an enabled service, or ServiceError."""
        if name not in SERVICE_NAMES:
            raise ServiceError(f"unknown service {name!r}")
        if not self.enabled(name):
            raise ServiceError(
                f"{name} service is not enabled (services.{name}.enabled is false or no interpreter is "
                f"registered: run scripts/install_env_{name}.sh or set WZ_{name.upper()}_PYTHON)", service=name)
        if name in self._problems:
            raise ServiceError(self._problems[name], service=name)
        if self._closed:
            raise ServiceError(f"{name}: services were shut down", service=name)
        return self._clients[name]

    def new_request_dir(self, name):
        """A fresh per-request directory <session>/services/<name>/<NNNN>_<HHMMSS>."""
        with self._counter_lock:
            self._request_counter += 1
            index = self._request_counter
        path = os.path.join(self.session_dir, "services", name, f"{index:04d}_{time.strftime('%H%M%S')}")
        os.makedirs(path, exist_ok=True)
        return path

    # ---- start-up ----

    def _select(self, names):
        if names is None:
            return [n for n in SERVICE_NAMES if n in self._clients]
        if isinstance(names, str):
            names = [names]
        selected = []
        for name in names:
            if name not in SERVICE_NAMES:
                raise ServiceError(f"unknown service {name!r}")
            if name in self._clients:
                selected.append(name)
            else:
                self.log(f"[services] {name}: not started (disabled or misconfigured)")
        return selected

    def _prepare_start(self, names):
        names = self._select(names)
        for name in names:
            self._start_requested[name] = True
            with self._claim_lock:
                self._start_claimed[name] = False
            self._start_events[name].clear()
        main_device_shared = any(self._tokens[n] == self.main_token and self.arbiter.is_exclusive(n) for n in names)
        if main_device_shared and not self.arbiter.can_park(MAIN_TENANT):
            self.log("[services] warning: services share the main GPU under the exclusive policy but "
                     "register_main_tenant() was not called; the main models cannot be parked")
        return names

    def _claim_start(self, name):
        """True for the one thread that runs the requested start-up of `name`."""
        with self._claim_lock:
            if self._start_claimed[name]:
                return False
            self._start_claimed[name] = True
            return True

    def _start_one(self, name):
        client = self._clients[name]
        skipped = False  # another thread (wait_ready) runs this start-up and sets the event
        try:
            if self._closed:  # shut down while an earlier worker of the group was loading
                return
            if self.arbiter.is_exclusive(name):
                if self._start_claimed[name]:
                    skipped = True  # already started by wait_ready(); do not swap it in for nothing
                    return
                # Load with the GPU to itself, then park right after the ready message. The claim
                # is taken inside the lease, so wait_ready() can run the start-up in a thread that
                # already holds this GPU without the two ever waiting for each other.
                with self.arbiter.lease(name):
                    if not self._claim_start(name):
                        skipped = True
                        return
                    client.start()
                    self.arbiter.park(name)
            else:
                if not self._claim_start(name):
                    skipped = True
                    return
                client.start()
                self.arbiter.note_parked(name, False)
        except ServiceError as e:
            self.log(f"[services] {name}: {e}")
        except Exception as e:  # never let a start-up thread die silently
            self.log(f"[services] {name}: unexpected error during start-up: {e}\n{traceback.format_exc()}")
        finally:
            if not skipped:
                self._start_events[name].set()

    def _start_selected(self, names):
        # Exclusive GPUs load their workers one after the other; resident ones load in parallel.
        groups, sequential = [], {}
        for name in names:
            if self.arbiter.is_exclusive(name):
                sequential.setdefault(self._tokens[name], []).append(name)
            else:
                groups.append([name])
        groups.extend(sequential.values())
        threads = []
        for group in groups:
            thread = threading.Thread(target=lambda g=group: [self._start_one(n) for n in g],
                                      name="wz-start-" + "-".join(group), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return {name: self._clients[name].state for name in names}

    def start(self, names=None):
        """Start the enabled workers and wait until each is ready or has failed.

        Returns {name: state}. Failures are logged; the failed services raise ServiceError on use.
        """
        return self._start_selected(self._prepare_start(names))

    def start_async(self, names=None):
        """Start the workers in a background thread; use wait_ready(name) to wait for one."""
        names = self._prepare_start(names)
        thread = threading.Thread(target=self._start_selected, args=(names,), name="wz-services-start", daemon=True)
        thread.start()
        return thread

    def wait_ready(self, name, timeout=None):
        """True when the service is running; waits for a pending start-up for up to `timeout` s.

        If the calling thread holds a lease on the service's GPU (e.g. inside
        lease('main_models') under the exclusive policy), the start-up thread could never get
        that GPU, so the worker is started right here instead; this ignores `timeout` and is
        bounded by services.<name>.init_timeout_s.
        """
        client = self._clients.get(name)
        if client is None:
            return False
        if client.state == "ready" and client.alive():
            return True
        if not self._start_requested.get(name):
            return False
        if not self._start_events[name].is_set() and self.arbiter.holds_lease(name):
            self.log(f"[services] {name}: starting it in the calling thread, which holds its GPU lease")
            self._start_one(name)
        else:
            self._start_events[name].wait(timeout)
        return client.state == "ready" and client.alive()

    def restart(self, name):
        """Kill and restart one worker (also after a failed start-up)."""
        client = self.client(name)
        if self.arbiter.is_exclusive(name):
            with self.arbiter.lease(name):
                client.restart()
                self.arbiter.note_parked(name, False)
        else:
            client.restart()
            self.arbiter.note_parked(name, False)

    # ---- GPU arbitration ----

    def register_main_tenant(self, park_fn, unpark_fn, device=None, release_cache_fn=None):
        """Register how the main process parks its heavy models (exclusive policy).

        park_fn moves the models to host RAM and should call torch.cuda.empty_cache(); unpark_fn
        moves them back. release_cache_fn (optional) is called before a worker runs on the main
        GPU, e.g. torch.cuda.empty_cache. Call this before start().
        """
        if device is not None:
            token = physical_device(device, self._visible)
            if token != self.main_token:
                self.log(f"[services] main process GPU is {token}, not gpu.main_device={self.main_device}; "
                         "re-deciding the GPU policies")
                self.main_device = int(device)
                self.main_token = token
                self.arbiter.add_tenant(MAIN_TENANT, token, "main", self._main_resident_gb)
                self.arbiter.resolve(self._gpu_totals)
                for name, client in self._clients.items():
                    if client.state == "stopped":  # not started yet: refresh the 'auto' settings
                        client.worker_config.update(self._resolved_flags(name, self.service_cfg(name)))
        self.arbiter.set_callbacks(MAIN_TENANT, park_fn=park_fn, unpark_fn=unpark_fn,
                                   release_cache_fn=release_cache_fn, parked=False)

    def lease(self, name):
        """Context manager giving `name` ('gen3c' | 'coz' | 'step1x' | 'main_models') its GPU.

        Re-entrant; a no-op under the resident policy and for disabled services.
        """
        if name == MAIN_TENANT or name in self._clients:
            return self.arbiter.lease(name)
        if name in SERVICE_NAMES:
            return contextlib.nullcontext()
        raise ServiceError(f"unknown lease name {name!r}")

    def render_paused(self):
        return not self.render_allowed.is_set()

    def render_frame(self):
        """Context manager for the render thread, around each frame:

            with svc.render_frame() as allowed:
                if allowed:
                    render()

        A worker lease on the main GPU waits (gpu.render_pause_timeout_s) for the frame in
        progress before it parks the main models.
        """
        return self.arbiter.render_frame()

    def wait_render_allowed(self, timeout=None):
        return self.render_allowed.wait(timeout)

    # ---- monitoring ----

    def mem(self, name):
        """GPU memory statistics of a worker (GiB). Never waits for a running request or start-up."""
        client = self.client(name)
        if client.state != "ready" or not client.alive():
            return {"state": client.state, "last_stats": client.last_stats}
        if client.busy_op:
            return {"busy": client.busy_op, "last_stats": client.last_stats}
        try:
            stats = client.mem(blocking=False)
        except ServiceError as e:
            return {"error": str(e), "last_stats": client.last_stats}
        if stats is None:  # a request, suspend/resume or restart holds the worker right now
            return {"busy": client.busy_op or "busy", "last_stats": client.last_stats}
        return dict(stats or {}, suspended=client.suspended, last_stats=client.last_stats)

    def status(self):
        services = {}
        for name in SERVICE_NAMES:
            entry = {"enabled": self.enabled(name)}
            if name in self._problems:
                entry["error"] = self._problems[name]
            client = self._clients.get(name)
            if client is not None:
                entry.update(client.status())
                entry["gpu"] = self._tokens[name]
                entry["policy"] = self.arbiter.tenant_policy(name)
                entry["ready_info"] = client.ready_info
            services[name] = entry
        return {"session_dir": self.session_dir, "main_gpu": self.main_token, "gpu": self.arbiter.status(),
                "services": services}

    # ---- shutdown ----

    def shutdown(self, timeout=10.0):
        """Stop every worker (asks them to exit, then kills their process groups)."""
        if self._closed:
            return
        self._closed = True
        for name, client in self._clients.items():
            try:
                client.stop(timeout=timeout, close=True)
            except Exception as e:
                self.log(f"[services] {name}: error during shutdown: {e}")
        try:
            atexit.unregister(self._atexit)
        except Exception:
            pass

    def _atexit(self):
        self.shutdown(timeout=3.0)
