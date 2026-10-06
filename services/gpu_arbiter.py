"""GPU arbitration between the WonderZoom main process and the model workers.

Tenants are 'main_models' (the heavy models of the main process) and the workers 'gen3c', 'coz'
and 'step1x'. Each tenant lives on one physical GPU. The policy is decided per physical GPU:

    resident   every tenant keeps its weights on the GPU; leases are no-ops.
    exclusive  tenants that share the GPU are time-multiplexed: lease(X) parks every other tenant
               of that GPU in host RAM (workers through their 'suspend' op, the main process
               through the park function registered by run.py) and resumes X.
    auto       exclusive when the summed resident estimates of the tenants exceed the GPU memory
               minus gpu.reserve_gb, resident otherwise.

A GPU with a single tenant is always resident. Leases are re-entrant per thread and nest: when an
inner lease of another tenant ends, the outer tenant is resumed. When the outermost lease ends the
tenant stays resident until another tenant needs the GPU (set gpu.restore_main_after_service to
move the main models back right away). While a worker holds a lease on the main process's GPU,
render_allowed is cleared so that the render thread can pause (gpu.pause_render_during_services).
The gate is cleared before anything is moved, and a render thread that draws inside
render_frame() is waited for, so no frame is rendered while the main models are parked and the
worker's weights are brought onto the GPU.
"""
import contextlib
import threading
import time

from .base import ServiceError

MAIN_TENANT = "main_models"


class Tenant:
    def __init__(self, name, token, kind, resident_gb=0.0):
        self.name = name
        self.token = token
        self.kind = kind  # 'main' | 'worker'
        self.resident_gb = float(resident_gb or 0.0)
        self.park_fn = None
        self.unpark_fn = None
        self.release_cache_fn = None
        self.parked = kind == "worker"  # a worker that is not running holds no GPU memory


class _Device:
    def __init__(self, token):
        self.token = token
        self.tenants = []
        self.policy = "resident"
        self.total_gb = None
        self.reason = ""
        self.cond = threading.Condition()
        self.owner = None
        self.stack = []


class GpuArbiter:
    def __init__(self, policy="auto", reserve_gb=4.0, pause_render_during_services=True,
                 restore_main_after_service=False, render_wait_s=10.0, log=print):
        self.policy = policy
        self.reserve_gb = float(reserve_gb)
        self.pause_render_during_services = bool(pause_render_during_services)
        self.restore_main_after_service = bool(restore_main_after_service)
        self.render_wait_s = float(render_wait_s)
        self._log = log or (lambda *args, **kwargs: None)
        self._tenants = {}
        self._devices = {}
        self._warned = set()
        # Set while the render thread may run; cleared while a worker leases the main GPU.
        self.render_allowed = threading.Event()
        self.render_allowed.set()
        # Held by the render thread while it draws a frame (render_frame()). Re-entrant, so that a
        # lease taken from inside render_frame() does not wait for itself.
        self._render_busy = threading.RLock()

    def log(self, message):
        self._log(f"[gpu] {message}")

    # ---- registration ----

    def add_tenant(self, name, token, kind, resident_gb=0.0):
        if name in self._tenants:
            self.remove_tenant(name)
        tenant = Tenant(name, str(token), kind, resident_gb)
        self._tenants[name] = tenant
        device = self._devices.setdefault(tenant.token, _Device(tenant.token))
        device.tenants.append(name)
        return tenant

    def remove_tenant(self, name):
        tenant = self._tenants.pop(name, None)
        if tenant is not None:
            device = self._devices.get(tenant.token)
            if device is not None and name in device.tenants:
                device.tenants.remove(name)

    def set_callbacks(self, name, park_fn=None, unpark_fn=None, release_cache_fn=None, parked=None):
        tenant = self._tenant(name)
        tenant.park_fn = park_fn
        tenant.unpark_fn = unpark_fn
        tenant.release_cache_fn = release_cache_fn
        if parked is not None:
            tenant.parked = bool(parked)

    def note_parked(self, name, parked):
        """Record a placement change made outside a lease (start-up, crash, restart)."""
        if name in self._tenants:
            self._tenants[name].parked = bool(parked)

    def _tenant(self, name):
        tenant = self._tenants.get(name)
        if tenant is None:
            raise ServiceError(f"unknown GPU tenant {name!r}; known: {', '.join(sorted(self._tenants))}")
        return tenant

    def has_tenant(self, name):
        return name in self._tenants

    def can_park(self, name):
        tenant = self._tenants.get(name)
        return tenant is not None and tenant.park_fn is not None

    def holds_lease(self, name):
        """True when the calling thread holds a lease on the GPU of tenant `name`."""
        tenant = self._tenants.get(name)
        device = self._devices.get(tenant.token) if tenant is not None else None
        return device is not None and device.owner == threading.get_ident()

    def co_resident_gb(self, name):
        """Summed resident_gb of the other tenants on the GPU of `name`."""
        tenant = self._tenants.get(name)
        device = self._devices.get(tenant.token) if tenant is not None else None
        if device is None:
            return 0.0
        return sum(self._tenants[n].resident_gb for n in device.tenants if n != name)

    # ---- policy ----

    def resolve(self, totals_gb=None):
        """Decide the policy of every GPU. totals_gb maps a device token to its memory in GiB."""
        totals_gb = totals_gb or {}
        for device in self._devices.values():
            device.total_gb = totals_gb.get(device.token)
            tenants = [self._tenants[n] for n in device.tenants]
            if len(tenants) < 2:
                device.policy, device.reason = "resident", "single tenant"
            elif self.policy in ("resident", "exclusive"):
                device.policy, device.reason = self.policy, f"gpu.policy={self.policy}"
            else:
                need = sum(t.resident_gb for t in tenants)
                if device.total_gb is None:
                    device.policy = "exclusive"
                    device.reason = f"auto: GPU memory unknown, tenants need ~{need:.0f} GB"
                else:
                    budget = device.total_gb - self.reserve_gb
                    device.policy = "exclusive" if need > budget else "resident"
                    device.reason = f"auto: tenants need ~{need:.0f} GB, budget {budget:.0f} GB"
            if len(tenants) > 1:
                self.log(f"GPU {device.token}: {device.policy} for {', '.join(device.tenants)} ({device.reason})")

    def device_policy(self, token):
        device = self._devices.get(str(token))
        return device.policy if device is not None else "resident"

    def tenant_policy(self, name):
        tenant = self._tenants.get(name)
        return self.device_policy(tenant.token) if tenant is not None else "resident"

    def device_memory_gb(self, name):
        tenant = self._tenants.get(name)
        device = self._devices.get(tenant.token) if tenant is not None else None
        return device.total_gb if device is not None else None

    def is_exclusive(self, name):
        return self.tenant_policy(name) == "exclusive"

    # ---- parking ----

    def _park(self, tenant, reason):
        if tenant.parked:
            return
        if tenant.park_fn is None:
            if tenant.name not in self._warned:
                self._warned.add(tenant.name)
                self.log(f"cannot park {tenant.name} on GPU {tenant.token} (no park function registered); "
                         "it stays on the GPU")
            return
        t0 = time.time()
        done = None
        try:
            done = tenant.park_fn()
        except Exception as e:
            if tenant.kind == "worker":
                # WorkerClient.suspend kills a worker whose suspend failed (error reply, timeout or
                # exit), so it holds no GPU memory; the next request restarts it.
                self.log(f"suspending {tenant.name} failed ({e}); treating it as parked")
                done = False
            else:
                raise ServiceError(f"parking {tenant.name} for {reason} failed: {e}") from e
        tenant.parked = True
        if done is not False:  # False: nothing to move (e.g. the worker is not running)
            self.log(f"parked {tenant.name} for {reason} in {time.time() - t0:.1f} s")

    def _unpark(self, tenant):
        if not tenant.parked:
            return
        t0 = time.time()
        done = tenant.unpark_fn() if tenant.unpark_fn is not None else False
        tenant.parked = False
        if done is not False:
            self.log(f"resumed {tenant.name} in {time.time() - t0:.1f} s")

    def _switch_to(self, device, name):
        target = self._tenants[name]
        for other in device.tenants:
            if other != name:
                self._park(self._tenants[other], name)
        if target.kind == "worker":
            main = self._tenants.get(MAIN_TENANT)
            if main is not None and main.token == device.token and main.release_cache_fn is not None:
                try:
                    main.release_cache_fn()
                except Exception as e:
                    self.log(f"releasing the main process cache failed: {e}")
        self._unpark(target)

    def park(self, name):
        """Park one tenant now (used after start-up in exclusive mode)."""
        tenant = self._tenant(name)
        if self.device_policy(tenant.token) == "exclusive":
            self._park(tenant, "start-up")

    # ---- render gate ----

    @contextlib.contextmanager
    def render_frame(self):
        """For the render thread: `with arbiter.render_frame() as allowed: if allowed: draw()`.

        Drawing inside the block lets a lease that pauses rendering wait for the frame in progress
        before it parks the main models and brings a worker onto the main GPU.
        """
        with self._render_busy:
            yield self.render_allowed.is_set()

    def _pauses_render(self, device, name):
        """True when making `name` the active tenant of `device` pauses the render thread."""
        if not self.pause_render_during_services or device.policy != "exclusive":
            return False
        main = self._tenants.get(MAIN_TENANT)
        tenant = self._tenants.get(name)
        return (main is not None and main.token == device.token
                and tenant is not None and tenant.kind == "worker")

    def _pause_render(self):
        """Clear the render gate, then wait for a frame that is being drawn in render_frame()."""
        self.render_allowed.clear()
        if self._render_busy.acquire(timeout=self.render_wait_s):
            self._render_busy.release()
        else:
            self.log(f"the render thread did not finish its frame within {self.render_wait_s:.0f} s; "
                     "continuing")

    # ---- leases ----

    def _update_render_gate(self, device):
        """Recompute the gate after the lease stack of `device` changed (call with device.cond held).

        Only the main GPU's leases affect the render thread; leases on other GPUs leave the gate
        alone, so they cannot reopen it while a worker is being brought onto the main GPU.
        """
        main = self._tenants.get(MAIN_TENANT)
        if main is None or main.token != device.token:
            return
        top = self._tenants.get(device.stack[-1]) if device.stack else None
        if top is not None and self._pauses_render(device, top.name):
            self.render_allowed.clear()
        else:
            self.render_allowed.set()

    def _release(self, device, name, restore=True):
        with device.cond:
            device.stack.pop()
            remaining = list(device.stack)
        try:
            if restore:
                if remaining and remaining[-1] != name:
                    if self._pauses_render(device, remaining[-1]):
                        self._pause_render()  # before the worker comes back onto the main GPU
                    self._switch_to(device, remaining[-1])
                elif not remaining and self.restore_main_after_service and self._tenants[name].kind == "worker":
                    main = self._tenants.get(MAIN_TENANT)
                    if main is not None and main.token == device.token and main.unpark_fn is not None:
                        self._switch_to(device, MAIN_TENANT)
        except Exception as e:
            self.log(f"restoring the GPU after the {name} lease failed: {e}")
        finally:
            with device.cond:
                if not device.stack:
                    device.owner = None
                    device.cond.notify_all()
                self._update_render_gate(device)

    @contextlib.contextmanager
    def lease(self, name):
        """Make `name` the active tenant of its GPU for the duration of the block (exclusive policy)."""
        tenant = self._tenant(name)
        device = self._devices[tenant.token]
        if device.policy != "exclusive":
            yield
            return
        me = threading.get_ident()
        with device.cond:
            while device.owner not in (None, me):
                device.cond.wait()
            device.owner = me
            device.stack.append(name)
        try:
            if self._pauses_render(device, name):
                # Before anything moves: no frame may be rendered while the main models are parked
                # and the worker's weights are brought onto the main GPU.
                self._pause_render()
            self._switch_to(device, name)
        except BaseException:
            # The switch may have parked the outer tenant already (e.g. the main models, before a
            # worker failed to resume): give it back its GPU, as at the end of a normal lease.
            self._release(device, name)
            raise
        with device.cond:
            self._update_render_gate(device)
        try:
            yield
        finally:
            self._release(device, name)

    # ---- monitoring ----

    def status(self):
        devices = {}
        for token, device in self._devices.items():
            devices[token] = {
                "policy": device.policy,
                "reason": device.reason,
                "total_gb": device.total_gb,
                "tenants": {n: {"parked": self._tenants[n].parked, "kind": self._tenants[n].kind,
                                "resident_gb": self._tenants[n].resident_gb} for n in device.tenants},
                "lease_stack": list(device.stack),
            }
        return {"policy": self.policy, "render_allowed": self.render_allowed.is_set(), "devices": devices}
