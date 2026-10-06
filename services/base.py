"""Process management for the WonderZoom model workers (standard library only).

WorkerClient starts one worker script (services/workers/<svc>_worker.py) with the interpreter of
the service's own environment, talks newline-delimited JSON with it over stdin/stdout (see
services/workers/_wz_protocol.py), enforces start-up and request timeouts, and kills and lazily
restarts the worker after a crash or a timeout. Worker stderr goes to <session>/logs/<svc>.log.
"""
import glob
import json
import os
import queue
import random
import re
import shlex
import signal
import subprocess
import threading
import time

WORKERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers")

# Environment variables that must not leak from the WonderZoom environment into a worker.
_STRIPPED_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE", "LD_LIBRARY_PATH")

_EOF = object()


class ServiceError(RuntimeError):
    """A model service is unavailable or one of its requests failed."""

    def __init__(self, message, service=None, op=None, remote_traceback=None):
        super().__init__(message)
        self.service = service
        self.op = op
        self.remote_traceback = remote_traceback


# ---------------------------------------------------------------------------------------------
# Environment and device helpers
# ---------------------------------------------------------------------------------------------

def env_prefix(python):
    """Environment prefix of an interpreter (<prefix>/bin/python). Symlinks are not resolved, so
    venv interpreters that link to a base Python still map to the venv."""
    return os.path.dirname(os.path.dirname(os.path.abspath(python)))


# Target directories of the conda CUDA packages (<prefix>/targets/<arch>/{lib,include}).
_CUDA_TARGETS = ("x86_64-linux", "sbsa-linux", "aarch64-linux")


def _nvrtc_majors(lib_dirs):
    """Major versions of the libnvrtc.so.<major>* files in lib_dirs (non-recursive)."""
    majors = set()
    for lib_dir in lib_dirs:
        for path in glob.glob(os.path.join(lib_dir, "libnvrtc.so.*")):
            match = re.match(r"libnvrtc\.so\.(\d+)", os.path.basename(path))
            if match:
                majors.add(match.group(1))
    return majors


def _toolkit_lib_dirs(cuda_home):
    return ([os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")]
            + [os.path.join(cuda_home, "targets", arch, "lib") for arch in _CUDA_TARGETS])


def auto_cuda_home(python, inherited=None):
    """CUDA_HOME for a worker whose environment ships a CUDA toolkit (conda cuda-nvcc/cuda-nvrtc).

    Transformer Engine runs glob('{CUDA_HOME}/**/libnvrtc.so*', recursive=True) at import time;
    over a whole environment prefix (site-packages included) that takes a minute or more on NFS.
    In order of preference:
      1. <prefix>/targets/<arch> when it holds libnvrtc and cuda_runtime.h (conda CUDA packages);
         the glob there is instant and TE finds the NVRTC headers under include/;
      2. the inherited CUDA_HOME (`inherited`, default os.environ) when it ships an NVRTC of the
         same major version as the environment (the paper-era drivers ran with it);
      3. the environment prefix.
    Returns None when the environment has no CUDA toolkit (the inherited CUDA_HOME is kept).
    """
    prefix = env_prefix(python)
    for arch in _CUDA_TARGETS:
        target = os.path.join(prefix, "targets", arch)
        if (glob.glob(os.path.join(target, "lib", "libnvrtc.so*"))
                and os.path.isfile(os.path.join(target, "include", "cuda_runtime.h"))):
            return target
    env_majors = _nvrtc_majors([os.path.join(prefix, "lib")])
    if not (os.path.isfile(os.path.join(prefix, "bin", "nvcc")) or env_majors):
        return None
    inherited = os.environ.get("CUDA_HOME") if inherited is None else inherited
    if inherited and os.path.isdir(inherited) and env_majors & _nvrtc_majors(_toolkit_lib_dirs(inherited)):
        return inherited
    return prefix


def draw_legacy_request_marker():
    """Consume one random.randint(1000, 9999) of the process-global `random` module.

    The paper-era pexpect drivers drew a request marker id this way for every Gen3C generation,
    Chain-of-Zoom call (before its seed) and Step1X edit. run.py seeds `random` with
    config['seed'] and the CoZ seeds come from the same stream, so they only match the paper run
    when every service call consumes the stream exactly as the old drivers did.
    """
    random.randint(1000, 9999)


def build_worker_env(python, pythonpath, cuda_visible_devices, cuda_home=None, hf_hub_offline=False,
                     extra=None, base=None):
    """Environment of a worker process.

    The parent environment is inherited (HF_HOME, HF_TOKEN, PATH, ...), except PYTHONPATH and
    LD_LIBRARY_PATH: a user LD_LIBRARY_PATH can make torch load a different cuDNN than the one its
    wheels ship. The worker sees exactly one GPU.
    """
    env = dict(os.environ if base is None else base)
    for key in _STRIPPED_ENV:
        env.pop(key, None)
    prefix = env_prefix(python)
    env["PATH"] = os.path.join(prefix, "bin") + os.pathsep + env.get("PATH", "")
    env["CONDA_PREFIX"] = prefix
    env["PYTHONPATH"] = pythonpath
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"  # ~/.local packages must not shadow the environment
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
    if hf_hub_offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    for key, value in (extra or {}).items():
        if value is None:
            env.pop(str(key), None)
        else:
            env[str(key)] = str(value)
    return env


def parent_visible_devices(environ=None):
    """The GPU list of the WonderZoom process before run.py narrowed CUDA_VISIBLE_DEVICES.

    run.py stores the original value in WZ_PARENT_VISIBLE_DEVICES. Returns None when every GPU
    is visible (logical index == physical index).
    """
    environ = os.environ if environ is None else environ
    raw = environ.get("WZ_PARENT_VISIBLE_DEVICES")
    if raw is None:
        raw = environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    return [token.strip() for token in raw.split(",") if token.strip()]


def physical_device(logical, visible=None):
    """Map a logical GPU index (as used in services.yaml) to a CUDA_VISIBLE_DEVICES token."""
    logical = int(logical)
    if logical < 0:
        raise ServiceError(f"invalid GPU index {logical}")
    if visible is None:
        return str(logical)
    if logical >= len(visible):
        raise ServiceError(f"GPU index {logical} does not exist: the visible devices are {','.join(visible)}")
    return visible[logical]


def require_file(path, what, service=None):
    """Absolute path of an existing file (workers run in another cwd), or ServiceError."""
    if not path:
        raise ServiceError(f"{what} is not set", service=service)
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise ServiceError(f"{what} not found: {path}", service=service)
    return path


def require_dir(path, what, service=None):
    """Absolute path of an existing directory, or ServiceError."""
    if not path:
        raise ServiceError(f"{what} is not set", service=service)
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(path):
        raise ServiceError(f"{what} not found: {path}", service=service)
    return path


def log_tail(path, max_bytes=4000):
    """Last lines of a worker log, for error messages."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", "replace")
        lines = text.splitlines()
        return "\n".join(lines[-25:])
    except OSError:
        return ""


# ---------------------------------------------------------------------------------------------
# Worker client
# ---------------------------------------------------------------------------------------------

class _Connection:
    def __init__(self, proc):
        self.proc = proc
        self.events = queue.Queue()
        self.pending = {}
        self.lock = threading.Lock()


class WorkerClient:
    """Owns one worker process and serializes requests to it.

    States: 'stopped' (not started yet), 'starting', 'ready', 'dead' (crashed or killed after a
    timeout; restarted on the next request), 'failed' (start-up failed; restart() retries),
    'closed' (shut down).
    """

    def __init__(self, name, python, script, cwd, worker_config=None, env=None, log_path=None,
                 init_timeout=600.0, request_timeout=600.0, park_timeout=900.0, log=print):
        self.name = name
        self.python = python
        self.script = script
        self.cwd = cwd
        self.worker_config = dict(worker_config or {})
        self.env = env
        self.log_path = log_path
        self.init_timeout = init_timeout
        self.request_timeout = request_timeout
        self.park_timeout = park_timeout
        self._log = log or (lambda *args, **kwargs: None)

        self._call_lock = threading.RLock()  # one request (or start/stop) at a time
        self._conn = None
        self._next_id = 1

        self.state = "stopped"
        self.busy_op = None
        self.ready_info = None
        self.last_error = None
        self.last_stats = None
        self.suspended = False
        self.starts = 0
        self.pid = None

    # ---- helpers ----

    def log(self, message):
        self._log(f"[services] {self.name}: {message}")

    def alive(self):
        conn = self._conn
        return conn is not None and conn.proc.poll() is None

    def _command(self):
        return [self.python, "-u", self.script, "--config-json", json.dumps(self.worker_config)]

    def _check_launchable(self):
        problems = []
        if not self.python:
            problems.append("no Python interpreter registered (run scripts/register_env.py or set the "
                            f"WZ_{self.name.upper()}_PYTHON environment variable)")
        elif not (os.path.isfile(self.python) and os.access(self.python, os.X_OK)):
            problems.append(f"interpreter {self.python} does not exist or is not executable")
        if not os.path.isfile(self.script):
            problems.append(f"worker script {self.script} is missing")
        if not self.cwd or not os.path.isdir(self.cwd):
            problems.append(f"repository {self.cwd} is missing (run scripts/setup_third_party.sh {self.name})")
        if problems:
            raise ServiceError(f"{self.name}: cannot start: " + "; ".join(problems), service=self.name)

    @staticmethod
    def _kill_process(proc, grace=5.0):
        """Terminate the worker's whole process group (it runs in its own session)."""
        if proc is None or proc.poll() is not None:
            return
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def _close_connection(self, conn):
        if conn is None:
            return
        self._kill_process(conn.proc)
        for stream in (conn.proc.stdin, conn.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def _reader_loop(self, conn):
        stream = conn.proc.stdout
        try:
            for raw in iter(stream.readline, b""):
                try:
                    msg = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    self.log(f"ignoring non-protocol output: {raw[:200]!r}")
                    continue
                if not isinstance(msg, dict):
                    continue
                if "event" in msg:
                    conn.events.put(msg)
                    continue
                with conn.lock:
                    waiter = conn.pending.pop(msg.get("id"), None)
                if waiter is not None:
                    waiter.put(msg)
                else:
                    self.log(f"ignoring reply with unknown id {msg.get('id')!r}")
        except (OSError, ValueError):
            pass
        finally:
            conn.events.put(_EOF)
            with conn.lock:
                waiters = list(conn.pending.values())
                conn.pending.clear()
            for waiter in waiters:
                waiter.put(_EOF)

    def _exit_message(self, conn, what):
        code = conn.proc.poll()
        if code is None:
            try:
                code = conn.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                code = None
        tail = log_tail(self.log_path) if self.log_path else ""
        message = f"{self.name} worker exited {what} (exit code {code})"
        if self.log_path:
            message += f"; log: {self.log_path}"
        if tail:
            message += "\n--- last lines of the worker log ---\n" + tail
        return message

    # ---- lifecycle ----

    def _spawn(self):
        self._check_launchable()
        if self._conn is not None:
            self._close_connection(self._conn)
            self._conn = None
        cmd = self._command()
        log_file = None
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            log_file = open(self.log_path, "ab", buffering=0)
            shown = " ".join(shlex.quote(c) for c in cmd[:3])
            log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} starting {self.name} worker: "
                           f"{shown} (cwd {self.cwd}, CUDA_VISIBLE_DEVICES="
                           f"{(self.env or os.environ).get('CUDA_VISIBLE_DEVICES')}) =====\n".encode())
        try:
            proc = subprocess.Popen(cmd, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=log_file, start_new_session=True,
                                    close_fds=True)
        except OSError as e:
            raise ServiceError(f"{self.name}: cannot launch {self.python}: {e}", service=self.name) from e
        finally:
            if log_file is not None:
                log_file.close()  # the child keeps its own descriptor
        conn = _Connection(proc)
        threading.Thread(target=self._reader_loop, args=(conn,), name=f"wz-{self.name}-reader",
                         daemon=True).start()
        self._conn = conn
        self.pid = proc.pid
        self.starts += 1
        self.suspended = False
        return conn

    def _start_locked(self):
        t0 = time.time()
        self.state = "starting"
        self.ready_info = None
        try:
            conn = self._spawn()
        except ServiceError as e:
            self.state = "failed"
            self.last_error = str(e)
            raise
        self.log(f"starting worker (pid {conn.proc.pid}, timeout {self.init_timeout:.0f} s"
                 + (f", log {self.log_path})" if self.log_path else ")"))
        try:
            event = conn.events.get(timeout=self.init_timeout if self.init_timeout else None)
        except queue.Empty:
            self._close_connection(conn)
            self.state = "failed"
            self.last_error = f"no ready message within {self.init_timeout:.0f} s"
            raise ServiceError(f"{self.name}: worker did not become ready within {self.init_timeout:.0f} s "
                               "and was killed", service=self.name) from None
        if event is _EOF:
            message = self._exit_message(conn, "during start-up")
            self._close_connection(conn)
            self.state = "failed"
            self.last_error = message
            raise ServiceError(message, service=self.name)
        kind = event.get("event")
        if kind == "ready":
            self.state = "ready"
            self.ready_info = event.get("info") or {}
            self.last_error = None
            self.log(f"ready after {time.time() - t0:.0f} s")
            return self.ready_info
        self._close_connection(conn)
        self.state = "failed"
        error = event.get("error") or f"unexpected start-up message {event!r}"
        self.last_error = error
        raise ServiceError(f"{self.name}: worker failed to start: {error}", service=self.name,
                           remote_traceback=event.get("traceback"))

    def start(self):
        """Start the worker if it is not running; returns the info of its ready message."""
        with self._call_lock:
            if self.state == "closed":
                raise ServiceError(f"{self.name}: service was shut down", service=self.name)
            if self.alive() and self.state == "ready":
                return self.ready_info
            return self._start_locked()

    def restart(self):
        """Kill the worker (if any) and start a new one, also after a failed start."""
        with self._call_lock:
            if self.state == "closed":
                raise ServiceError(f"{self.name}: service was shut down", service=self.name)
            if self._conn is not None:
                self._close_connection(self._conn)
                self._conn = None
            return self._start_locked()

    def ensure_started(self):
        with self._call_lock:
            if self.state == "closed":
                raise ServiceError(f"{self.name}: service was shut down", service=self.name)
            if self.state == "failed":
                raise ServiceError(f"{self.name}: service failed to start: {self.last_error}",
                                   service=self.name)
            if self.alive() and self.state == "ready":
                return
            if self.state in ("ready", "dead"):
                self.log("worker is not running; restarting it")
            self._start_locked()

    def _mark_dead(self, conn, reason):
        self._close_connection(conn)
        if self._conn is conn:
            self._conn = None
        if self.state != "closed":
            self.state = "dead"
        self.suspended = False
        self.last_error = reason

    # ---- requests ----

    def request(self, op, args=None, timeout=None, start=True):
        """Send one request and wait for its reply. Raises ServiceError on any failure.

        start: start (or restart) the worker when it is not running. With start=False a missing
        worker raises ServiceError.
        """
        timeout = self.request_timeout if timeout is None else timeout
        with self._call_lock:
            if start:
                self.ensure_started()
            elif not (self.alive() and self.state == "ready"):
                raise ServiceError(f"{self.name}: worker is not running", service=self.name, op=op)
            conn = self._conn
            rid = self._next_id
            self._next_id += 1
            waiter = queue.Queue(maxsize=1)
            with conn.lock:
                conn.pending[rid] = waiter
            line = json.dumps({"id": rid, "op": op, "args": args or {}}) + "\n"
            self.busy_op = op
            try:
                try:
                    conn.proc.stdin.write(line.encode("utf-8"))
                    conn.proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError) as e:
                    with conn.lock:
                        conn.pending.pop(rid, None)
                    message = self._exit_message(conn, f"before {op}")
                    self._mark_dead(conn, message)
                    raise ServiceError(f"{message} ({e}); it will be restarted on the next call",
                                       service=self.name, op=op) from e
                try:
                    reply = waiter.get(timeout=timeout if timeout else None)
                except queue.Empty:
                    with conn.lock:
                        conn.pending.pop(rid, None)
                    reason = f"{op} timed out after {timeout:.0f} s"
                    self._mark_dead(conn, reason)
                    raise ServiceError(f"{self.name}.{reason}; the worker was killed and will be restarted "
                                       "on the next call", service=self.name, op=op) from None
            finally:
                self.busy_op = None
            if reply is _EOF:
                message = self._exit_message(conn, f"during {op}")
                self._mark_dead(conn, message)
                raise ServiceError(message + "\nThe worker will be restarted on the next call.",
                                   service=self.name, op=op)
            if "suspended" in reply:
                self.suspended = bool(reply["suspended"])
            if reply.get("stats") is not None:
                self.last_stats = dict(reply["stats"], op=op)
            if not reply.get("ok"):
                error = reply.get("error") or "unknown error"
                self.last_error = error
                raise ServiceError(f"{self.name}.{op} failed: {error}", service=self.name, op=op,
                                   remote_traceback=reply.get("traceback"))
            return reply.get("result")

    def _placement_request(self, op):
        """Send 'suspend' or 'resume'. If it fails while the worker keeps running (an error reply,
        e.g. out of memory halfway through moving the weights), the worker is killed: its models
        may be split between host and GPU, and the GPU arbiter must be able to count on a failed
        tenant holding no GPU memory. The next request restarts it."""
        try:
            self.request(op, timeout=self.park_timeout, start=False)
        except Exception as e:
            conn = self._conn
            if conn is not None and conn.proc.poll() is None:
                self.log(f"{op} failed; killing the worker so that it frees the GPU "
                         f"(it is restarted on the next call): {e}")
                self._mark_dead(conn, f"{op} failed: {e}")
            raise

    def suspend(self):
        """Park the models in host RAM. Returns False when there was nothing to do
        (worker not running or already suspended). On failure the worker is killed."""
        with self._call_lock:
            if not (self.alive() and self.state == "ready") or self.suspended:
                return False
            self._placement_request("suspend")
            self.suspended = True
            return True

    def resume(self):
        """Move the models back to the GPU. Returns False when there was nothing to do
        (worker not running or not suspended). On failure the worker is killed."""
        with self._call_lock:
            if not (self.alive() and self.state == "ready") or not self.suspended:
                return False
            self._placement_request("resume")
            self.suspended = False
            return True

    def mem(self, reset_peak=False, timeout=60.0, blocking=True):
        """GPU memory statistics of the worker. With blocking=False, returns None instead of
        waiting when another call (a request, a start-up, suspend/resume) holds the worker."""
        if not self._call_lock.acquire(blocking=blocking):
            return None
        try:
            return self.request("mem", {"reset_peak": bool(reset_peak)}, timeout=timeout, start=False)
        finally:
            self._call_lock.release()

    def stop(self, timeout=10.0, close=False):
        """Ask the worker to exit, then kill its process group. Does not wait for a running request."""
        acquired = self._call_lock.acquire(timeout=0.5)
        try:
            conn = self._conn
            if conn is not None and conn.proc.poll() is None and acquired and self.state == "ready":
                rid = self._next_id
                self._next_id += 1
                with conn.lock:
                    conn.pending[rid] = queue.Queue(maxsize=1)  # the reply is not awaited
                try:
                    conn.proc.stdin.write((json.dumps({"id": rid, "op": "shutdown", "args": {}}) + "\n").encode())
                    conn.proc.stdin.flush()
                    conn.proc.wait(timeout=timeout)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            if conn is not None:
                self._close_connection(conn)
            self._conn = None
            self.suspended = False
            self.state = "closed" if close else ("stopped" if self.state != "failed" else "failed")
        finally:
            if acquired:
                self._call_lock.release()

    def status(self):
        """Snapshot for monitoring; never blocks on a running request."""
        return {
            "state": self.state,
            "alive": self.alive(),
            "pid": self.pid if self.alive() else None,
            "busy": self.busy_op,
            "suspended": self.suspended,
            "starts": self.starts,
            "last_error": self.last_error,
            "last_stats": self.last_stats,
            "log": self.log_path,
        }
