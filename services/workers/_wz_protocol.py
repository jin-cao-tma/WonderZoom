"""JSON-lines protocol shared by the WonderZoom model workers (standard library only).

A worker is launched by services/base.py (WorkerClient) as

    <env python> -u services/workers/<svc>_worker.py --config-json '<json>'

with cwd and PYTHONPATH set to the third-party repository (external/GEN3C, ...). Besides
PYTHONPATH, only this directory is on sys.path, so WonderZoom's own packages (utils/, models/,
scene/) never shadow the third-party code.

Messages are single-line JSON objects:

    worker -> client, once at startup:
        {"event": "ready", "info": {...}}
        {"event": "fatal", "error": "...", "traceback": "..."}
    client -> worker:
        {"id": n, "op": "...", "args": {...}}
    worker -> client:
        {"id": n, "ok": true, "result": ..., "stats": {...}, "suspended": false}
        {"id": n, "ok": false, "error": "...", "traceback": "...", "stats": {...}, "suspended": false}

Common ops: ping, mem, suspend, resume, shutdown. Service ops are methods named op_<name> on the
object returned by the worker's load function.

The protocol uses private duplicates of the original stdin/stdout. File descriptor 1 is pointed at
stderr, so prints (also from C extensions) of the model libraries end up in the worker log instead
of corrupting the protocol, and file descriptor 0 is pointed at /dev/null.
"""
import argparse
import gc
import json
import os
import sys
import threading
import time
import traceback

PROTOCOL_VERSION = 1


def _json_default(obj):
    """Make numpy scalars/arrays, paths and sets JSON-serializable."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    return str(obj)


class Channel:
    """Protocol I/O on private copies of the original stdin/stdout."""

    def __init__(self):
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        # Private, non-inheritable duplicates of the protocol pipes.
        self._out_fd = os.dup(1)
        self._in_fd = os.dup(0)
        # fd 1 -> stderr: library output goes to the worker log.
        os.dup2(2, 1)
        # fd 0 -> /dev/null: a library calling input() gets EOF instead of reading protocol lines.
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        os.close(devnull)
        sys.stdout = sys.stderr
        sys.stdin = open(os.devnull, "r")
        self._out = os.fdopen(self._out_fd, "w", encoding="utf-8", buffering=1)
        self._in = os.fdopen(self._in_fd, "rb")
        self._lock = threading.Lock()

    def send(self, obj):
        data = json.dumps(obj, default=_json_default) + "\n"
        with self._lock:
            try:
                self._out.write(data)
                self._out.flush()
            except (BrokenPipeError, OSError, ValueError):
                # The client is gone: nothing left to serve.
                os._exit(2)

    def recv(self):
        """Return the next request (dict), or None at EOF."""
        while True:
            raw = self._in.readline()
            if not raw:
                return None
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError as e:
                self.send({"id": None, "ok": False, "error": f"invalid JSON request: {e}"})
                continue
            if not isinstance(msg, dict):
                self.send({"id": None, "ok": False, "error": "request must be a JSON object"})
                continue
            return msg


def _start_parent_watchdog(interval=2.0):
    """Exit when the WonderZoom process that started this worker disappears.

    The worker runs in its own session, so it does not receive the parent's signals; without this,
    a killed server would leave a worker holding GPU memory until its current request finishes.
    """
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(interval)
            if os.getppid() != parent:
                os._exit(3)

    threading.Thread(target=watch, name="wz-parent-watchdog", daemon=True).start()


def _torch():
    return sys.modules.get("torch")


def _cuda_ready(torch):
    try:
        return torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized()
    except Exception:
        return False


def cuda_memory_stats():
    """Memory statistics of the worker's GPU in GiB (empty before CUDA is initialized)."""
    torch = _torch()
    if not _cuda_ready(torch):
        return {}
    try:
        gib = float(1 << 30)
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / gib, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / gib, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gib, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / gib, 3),
            "device_used_gb": round((total - free) / gib, 3),
            "device_total_gb": round(total / gib, 3),
        }
    except Exception as e:  # never fail a reply because of statistics
        return {"error": str(e)}


def reset_peak_memory():
    torch = _torch()
    if _cuda_ready(torch):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def empty_cuda_cache():
    """Release cached GPU memory (same order as the original service helpers)."""
    torch = _torch()
    if _cuda_ready(torch):
        torch.cuda.empty_cache()
    gc.collect()


def release_cuda_memory():
    """Drop unreferenced objects first, then return the cached blocks to the driver."""
    gc.collect()
    torch = _torch()
    if _cuda_ready(torch):
        torch.cuda.empty_cache()


def _error_text(exc):
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="WonderZoom model worker")
    parser.add_argument("--config-json", default=None, help="worker configuration as a JSON object")
    parser.add_argument("--config-file", default=None, help="worker configuration as a JSON file")
    parser.add_argument("--dry-import", action="store_true",
                        help="import the model modules, report and exit (no weights are loaded)")
    return parser.parse_args(argv)


def _load_config(args):
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.config_json:
        return json.loads(args.config_json)
    return {}


class _Server:
    def __init__(self, name, channel, service):
        self.name = name
        self.channel = channel
        self.service = service
        self.suspended = False
        self.started = time.time()

    # ---- common ops ----

    def op_ping(self):
        return {"pong": True, "service": self.name, "pid": os.getpid(),
                "uptime_s": round(time.time() - self.started, 1), "suspended": self.suspended}

    def op_mem(self, reset_peak=False):
        stats = cuda_memory_stats()
        if reset_peak:
            reset_peak_memory()
        return stats

    def op_suspend(self):
        if not self.suspended:
            t0 = time.time()
            fn = getattr(self.service, "suspend", None)
            if fn is not None:
                fn()
            release_cuda_memory()
            self.suspended = True
            print(f"[{self.name}] suspended in {time.time() - t0:.1f} s", file=sys.stderr, flush=True)
        return cuda_memory_stats()

    def op_resume(self):
        if self.suspended:
            t0 = time.time()
            fn = getattr(self.service, "resume", None)
            if fn is not None:
                fn()
            self.suspended = False
            print(f"[{self.name}] resumed in {time.time() - t0:.1f} s", file=sys.stderr, flush=True)
        return cuda_memory_stats()

    # ---- loop ----

    def handle(self, msg):
        rid = msg.get("id")
        op = msg.get("op")
        args = msg.get("args") or {}
        if not isinstance(args, dict):
            return {"id": rid, "ok": False, "error": "args must be a JSON object"}

        common = getattr(self, f"op_{op}", None) if isinstance(op, str) else None
        handler = common
        if handler is None and isinstance(op, str):
            handler = getattr(self.service, f"op_{op}", None)
        if handler is None:
            return {"id": rid, "ok": False, "error": f"unknown op {op!r}", "suspended": self.suspended}

        is_service_op = common is None
        t0 = time.time()
        try:
            if is_service_op:
                if self.suspended:
                    # The client resumes a worker before using it; this only guards against stale
                    # state. Inside the try, so that a failed resume becomes an error reply.
                    print(f"[{self.name}] op {op!r} received while suspended: resuming first",
                          file=sys.stderr, flush=True)
                    self.op_resume()
                reset_peak_memory()
                t0 = time.time()
            result = handler(**args)
            reply = {"id": rid, "ok": True, "result": result}
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            reply = {"id": rid, "ok": False, "error": _error_text(e), "traceback": tb}
        if is_service_op:
            stats = cuda_memory_stats()
            stats["seconds"] = round(time.time() - t0, 3)
            reply["stats"] = stats
        reply["suspended"] = self.suspended
        return reply

    def serve(self):
        while True:
            msg = self.channel.recv()
            if msg is None:  # client closed the pipe
                return 0
            if msg.get("op") == "shutdown":
                self.channel.send({"id": msg.get("id"), "ok": True, "result": {"bye": True},
                                   "suspended": self.suspended})
                return 0
            self.channel.send(self.handle(msg))


def run_worker(name, load_fn, dry_import_fn=None, argv=None):
    """Entry point of a worker script.

    Args:
        name: service name ('gen3c', 'coz', 'step1x').
        load_fn: cfg -> service object. The object exposes op_<name>(**args) methods, optional
            suspend()/resume() and optional info() -> dict for the ready message.
        dry_import_fn: cfg -> dict. Imports the model modules without loading weights.
    """
    channel = Channel()  # first: protect the protocol pipe from library output
    try:
        args = _parse_args(sys.argv[1:] if argv is None else argv)
        cfg = _load_config(args)
    except BaseException as e:  # includes argparse's SystemExit
        channel.send({"event": "fatal", "error": f"bad worker arguments: {_error_text(e)}",
                      "traceback": traceback.format_exc()})
        os._exit(1)

    _start_parent_watchdog()

    if args.dry_import:
        t0 = time.time()
        try:
            info = dry_import_fn(cfg) if dry_import_fn is not None else {}
            info = dict(info or {})
            info.setdefault("seconds", round(time.time() - t0, 1))
            info.setdefault("python", sys.version.split()[0])
            channel.send({"event": "dry-import", "ok": True, "service": name, "info": info})
            code = 0
        except BaseException as e:
            channel.send({"event": "fatal", "service": name, "error": _error_text(e),
                          "traceback": traceback.format_exc()})
            code = 1
        sys.stderr.flush()
        os._exit(code)

    print(f"[{name}] worker pid {os.getpid()} starting (cwd {os.getcwd()})", file=sys.stderr, flush=True)
    t0 = time.time()
    try:
        service = load_fn(cfg)
    except BaseException as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        channel.send({"event": "fatal", "service": name, "error": _error_text(e), "traceback": tb})
        sys.stderr.flush()
        os._exit(1)

    info = {"service": name, "pid": os.getpid(), "protocol": PROTOCOL_VERSION,
            "python": sys.version.split()[0], "load_seconds": round(time.time() - t0, 1)}
    try:
        extra = service.info() if hasattr(service, "info") else {}
        info.update(extra or {})
    except Exception as e:
        info["info_error"] = _error_text(e)
    info["mem"] = cuda_memory_stats()
    print(f"[{name}] ready after {info['load_seconds']} s", file=sys.stderr, flush=True)
    channel.send({"event": "ready", "info": info})

    code = 1
    try:
        code = _Server(name, channel, service).serve()
    except BaseException:  # e.g. SystemExit raised inside a handler
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        # Skip interpreter teardown: some CUDA libraries can hang in their exit handlers.
        os._exit(code)
