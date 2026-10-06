#!/usr/bin/env python3
"""Register the Python interpreter of a WonderZoom environment.

The install scripts call this as their last step. It records absolute interpreter paths in
config/services.local.yaml (machine-specific, gitignored), which services.load_services_config()
merges over config/services.yaml. The file has this shape:

    main:
      python: "/abs/path/to/envs/wz-main/bin/python"
    services:
      gen3c:
        python: "/abs/path/to/envs/wz-gen3c/bin/python"
      coz:
        python: "/abs/path/to/envs/wz-coz/bin/python"
      step1x:
        python: "/abs/path/to/envs/wz-step1x/bin/python"

Usage:
    python scripts/register_env.py main  /path/to/envs/wz-main/bin/python
    python scripts/register_env.py gen3c "$CONDA_PREFIX/bin/python"
    python scripts/register_env.py --get main         # print the effective interpreter (exit 1 if none)
    python scripts/register_env.py --show             # print all registrations
    python scripts/register_env.py --unregister coz

--get honours the environment overrides WZ_MAIN_PYTHON, WZ_GEN3C_PYTHON, WZ_COZ_PYTHON and
WZ_STEP1X_PYTHON, like the services config loader does.

Standard library only, so any Python 3 can run it. Other existing keys in the file are kept.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE = os.path.join(ROOT, "config", "services.local.yaml")
NAMES = ("main", "gen3c", "coz", "step1x")
SERVICE_NAMES = ("gen3c", "coz", "step1x")
HEADER = (
    "# Machine-specific WonderZoom settings, merged over config/services.yaml.\n"
    "# Written by scripts/register_env.py; do not commit (gitignored).\n"
)


# ---------------------------------------------------------------------------------------
# Minimal YAML reading/writing for the simple nested mappings used in this file
# ---------------------------------------------------------------------------------------
def _strip_comment(line):
    """Remove a trailing '# comment' that is not inside quotes."""
    quote = None
    escaped = False
    for i, ch in enumerate(line):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _parse_scalar(text):
    text = text.strip()
    if text in ("", "~", "null", "Null", "NULL"):
        return None
    if text in ("true", "True", "TRUE"):
        return True
    if text in ("false", "False", "FALSE"):
        return False
    if text == "{}":
        return {}
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        return json.loads(text)
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        return text[1:-1].replace("''", "'")
    if text[0] in "[{&*!|>":
        raise ValueError("unsupported YAML value: %r" % text)
    if re.fullmatch(r"[-+]?[0-9]+", text):
        return int(text)
    if re.fullmatch(r"[-+]?([0-9]+\.[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?", text):
        return float(text)
    return text


def _parse_simple_yaml(text):
    root = {}
    stack = [(-1, root)]
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw).rstrip()
        if not line.strip() or line.strip() in ("---", "..."):
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise ValueError("line %d: tabs are not allowed for indentation" % lineno)
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if body.startswith("- "):
            raise ValueError("line %d: lists are not supported by the fallback parser" % lineno)
        key, sep, value = body.partition(":")
        if not sep:
            raise ValueError("line %d: expected 'key: value'" % lineno)
        key = _parse_scalar(key)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError("line %d: bad indentation" % lineno)
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError("line %d: bad nesting" % lineno)
        if value.strip() == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_registry(path=DEFAULT_FILE):
    """Return the parsed services.local.yaml as a dict ({} if the file does not exist)."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # PyYAML (present in every WonderZoom env) handles anything a user may write
    except ImportError:
        yaml = None
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        try:
            data = _parse_simple_yaml(text)
        except ValueError as e:
            raise SystemExit(
                "error: cannot parse %s without PyYAML (%s); fix the file by hand or delete it." % (path, e)
            ) from None
    if not isinstance(data, dict):
        raise SystemExit("error: %s must contain a mapping at the top level" % path)
    return data


def _emit_scalar(value):
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    # Lists and other values: JSON flow style is valid YAML.
    return json.dumps(value)


def _emit(data, indent=0):
    lines = []
    for key, value in data.items():
        pad = " " * indent
        if isinstance(value, dict) and value:
            lines.append("%s%s:" % (pad, key))
            lines.extend(_emit(value, indent + 2))
        elif isinstance(value, dict):
            lines.append("%s%s: {}" % (pad, key))
        else:
            lines.append("%s%s: %s" % (pad, key, _emit_scalar(value)))
    return lines


def _prune(data):
    """Remove empty mappings (e.g. a service whose interpreter was unregistered)."""
    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value = _prune(value)
            if not value:
                continue
        out[key] = value
    return out


def save_registry(data, path=DEFAULT_FILE):
    data = _prune(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = HEADER + "\n".join(_emit(data)) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".services.local.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------------------
# Registry helpers (also imported by scripts/check_install.py)
# ---------------------------------------------------------------------------------------
def _section(data, name):
    if name == "main":
        return data.get("main") if isinstance(data.get("main"), dict) else None
    services = data.get("services")
    if isinstance(services, dict) and isinstance(services.get(name), dict):
        return services[name]
    return None


def registered_python(name, path=DEFAULT_FILE, use_env=True):
    """Effective interpreter for 'main' | 'gen3c' | 'coz' | 'step1x', or None."""
    if name not in NAMES:
        raise ValueError("unknown environment %r (expected one of %s)" % (name, ", ".join(NAMES)))
    if use_env:
        override = os.environ.get("WZ_%s_PYTHON" % name.upper())
        if override:
            return override
    section = _section(load_registry(path), name)
    if section and section.get("python"):
        return str(section["python"])
    return None


def register(name, python, path=DEFAULT_FILE):
    data = load_registry(path)
    if name == "main":
        section = data.get("main")
        if not isinstance(section, dict):
            section = data["main"] = {}
    else:
        services = data.get("services")
        if not isinstance(services, dict):
            services = data["services"] = {}
        section = services.get(name)
        if not isinstance(section, dict):
            section = services[name] = {}
    section["python"] = python
    # Keep a stable key order: main, services, then anything else.
    ordered = {}
    for key in ("main", "services"):
        if key in data:
            ordered[key] = data[key]
    for key, value in data.items():
        ordered.setdefault(key, value)
    if isinstance(ordered.get("services"), dict):
        services = ordered["services"]
        ordered["services"] = {k: services[k] for k in SERVICE_NAMES if k in services}
        for key, value in services.items():
            ordered["services"].setdefault(key, value)
    save_registry(ordered, path)


def unregister(name, path=DEFAULT_FILE):
    data = load_registry(path)
    section = _section(data, name)
    if not section or "python" not in section:
        return False
    del section["python"]
    save_registry(data, path)
    return True


def _probe(python):
    """Return (major, minor) of the interpreter, or None if it cannot be run."""
    try:
        out = subprocess.run(
            [python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=True,
            universal_newlines=True,
        ).stdout.strip()
        major, minor = out.split(".")
        return int(major), int(minor)
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Register a WonderZoom environment interpreter in config/services.local.yaml."
    )
    parser.add_argument("name", nargs="?", choices=NAMES, help="environment to register")
    parser.add_argument("python", nargs="?", help="path to the environment's python executable")
    parser.add_argument("--get", metavar="NAME", choices=NAMES, help="print the effective interpreter")
    parser.add_argument("--show", action="store_true", help="print all registered interpreters")
    parser.add_argument("--unregister", metavar="NAME", choices=NAMES, help="remove a registration")
    parser.add_argument("--file", default=DEFAULT_FILE, help="registry file (default: %(default)s)")
    parser.add_argument("--no-env", action="store_true", help="--get: ignore WZ_<NAME>_PYTHON overrides")
    parser.add_argument("--no-check", action="store_true", help="do not run the interpreter to validate it")
    args = parser.parse_args(argv)
    path = os.path.abspath(args.file)

    if args.get:
        python = registered_python(args.get, path, use_env=not args.no_env)
        if not python:
            print(
                "error: no '%s' interpreter registered in %s; run scripts/install_env_%s.sh or "
                "'python scripts/register_env.py %s /path/to/bin/python'"
                % (args.get, path, args.get, args.get),
                file=sys.stderr,
            )
            return 1
        print(python)
        return 0

    if args.show:
        for name in NAMES:
            python = registered_python(name, path, use_env=False)
            override = os.environ.get("WZ_%s_PYTHON" % name.upper())
            line = "%-7s %s" % (name, python or "(not registered)")
            if override:
                line += "   [overridden by WZ_%s_PYTHON=%s]" % (name.upper(), override)
            print(line)
        return 0

    if args.unregister:
        if unregister(args.unregister, path):
            print("unregistered %s in %s" % (args.unregister, path))
        else:
            print("%s was not registered in %s" % (args.unregister, path))
        return 0

    if not args.name or not args.python:
        parser.error("expected NAME and PYTHON (or one of --get/--show/--unregister)")

    python = os.path.abspath(os.path.expanduser(args.python))
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        print("error: %s is not an executable file" % python, file=sys.stderr)
        return 1
    if not args.no_check:
        version = _probe(python)
        if version is None:
            print("error: could not run %s" % python, file=sys.stderr)
            return 1
        if version != (3, 10):
            print(
                "warning: %s is Python %d.%d; WonderZoom environments are tested with 3.10"
                % ((python,) + version),
                file=sys.stderr,
            )
    register(args.name, python, path)
    shown = os.path.relpath(path, ROOT)
    if shown.startswith(os.pardir):
        shown = path
    print("registered %s -> %s (in %s)" % (args.name, python, shown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
