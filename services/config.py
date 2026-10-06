"""Loading of config/services.yaml (+ the install-generated config/services.local.yaml).

Precedence, lowest to highest: config/services.yaml, config/services.local.yaml (written by
scripts/register_env.py), environment variables, and the `overrides` argument (CLI flags).

Environment variables:
    WZ_GPU_POLICY       gpu.policy (auto | resident | exclusive)
    WZ_CKPT_DIR         paths.checkpoints_dir
    WZ_EXTERNAL_DIR     paths.external_dir (third-party clones; same variable as the install scripts)
    WZ_GEN3C_PYTHON     services.gen3c.python   (likewise WZ_COZ_PYTHON, WZ_STEP1X_PYTHON)

Relative paths are resolved against the repository root (the directory that contains run.py).
Keys named python, *_dir, *_path, *_checkpoint, *_config or *_file are paths. Keys named *_model
hold a Hugging Face id or a local directory; they are treated as paths only when they start with
'/', './', '../' or '~'.
"""
import os

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_NAMES = ("gen3c", "coz", "step1x")
GPU_POLICIES = ("auto", "resident", "exclusive")

_PATH_SUFFIXES = ("_dir", "_path", "_checkpoint", "_config", "_file")
_PATH_SECTIONS = ("paths", "main", "services", "objects", "main_models")


def _abspath(value, root):
    value = os.path.expanduser(str(value))
    if not os.path.isabs(value):
        value = os.path.join(root, value)
    return os.path.normpath(value)


def _looks_like_path(value):
    return isinstance(value, str) and value.startswith(("/", "./", "../", "~"))


def _resolve_paths(node, root):
    """Recursively make path-valued keys absolute (in place, on a plain container)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                _resolve_paths(value, root)
            elif isinstance(value, str) and value:
                name = str(key)
                if name == "python" or name.endswith(_PATH_SUFFIXES):
                    node[key] = _abspath(value, root)
                elif name.endswith("_model") and _looks_like_path(value):
                    node[key] = _abspath(value, root)
                elif name == "cuda_home" and value != "auto":
                    node[key] = _abspath(value, root)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                _resolve_paths(item, root)


def load_services_config(main_path="config/services.yaml", local_path="config/services.local.yaml",
                         overrides=None):
    """Load the services configuration and return an OmegaConf DictConfig with absolute paths.

    Args:
        main_path: services.yaml (relative paths are relative to the repository root).
        local_path: optional install-generated file merged on top (ignored when missing or None).
        overrides: dict of CLI overrides. Recognized keys:
            policy        -> gpu.policy
            main_gpu      -> gpu.main_device
            no_services   -> when true, disables every service
            any dotted key such as 'services.step1x.offload' -> that key
          None values are ignored.
    """
    root = REPO_ROOT
    main_file = _abspath(main_path, root)
    if not os.path.isfile(main_file):
        raise FileNotFoundError(f"services config not found: {main_file}")
    cfg = OmegaConf.load(main_file)
    if local_path:
        local_file = _abspath(local_path, root)
        if os.path.isfile(local_file):
            local = OmegaConf.load(local_file)
            if local is not None and len(local) > 0:
                cfg = OmegaConf.merge(cfg, local)

    # Overrides are applied before interpolations are resolved, so that keys such as
    # ${paths.checkpoints_dir} or ${services.coz.repo_dir} follow them.
    updates = []
    env = os.environ
    if env.get("WZ_GPU_POLICY"):
        updates.append(("gpu.policy", env["WZ_GPU_POLICY"].strip()))
    if env.get("WZ_CKPT_DIR"):
        updates.append(("paths.checkpoints_dir", env["WZ_CKPT_DIR"]))
    if env.get("WZ_EXTERNAL_DIR"):
        updates.append(("paths.external_dir", env["WZ_EXTERNAL_DIR"]))
    for name in SERVICE_NAMES:
        python = env.get(f"WZ_{name.upper()}_PYTHON")
        if python:
            updates.append((f"services.{name}.python", python))
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "policy":
            updates.append(("gpu.policy", str(value)))
        elif key == "main_gpu":
            updates.append(("gpu.main_device", int(value)))
        elif key == "no_services":
            if value:
                services = cfg.get("services") or {}
                updates.extend((f"services.{name}.enabled", False) for name in services)
        else:
            updates.append((key, value))
    for key, value in updates:
        OmegaConf.update(cfg, key, value, merge=True, force_add=True)

    # Interpolations such as ${oc.env:WZ_CKPT_DIR,checkpoints} are resolved here, once.
    data = OmegaConf.to_container(cfg, resolve=True)
    for section in ("gpu", "paths", "services", "main"):
        if not isinstance(data.get(section), dict):
            data[section] = {}

    policy = str(data["gpu"].get("policy", "auto")).lower()
    if policy not in GPU_POLICIES:
        raise ValueError(f"gpu.policy must be one of {GPU_POLICIES}, got {policy!r}")
    data["gpu"]["policy"] = policy

    for section in _PATH_SECTIONS:
        if isinstance(data.get(section), dict):
            _resolve_paths(data[section], root)
    data["paths"]["repo_root"] = root
    data["paths"]["config_file"] = main_file

    return OmegaConf.create(data)


def service_config(cfg: DictConfig, name: str):
    """The services.<name> section, or None."""
    services = cfg.get("services") or {}
    return services.get(name) if name in services else None
