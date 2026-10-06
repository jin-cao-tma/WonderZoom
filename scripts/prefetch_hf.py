#!/usr/bin/env python3
"""Plan and download the WonderZoom checkpoints hosted on the Hugging Face Hub.

Normally run through scripts/download_checkpoints.sh, which also fetches the files that are not
on the Hub (see third_party/checksums.sha256). It works with any Python that has
huggingface_hub >= 0.23 (the wz-main environment has it).

Every repository is pinned to the revision in third_party/pins.env and restricted to an explicit
allow-list, so only the files the code loads are downloaded:
  - Gen3C and Step1X-Edit weights go to local directories under the checkpoints dir
    (checkpoints/ or $WZ_CKPT_DIR), in the layout the workers expect;
  - everything else goes to the standard Hugging Face cache ($HF_HOME/hub).

Examples:
    python scripts/prefetch_hf.py --groups core,coz --dry-run
    python scripts/prefetch_hf.py --all
Exit codes: 0 success, 1 error, 2 bad usage, 3 no access to a gated model.
"""

import argparse
import fnmatch
import hashlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS_FILE = os.path.join(ROOT, "third_party", "pins.env")
CHECKSUMS_FILE = os.path.join(ROOT, "third_party", "checksums.sha256")
GROUPS = ("core", "gen3c", "coz", "step1x", "objects", "scenes")
GROUP_HELP = {
    "core": "main-process models (required)",
    "gen3c": "Gen3C camera-move video model (required for camera moves)",
    "coz": "Chain-of-Zoom super-resolution (required for zoom-in; SD3 is gated)",
    "step1x": "Step1X-Edit object insertion (optional)",
    "objects": "GroundedSAM, SD2 inpainting, INR harmonization (optional)",
    "scenes": "released pre-generated scenes for run_render_only.py (optional)",
}

# ---------------------------------------------------------------------------------------
# Manifest. 'key' selects HF_REPO_<key> / HF_REV_<key> in pins.env. 'dest' is 'cache' (HF
# cache), 'ckpt:<subdir>' (checkpoints dir) or 'root:' (repository root). 'allow' lists the
# files to fetch (fnmatch patterns; None = the whole repository). 'unpinned_loader' marks
# repositories that third-party code loads by name only (no revision); for those the cache
# also gets refs/main when the pinned revision is the current main, so that offline mode
# (HF_HUB_OFFLINE=1) resolves them too.
# ---------------------------------------------------------------------------------------
HF_ITEMS = [
    # --- core: models loaded by the main process --------------------------------------
    dict(name="oneformer", group="core", key="ONEFORMER", dest="cache", unpinned_loader=True,
         purpose="sky segmentation (OneFormer ADE20k Swin-L)",
         allow=["config.json", "preprocessor_config.json", "pytorch_model.bin", "merges.txt",
                "vocab.json", "special_tokens_map.json", "tokenizer_config.json"]),
    dict(name="oneformer_demo", group="core", key="ONEFORMER_DEMO", repo_type="dataset", dest="cache",
         unpinned_loader=True, purpose="ADE20k class metadata read by OneFormerProcessor",
         allow=["ade20k_panoptic.json"]),
    dict(name="marigold_normals", group="core", key="MARIGOLD_NORMALS", dest="cache", unpinned_loader=True,
         purpose="normal estimation (Marigold normals v0-1)",
         allow=["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
                "text_encoder/model.safetensors", "tokenizer/*", "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors"]),
    dict(name="geometrycrafter", group="core", key="GEOMETRYCRAFTER", dest="cache", unpinned_loader=True,
         purpose="video depth for camera moves (GeometryCrafter diffusion UNet + point-map VAE)",
         allow=["unet_diff/config.json", "unet_diff/diffusion_pytorch_model.safetensors",
                "point_map_vae/config.json", "point_map_vae/diffusion_pytorch_model.safetensors"]),
    dict(name="svd_xt", group="core", key="SVD_XT", dest="cache", unpinned_loader=True,
         purpose="SVD-xt image encoder and VAE used by the GeometryCrafter pipeline (fp16)",
         allow=["model_index.json", "feature_extractor/preprocessor_config.json",
                "image_encoder/config.json", "image_encoder/model.fp16.safetensors",
                "scheduler/scheduler_config.json", "vae/config.json",
                "vae/diffusion_pytorch_model.fp16.safetensors"]),
    dict(name="moge_vitl", group="core", key="MOGE_VITL", dest="cache",
         purpose="single-image geometry (MoGe ViT-L)", allow=["model.pt"]),
    # --- gen3c: local directories in the layout Gen3cPipeline(checkpoint_dir=...) expects --
    dict(name="gen3c_cosmos_7b", group="gen3c", key="GEN3C", dest="ckpt:gen3c/Gen3C-Cosmos-7B",
         purpose="Gen3C-Cosmos-7B diffusion transformer", allow=["model.pt", "config.json"]),
    dict(name="cosmos_tokenizer", group="gen3c", key="COSMOS_TOKENIZER",
         dest="ckpt:gen3c/Cosmos-Tokenize1-CV8x8x8-720p", purpose="Cosmos video tokenizer", allow=None,
         require=["encoder.jit", "decoder.jit", "mean_std.pt", "image_mean_std.pt"]),
    dict(name="t5_11b", group="gen3c", key="T5_11B", dest="ckpt:gen3c/google-t5/t5-11b",
         purpose="T5-11B text encoder (no tf_model.h5)",
         allow=["config.json", "pytorch_model.bin", "spiece.model", "tokenizer.json"]),
    # --- coz -------------------------------------------------------------------------------
    dict(name="sd3_medium", group="coz", key="SD3_MEDIUM", dest="cache", gated=True, unpinned_loader=True,
         purpose="Stable Diffusion 3 Medium, Chain-of-Zoom SR backbone (gated)",
         allow=["model_index.json", "scheduler/scheduler_config.json",
                "text_encoder/config.json", "text_encoder/model.safetensors",
                "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
                "text_encoder_3/config.json", "text_encoder_3/model.safetensors.index.json",
                "text_encoder_3/model-0000?-of-00002.safetensors",
                "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
                "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
                "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
    dict(name="qwen25_vl_3b", group="coz", key="QWEN25_VL_3B", dest="cache", unpinned_loader=True,
         purpose="Qwen2.5-VL-3B prompt model for Chain-of-Zoom",
         allow=["*.json", "*.safetensors", "merges.txt"]),
    # --- step1x ----------------------------------------------------------------------------
    dict(name="step1x_edit", group="step1x", key="STEP1X_EDIT", dest="ckpt:step1x",
         purpose="Step1X-Edit v1.0 DiT + FLUX VAE (not the v1.1 weights or the LoRA)",
         allow=["step1x-edit-i1258.safetensors", "vae.safetensors"]),
    dict(name="qwen25_vl_7b", group="step1x", key="QWEN25_VL_7B", dest="cache", unpinned_loader=True,
         purpose="Qwen2.5-VL-7B conditioner for Step1X-Edit",
         allow=["*.json", "*.safetensors", "merges.txt"]),
    # --- objects ---------------------------------------------------------------------------
    dict(name="bert_base_uncased", group="objects", key="BERT_BASE_UNCASED", dest="cache", unpinned_loader=True,
         purpose="BERT text encoder of GroundingDINO",
         allow=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]),
    dict(name="sd2_inpainting", group="objects", key="SD2_INPAINT", dest="cache", unpinned_loader=True,
         purpose="Stable Diffusion 2 inpainting for background plates (fp16 variant)",
         allow=["model_index.json", "feature_extractor/preprocessor_config.json",
                "scheduler/scheduler_config.json", "text_encoder/config.json",
                "text_encoder/model.fp16.safetensors", "tokenizer/*", "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
                "vae/diffusion_pytorch_model.fp16.safetensors"]),
    # --- scenes ----------------------------------------------------------------------------
    dict(name="released_scenes", group="scenes", key="SCENES", repo_type="dataset", dest="root:",
         purpose="released scenes for run_render_only.py, saved to gaussian/",
         allow=["gaussian/*.pth"]),
]

GATED_HELP = """\
No access to the gated model {repo} ({why}).
It is needed by Chain-of-Zoom (--coz). To get access:
  1. Log in on https://huggingface.co, open https://huggingface.co/{repo}
     and accept the license (Stability AI Non-Commercial Research Community License:
     non-commercial use only; read its terms).
  2. Give this machine a token with read access: `huggingface-cli login` (or `hf auth login`),
     or `export HF_TOKEN=hf_...`. Token found on this machine: {token}.
  3. Re-run the download, e.g. `bash scripts/download_checkpoints.sh --coz`.
Use --skip-gated to download everything else without it."""


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------
def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return ("%.0f %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1000.0
    return "%.1f TB" % n


def load_pins(path=PINS_FILE):
    pins = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep or not key.replace("_", "").isalnum():
                raise SystemExit("%s:%d: expected KEY=value" % (path, lineno))
            pins[key] = value
    return pins


def load_checksums(path=CHECKSUMS_FILE):
    """Entries of third_party/checksums.sha256 as dicts (sha256, size, group, url, path)."""
    entries, meta = [], {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                fields = dict(tok.split("=", 1) for tok in line[1:].split() if "=" in tok)
                if fields.get("size", "").isdigit() and fields.get("url"):
                    meta = fields
                continue
            if not line:
                continue
            sha, rel = line.split(None, 1)
            entries.append(dict(sha256=sha, path=rel.lstrip("*"), size=int(meta.get("size", 0)),
                                group=meta.get("group", "core"), url=meta.get("url", "")))
            meta = {}
    return entries


def default_ckpt_dir():
    path = os.environ.get("WZ_CKPT_DIR") or "checkpoints"
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def item_repo(item, pins):
    key = item["key"]
    try:
        return pins["HF_REPO_" + key], pins["HF_REV_" + key]
    except KeyError as e:
        raise SystemExit("third_party/pins.env is missing %s" % e) from None


def item_destination(item, ckpt_dir):
    kind, _, sub = item["dest"].partition(":")
    if kind == "ckpt":
        return os.path.join(ckpt_dir, sub)
    if kind == "root":
        return os.path.join(ROOT, sub) if sub else ROOT
    return None  # HF cache


def show_path(path):
    """Path relative to the repository root when inside it, absolute otherwise."""
    rel = os.path.relpath(path, ROOT)
    return path if rel == os.pardir or rel.startswith(os.pardir + os.sep) else rel


def matches(name, patterns):
    return patterns is None or any(fnmatch.fnmatchcase(name, p) for p in patterns)


def free_bytes(path):
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    st = os.stat(path)
    return st.st_dev, shutil.disk_usage(path).free


def file_digest(path, algo):
    h = hashlib.new(algo)
    if algo == "sha1":  # git blob id of a non-LFS file
        h.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# huggingface_hub access
# ---------------------------------------------------------------------------------------
def import_hub():
    try:
        import huggingface_hub
    except ImportError:
        raise SystemExit(
            "error: huggingface_hub is not installed in %s.\n"
            "Use the main environment's python (scripts/install_env_main.sh) or "
            "`pip install 'huggingface_hub>=0.23'`." % sys.executable
        ) from None
    version = tuple(int(x) for x in huggingface_hub.__version__.split(".")[:2] if x.isdigit())
    if version < (0, 23):
        raise SystemExit("error: huggingface_hub %s is too old; need >= 0.23" % huggingface_hub.__version__)
    return huggingface_hub


def hub_token(hh):
    try:
        return hh.get_token()
    except AttributeError:  # very old versions
        return hh.HfFolder.get_token()


def check_access(hh, repo_id, revision, repo_type, filename):
    """Return (ok, reason) for reading FILENAME of a (possibly gated) repository."""
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    url = hh.hf_hub_url(repo_id, filename, revision=revision, repo_type=repo_type)
    try:
        hh.get_hf_file_metadata(url)
        return True, "access granted"
    except GatedRepoError:
        return False, "license not accepted for this account, or no token"
    except RepositoryNotFoundError:
        return False, "repository not found or token not authorized"
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        return False, "HTTP error %s" % status


def cache_folder(hh, repo_id, repo_type):
    name = "%ss--%s" % (repo_type, repo_id.replace("/", "--"))
    return os.path.join(hh.constants.HF_HUB_CACHE, name)


class Plan(object):
    """Files of one manifest item at its pinned revision, and which of them are missing."""

    def __init__(self, item, repo_id, revision, files, dest_dir):
        self.item = item
        self.repo_id = repo_id
        self.revision = revision
        self.files = files  # list of (rfilename, size, lfs_sha256 or None, blob_id)
        self.dest_dir = dest_dir
        self.missing = []
        self.present = []

    @property
    def repo_type(self):
        return self.item.get("repo_type", "model")

    @property
    def total(self):
        return sum(f[1] for f in self.files)

    @property
    def missing_bytes(self):
        return sum(f[1] for f in self.missing)

    def local_path(self, hh, rfilename):
        if self.dest_dir is not None:
            return os.path.join(self.dest_dir, rfilename)
        found = hh.try_to_load_from_cache(self.repo_id, rfilename, revision=self.revision,
                                          repo_type=self.repo_type)
        return found if isinstance(found, str) else None

    def refresh(self, hh):
        self.missing, self.present = [], []
        for entry in self.files:
            path = self.local_path(hh, entry[0])
            ok = bool(path) and os.path.isfile(path) and os.path.getsize(path) == entry[1]
            (self.present if ok else self.missing).append(entry)


def build_plans(hh, api, items, pins, ckpt_dir):
    plans, errors = [], []
    for item in items:
        repo_id, revision = item_repo(item, pins)
        repo_type = item.get("repo_type", "model")
        try:
            info = api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
        except Exception as e:
            errors.append("%s (%s@%s): cannot list files: %s: %s"
                          % (item["name"], repo_id, revision[:12], type(e).__name__, str(e).splitlines()[0]))
            continue
        if info.sha != revision:
            errors.append("%s: %s resolved to %s, expected the pinned %s" % (item["name"], repo_id, info.sha, revision))
            continue
        files = []
        for s in info.siblings or []:
            if matches(s.rfilename, item.get("allow")):
                lfs = getattr(s, "lfs", None)
                lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
                if isinstance(lfs, dict):
                    lfs_sha = lfs.get("sha256")
                files.append((s.rfilename, int(s.size or 0), lfs_sha, getattr(s, "blob_id", None)))
        allow = (item.get("allow") or []) + (item.get("require") or [])
        for pattern in allow:
            if not any(fnmatch.fnmatchcase(f[0], pattern) for f in files):
                errors.append("%s: no file in %s@%s matches '%s'" % (item["name"], repo_id, revision[:12], pattern))
        plan = Plan(item, repo_id, revision, sorted(files), item_destination(item, ckpt_dir))
        plan.refresh(hh)
        plans.append(plan)
    return plans, errors


def non_hf_entries(groups, ckpt_dir):
    out = []
    for e in load_checksums():
        if e["group"] in groups:
            path = os.path.join(ckpt_dir, e["path"])
            present = os.path.isfile(path) and os.path.getsize(path) == e["size"]
            out.append(dict(e, local=path, present=present))
    return out


def print_plan(hh, plans, extra, groups, ckpt_dir):
    print("Hugging Face cache : %s" % hh.constants.HF_HUB_CACHE)
    print("Checkpoints dir    : %s" % ckpt_dir)
    grand_total = grand_missing = 0
    for group in groups:
        gp = [p for p in plans if p.item["group"] == group]
        ge = [e for e in extra if e["group"] == group]
        if not gp and not ge:
            continue
        total = sum(p.total for p in gp) + sum(e["size"] for e in ge)
        missing = sum(p.missing_bytes for p in gp) + sum(e["size"] for e in ge if not e["present"])
        grand_total += total
        grand_missing += missing
        print("")
        print("[%s] %s: %s total, %s to download" % (group, GROUP_HELP[group], human(total), human(missing)))
        for p in gp:
            if p.dest_dir is None:
                where = "HF cache"
            else:
                sub = os.path.dirname(p.files[0][0]) if p.files else ""
                where = show_path(os.path.join(p.dest_dir, sub)) + "/"
            state = "present" if not p.missing else "%d/%d files missing" % (len(p.missing), len(p.files))
            gated = "  [gated]" if p.item.get("gated") else ""
            print("  %-28s %10s  %s@%s -> %s  (%s)%s"
                  % (p.item["name"], human(p.total), p.repo_id, p.revision[:10], where, state, gated))
        for e in ge:
            src = "Google Drive" if e["url"].startswith("gdrive:") else e["url"].split("/")[2]
            print("  %-28s %10s  %s -> %s  (%s)"
                  % (os.path.basename(e["path"]), human(e["size"]), src,
                     show_path(e["local"]), "present" if e["present"] else "missing"))
    shown = [g for g in groups if any(p.item["group"] == g for p in plans) or any(e["group"] == g for e in extra)]
    print("")
    print("TOTAL: %s for --%s, %s still to download"
          % (human(grand_total), " --".join(shown) if shown else "(nothing)", human(grand_missing)))
    return grand_missing


def check_space(plans, extra, ckpt_dir, hh):
    """Fail early when a target filesystem lacks room for the missing files."""
    need = {}
    for p in plans:
        target = p.dest_dir if p.dest_dir is not None else hh.constants.HF_HUB_CACHE
        dev, free = free_bytes(target)
        n = need.setdefault(dev, [0, free, target])
        n[0] += p.missing_bytes
    for e in extra:
        if not e["present"]:
            dev, free = free_bytes(e["local"])
            n = need.setdefault(dev, [0, free, os.path.dirname(e["local"])])
            n[0] += e["size"]
    ok = True
    margin = 2 * 1000 ** 3
    for needed, free, target in need.values():
        if needed and needed + margin > free:
            ok = False
            print("error: %s needs %s but only %s is free on its filesystem"
                  % (target, human(needed + margin), human(free)), file=sys.stderr)
    return ok


def write_main_ref(hh, api, plan):
    """Point refs/main at the pinned snapshot when the pin is the current main (see manifest)."""
    try:
        main_sha = api.repo_info(plan.repo_id, repo_type=plan.repo_type, revision="main").sha
    except Exception:
        return
    if main_sha != plan.revision:
        print("  note: %s main is now %s; loaders must request revision=%s to use this snapshot offline"
              % (plan.repo_id, main_sha[:10], plan.revision[:10]))
        return
    ref = os.path.join(cache_folder(hh, plan.repo_id, plan.repo_type), "refs", "main")
    try:
        with open(ref) as f:
            if f.read().strip() == plan.revision:
                return
    except OSError:
        pass
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    with open(ref, "w") as f:
        f.write(plan.revision)


def download(hh, api, plan, max_workers):
    kwargs = dict(repo_id=plan.repo_id, repo_type=plan.repo_type, revision=plan.revision,
                  allow_patterns=[f[0] for f in plan.files], max_workers=max_workers)
    if plan.dest_dir is not None:
        os.makedirs(plan.dest_dir, exist_ok=True)
        kwargs["local_dir"] = plan.dest_dir
    hh.snapshot_download(**kwargs)
    plan.refresh(hh)
    if plan.missing:
        raise RuntimeError("still missing after download: %s" % ", ".join(f[0] for f in plan.missing))
    if plan.dest_dir is None and plan.item.get("unpinned_loader"):
        write_main_ref(hh, api, plan)


def verify(hh, plan):
    bad = []
    for rfilename, _size, lfs_sha, blob_id in plan.files:
        path = plan.local_path(hh, rfilename)
        if lfs_sha:
            ok = file_digest(path, "sha256") == lfs_sha
        elif blob_id:
            ok = file_digest(path, "sha1") == blob_id
        else:
            ok = True
        if not ok:
            bad.append(rfilename)
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--groups", default="", help="comma-separated: " + ",".join(GROUPS))
    parser.add_argument("--all", action="store_true", help="all groups")
    parser.add_argument("--dry-run", action="store_true", help="only print what would be downloaded")
    parser.add_argument("--ckpt-dir", default=None, help="checkpoints dir (default: $WZ_CKPT_DIR or checkpoints/)")
    parser.add_argument("--skip-gated", action="store_true", help="skip gated models without access")
    parser.add_argument("--verify", action="store_true", help="re-hash downloaded files against the Hub")
    parser.add_argument("--no-space-check", action="store_true", help="do not check free disk space")
    parser.add_argument("--max-workers", type=int, default=8, help="parallel downloads per repository")
    parser.add_argument("--only", default="", help="comma-separated item names (e.g. sd3_medium,repvit_sam.pt); "
                        "default: every item of the selected groups")
    args = parser.parse_args(argv)
    try:  # keep stdout and stderr in order when both go to the same pipe or log
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    only = [n for n in args.only.split(",") if n]
    known = {it["name"] for it in HF_ITEMS} | {os.path.basename(e["path"]) for e in load_checksums()}
    if [n for n in only if n not in known]:
        parser.error("unknown --only item(s): %s; known: %s"
                     % (", ".join(n for n in only if n not in known), ", ".join(sorted(known))))
    groups = list(GROUPS) if (args.all or (only and not args.groups)) else [g for g in args.groups.split(",") if g]
    unknown = [g for g in groups if g not in GROUPS]
    if not groups or unknown:
        parser.error("choose --all or --groups from %s%s" % (",".join(GROUPS), (" (unknown: %s)" % unknown) if unknown else ""))
    groups = [g for g in GROUPS if g in groups]
    ckpt_dir = os.path.abspath(args.ckpt_dir or default_ckpt_dir())

    for var in ("TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "DIFFUSERS_CACHE"):
        if os.environ.get(var):
            print("warning: %s=%s is set; libraries may look there instead of the HF cache used here. "
                  "Unset it, or set HF_HOME only." % (var, os.environ[var]), file=sys.stderr)

    hh = import_hub()
    api = hh.HfApi()
    pins = load_pins()
    items = [it for it in HF_ITEMS if it["group"] in groups and (not only or it["name"] in only)]
    plans, errors = build_plans(hh, api, items, pins, ckpt_dir)
    extra = [e for e in non_hf_entries(groups, ckpt_dir) if not only or os.path.basename(e["path"]) in only]
    if errors:
        for e in errors:
            print("error: " + e, file=sys.stderr)
        return 1

    # Gated repositories are checked before anything is downloaded, so a missing license
    # acceptance fails fast instead of after hours of other downloads.
    denied = []
    for plan in plans:
        if plan.item.get("gated") and (plan.missing or args.dry_run):
            ok, why = check_access(hh, plan.repo_id, plan.revision, plan.repo_type, plan.files[0][0])
            print("access check: %s -> %s" % (plan.repo_id, why))
            if not ok:
                denied.append((plan, why))
    print_plan(hh, plans, extra, groups, ckpt_dir)
    token = "yes" if hub_token(hh) else "no"
    for plan, why in denied:
        print("\n" + GATED_HELP.format(repo=plan.repo_id, why=why, token=token) + "\n", file=sys.stderr)
    if args.dry_run:
        print("(dry run: nothing was downloaded)")
        return 0
    if denied and not args.skip_gated:
        return 3
    denied_plans = [p for p, _ in denied]
    plans = [p for p in plans if p not in denied_plans]

    if not args.no_space_check and not check_space(plans, extra, ckpt_dir, hh):
        print("Free some space, point HF_HOME / WZ_CKPT_DIR elsewhere, or pass --no-space-check.", file=sys.stderr)
        return 1

    failures = []
    for plan in plans:
        label = "%s (%s@%s)" % (plan.item["name"], plan.repo_id, plan.revision[:10])
        if not plan.missing:
            print("ok        %s: all %d files present" % (label, len(plan.files)))
            if plan.dest_dir is None and plan.item.get("unpinned_loader"):
                write_main_ref(hh, api, plan)
        else:
            print("download  %s: %d files, %s" % (label, len(plan.missing), human(plan.missing_bytes)))
            try:
                download(hh, api, plan, args.max_workers)
            except Exception as e:
                failures.append("%s: %s: %s" % (label, type(e).__name__, e))
                print("FAILED    %s: %s" % (label, e), file=sys.stderr)
                continue
        if args.verify:
            bad = verify(hh, plan)
            if bad:
                failures.append("%s: hash mismatch for %s" % (label, ", ".join(bad)))
                print("FAILED    %s: hash mismatch for %s (delete them and re-run)" % (label, ", ".join(bad)),
                      file=sys.stderr)
            else:
                print("verified  %s" % label)

    if failures:
        print("\n%d Hugging Face download(s) failed:" % len(failures), file=sys.stderr)
        for f in failures:
            print("  - " + f, file=sys.stderr)
        return 1
    if denied:
        print("skipped gated: %s" % ", ".join(p.repo_id for p in denied_plans))
    print("Hugging Face downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
