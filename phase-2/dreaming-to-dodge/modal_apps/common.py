"""Shared Modal objects + helpers for the IRIS-from-scratch world-model lab.

Mirrors rlhf-lab/modal_apps/common.py: one App, one training image (torch +
gymnasium/ale-py + numpy + einops + matplotlib + imageio), one Volume that
persists the replay buffer, checkpoints, dreams and results across every
function and run, and a pile of pure helpers (config loading, replay buffer,
JSON-safe conversion) with NO heavy deps so they import instantly on the client.

Every Modal function returns a plain JSON-able dict (floats/ints/lists/strings)
and commits the Volume before returning, so the local client only needs `modal`.
"""
from __future__ import annotations

import os

import modal

# The from-scratch model code (tokenizer.py, world_model.py, ...) lives in the
# repo ROOT, one level above this modal_apps/ directory. We ship those files
# into the container explicitly (add_local_file) so the Modal functions can
# `from tokenizer import ...` regardless of the client's cwd / sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_MODEL_FILES = ["tokenizer", "world_model", "actor_critic", "imagination", "envs", "controller"]

APP_PREFIX = "iris-wm"
app = modal.App(APP_PREFIX)

# ----------------------------------------------------------------------------
# Volume (data / checkpoints / dreams / results) + shared torch cache
# ----------------------------------------------------------------------------
VOL = modal.Volume.from_name("iris-wm-vol", create_if_missing=True)
CACHE = modal.Volume.from_name("iris-wm-cache", create_if_missing=True)
VOL_PATH = "/vol"
CACHE_PATH = "/root/.cache/torch"
VOLUMES = {VOL_PATH: VOL, CACHE_PATH: CACHE}

# ----------------------------------------------------------------------------
# Image.  ale-py is baked in (with ROMs) so the fallback Pong env is ready, but
# nothing forces its use -- Catch is pure numpy.
# ----------------------------------------------------------------------------
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    # runtime libs the vizdoom wheel links against (headless: no X server needed
    # as long as the game window is invisible and SDL uses the dummy driver).
    .apt_install(
        "libsdl2-2.0-0",
        "libopenal1",
        "libgl1-mesa-glx",
        "libgl1-mesa-dri",   # software (llvmpipe) GL so rendering works with no GPU display
        "libglu1-mesa",
        "libgomp1",
        "libbz2-1.0",
        "libjpeg62-turbo",
        "libpng16-16",
        "libfluidsynth3",
        "xvfb",              # virtual framebuffer: gives VizDoom's GL renderer a display
    )
    .pip_install(
        "torch==2.6.0",
        # numpy MUST be <2 for vizdoom 1.2.3: under numpy>=2 its screen-buffer
        # readback returns a uniform-gray image (ViZDoom issue #589). 1.26.4 is
        # compatible with torch 2.6 / matplotlib / gymnasium and keeps the same
        # PCG64 RNG stream, so Catch/epidemic determinism is unaffected.
        "numpy==1.26.4",
        "einops==0.8.0",
        "matplotlib==3.9.2",
        "imageio==2.36.0",
        "pyyaml==6.0.2",
        "gymnasium==1.0.0",
        "ale-py==0.10.1",
        "vizdoom==1.2.3",
    )
    .pip_install("cma==3.3.0")   # separate layer (keeps the heavy torch layer cached)
    # audio off (no sound card); force SOFTWARE GL (llvmpipe) so VizDoom renders
    # real pixels into the Xvfb display without needing a GPU-attached X server.
    # NB: do NOT set SDL_VIDEODRIVER=dummy -- that yields blank (uniform) frames.
    .env({"TOKENIZERS_PARALLELISM": "false", "SDL_AUDIODRIVER": "dummy",
          "LIBGL_ALWAYS_SOFTWARE": "1"})
)

# ship our from-scratch model code (repo root) + this helper module into the
# container, mounting each file at /root so it is importable as a top-level
# module.  add_local_file(copy=True) so the files are baked into the image
# layer (needed because functions import them at module load).
for _m in _MODEL_FILES:
    IMAGE = IMAGE.add_local_file(
        os.path.join(_ROOT, f"{_m}.py"), f"/root/{_m}.py", copy=True)
IMAGE = IMAGE.add_local_file(
    os.path.join(_HERE, "common.py"), "/root/common.py", copy=True)

GPU_SMALL = "A10G"     # tokenizer / world model / policy (the whole pipeline)
GPU_CHEAP = "L4"       # smoke test
GPU_MED = "A100-40GB"  # stretch: Pong

# ----------------------------------------------------------------------------
# Paths on the Volume
# ----------------------------------------------------------------------------
def data_path(tag):    return f"{VOL_PATH}/data/{tag}.npz"
def ckpt_path(tag, k): return f"{VOL_PATH}/ckpt/{tag}/{k}.pt"
def results_dir():     return f"{VOL_PATH}/results"
def dreams_dir():      return f"{VOL_PATH}/dreams"


# ----------------------------------------------------------------------------
# Config loading (pure) -- the YAML lives on the client and is passed in as a
# dict, so functions never need the file on the Volume.
# ----------------------------------------------------------------------------
def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# Replay buffer stored as a single .npz on the Volume (uint8 frames to save
# space). Pure numpy so it imports on the client too.
# ----------------------------------------------------------------------------
def save_buffer(path, frames, actions, rewards, dones):
    import os, numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path,
                        frames=(frames * 255).astype("uint8"),
                        actions=actions.astype("int16"),
                        rewards=rewards.astype("float32"),
                        dones=dones.astype("uint8"))


def load_buffer(path):
    import numpy as np
    d = np.load(path)
    return (d["frames"].astype("float32") / 255.0,
            d["actions"].astype("int64"),
            d["rewards"].astype("float32"),
            d["dones"].astype("int64"))


def append_buffer(path, frames, actions, rewards, dones, capacity):
    """Append a new round of experience, trimming to `capacity` (FIFO)."""
    import os, numpy as np
    if os.path.exists(path):
        of, oa, orw, od = load_buffer(path)
        frames = np.concatenate([of, frames], 0)
        actions = np.concatenate([oa, actions], 0)
        rewards = np.concatenate([orw, rewards], 0)
        dones = np.concatenate([od, dones], 0)
    if len(frames) > capacity:
        frames, actions, rewards, dones = (x[-capacity:] for x in
                                           (frames, actions, rewards, dones))
    save_buffer(path, frames, actions, rewards, dones)
    return len(frames)


# ----------------------------------------------------------------------------
# JSON hygiene: convert torch/numpy scalars to python floats recursively.
# ----------------------------------------------------------------------------
def jsonify(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return jsonify(obj.detach().cpu().tolist())
    except Exception:
        pass
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def write_result(name, obj):
    import os, json
    os.makedirs(results_dir(), exist_ok=True)
    p = f"{results_dir()}/{name}.json"
    with open(p, "w") as f:
        json.dump(jsonify(obj), f, indent=2)
    return p
