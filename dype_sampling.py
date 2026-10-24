# -*- coding: UTF-8 -*-
"""Official (diffusers) sigma grids, replayed on comfy's native euler.

DyPE does not change the sampler, but the *shift* of the flow schedule is
resolution dependent, and the reference pipelines read it straight from
`scheduler_config.json`:

    flux1      FlowMatchEuler, use_dynamic_shifting=True,
               base_image_seq_len 256 / max_image_seq_len 4096,
               base_shift 0.5 / max_shift 1.15
    qwen-image FlowMatchEuler, use_dynamic_shifting=True,
               base 256 / max 8192, base_shift 0.5 / max_shift 0.9,
               shift_terminal 0.02   (+ terminal stretch after the shift)
    z-image    FlowMatchEuler, use_dynamic_shifting=False, shift 3.0

diffusers:  sigmas = linspace(1, 1/n);  sigma = exp(mu)/(exp(mu)+1/t-1);  + terminal 0
comfy euler: x += (sigma_next - sigma) * (x - denoised)/sigma   == diffusers step  (verified)
"""

import math

import numpy as np
import torch

DEFAULTS = {
    # family: dict for calculate_shift / static shift
    "flux": dict(dynamic=True, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15, shift_terminal=None),
    "qwen": dict(dynamic=True, base_seq=256, max_seq=8192, base_shift=0.5, max_shift=0.9, shift_terminal=0.02),
    "zimage": dict(dynamic=False, shift=3.0, shift_terminal=None),
}


def calculate_shift(seq_len, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15):
    """diffusers.pipelines.flux.pipeline_flux.calculate_shift (== comfy's ModelSamplingFlux node)."""
    m = (max_shift - base_shift) / (max_seq - base_seq)
    b = base_shift - m * base_seq
    return seq_len * m + b


def time_shift(mu, sigma, t):
    """FlowMatchEulerDiscreteScheduler._time_shift_exponential."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def stretch_shift_to_terminal(t, terminal):
    one_minus_z = 1.0 - t
    scale_factor = one_minus_z[-1] / (1.0 - terminal)
    return 1.0 - (one_minus_z / scale_factor)


def official_sigmas(family, tokens, steps, denoise=1.0):
    """Full sigma table (len = steps+1, terminal 0), diffusers-exact."""
    cfg = DEFAULTS[family]
    t = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    if cfg.get("dynamic"):
        mu = calculate_shift(tokens, cfg["base_seq"], cfg["max_seq"], cfg["base_shift"], cfg["max_shift"])
        t = time_shift(mu, 1.0, t)
    else:
        s = cfg["shift"]
        t = s * t / (1.0 + (s - 1.0) * t)
    if cfg.get("shift_terminal"):
        t = stretch_shift_to_terminal(t, cfg["shift_terminal"])
    sig = torch.from_numpy(np.concatenate([t, np.zeros(1, dtype=np.float32)]))
    if denoise < 1.0:
        cut = int(round(steps * (1.0 - float(denoise))))
        cut = max(0, min(cut, steps - 1))
        sig = sig[cut:]
    return sig


def describe(family, tokens, steps):
    cfg = DEFAULTS[family]
    if cfg.get("dynamic"):
        mu = calculate_shift(tokens, cfg["base_seq"], cfg["max_seq"], cfg["base_shift"], cfg["max_shift"])
        extra = f"dynamic mu={mu:.4f} shift={math.exp(mu):.3f}"
    else:
        extra = f"static shift={cfg['shift']}"
    term = cfg.get("shift_terminal")
    if term:
        extra += f" terminal={term}"
    return f"{family}: {tokens} tokens, {steps} steps, {extra}"
