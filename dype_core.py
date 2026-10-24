# -*- coding: UTF-8 -*-
"""DyPE (Dynamic Position Extrapolation) on the **native ComfyUI DiT stack**.

The only structural difference between DyPE and stock inference is the RoPE
frequency table: when an image axis runs past the training window, the
frequencies are re-mixed (YaRN beta/gamma bands) and the mixing bands are
widened by kappa = t^lambda (t = normalised timestep, 1 = pure noise).
Nothing in the transformer weights, the VAE or the sampler changes.

So the whole port is:

    comfy native loader  ->  replace `diffusion_model.pe_embedder`
                          ->  feed the DiT its current sigma (model_function_wrapper)
                          ->  native euler + official sigma grid
                          ->  native VAEDecode

No diffusers, no transformers, no peft -> immune to upstream version drift.

Reference implementation: https://github.com/guyyariv/DyPE
  * `DyPE/flux/transformer_flux.py::get_1d_rotary_pos_embed`  (flux, yarn/ntk)
  * `qwen/transformer_qwenimage.py::QwenEmbedRope.rope_params_yarn` (qwen image)

Axis convention (identical for the 3 supported families):
    axis 0 = text/frame axis  -> never extrapolated
    axis 1 = height, axis 2 = width -> extrapolated when span > base_patches
`base_patches` = training resolution / 16 px per token
    flux1      1024/16 = 64
    qwen-image 1328/16 = 83
    z-image    1024/16 = 64
"""

import math

import torch

import comfy.model_management

# --------------------------------------------------------------------------------------
# DyPE math (ported verbatim from the reference implementation)
# --------------------------------------------------------------------------------------


def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    low = math.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = math.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_mask(min_val, max_val, dim, dtype=torch.float32):
    if min_val == max_val:
        max_val += 0.001  # prevent singularity
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1).to(dtype)


def find_newbase_ntk(dim, base, scale):
    return base * (scale ** (dim / (dim - 2)))


def dype_freqs(dim, theta, scale, ori_max_pe_len, kappa=1.0, dtype=torch.float64, device=None):
    """DyPE / YaRN frequency vector for ONE axis.

    Returns (freqs[dim//2], mscale) - the per-token phase is `pos * freqs`.
    `scale` = current_span / base_patches (> 1). kappa = t^lambda widens both
    correction bands; kappa == 1 reproduces plain (static) YaRN.
    """
    arange = torch.arange(0, dim, 2, dtype=dtype, device=device)
    expo = arange / dim
    theta_t = torch.tensor(theta, dtype=dtype, device=device)
    scale_t = torch.tensor(float(scale), dtype=dtype, device=device)

    freqs_base = 1.0 / (theta_t ** expo)
    freqs_linear = 1.0 / torch.einsum("..., f -> ... f", scale_t, (theta_t ** expo))
    new_base = find_newbase_ntk(dim, theta, scale_t)
    if new_base.dim() > 0:
        new_base = new_base.view(-1, 1)
    freqs_ntk = 1.0 / torch.pow(new_base, expo)
    if freqs_ntk.dim() > 1:
        freqs_ntk = freqs_ntk.squeeze()

    beta_0, beta_1 = 1.25 * kappa, 0.75 * kappa
    low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    mask = 1.0 - linear_ramp_mask(low, high, dim // 2, dtype=dtype).to(device)
    freqs = freqs_linear * (1 - mask) + freqs_ntk * mask

    gamma_0, gamma_1 = 16.0 * kappa, 2.0 * kappa
    low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    mask = 1.0 - linear_ramp_mask(low, high, dim // 2, dtype=dtype).to(device)
    freqs = freqs * (1 - mask) + freqs_base * mask

    mscale = 0.1 * math.log(float(scale)) + 1.0
    return freqs, mscale


def _stock_freqs(dim, theta, dtype, device):
    """Exactly comfy's `comfy.ldm.flux.math.rope` frequency table."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=dtype, device=device)
    return 1.0 / (theta ** scale)


def _to_rope_matrix(phase, dim, mscale=1.0):
    """(b, n, dim/2) phase -> (b, n, dim/2, 2, 2) rotation matrix (comfy layout).

    `mscale` is the YaRN magnitude correction; it multiplies cos/sin (NOT the
    phase - that would shift the frequency). Upstream order: float64 phase ->
    float32 cos/sin -> multiply mscale."""
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    if mscale != 1.0:
        cos = cos.to(torch.float32) * mscale
        sin = sin.to(torch.float32) * mscale
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    return out.reshape(*phase.shape[:-1], dim // 2, 2, 2)


# --------------------------------------------------------------------------------------
# Native-ConditionalContext: timestep + image grid in, comfy-compatible rope out
# --------------------------------------------------------------------------------------


class DyPEEmbedND(torch.nn.Module):
    """Drop-in replacement for `comfy.ldm.flux.layers.EmbedND` with DyPE.

    * No parameters / no buffers -> swapping it never touches a state dict and is
      invisible to comfy's weight management.
    * Output layout is byte-compatible with `EmbedND`: (b, 1, n, dim/2, 2, 2).
    * Inside the training window (span <= base_patches) the code path is comfy's
      own `rope()` math, so <=base resolution output is bit-identical to stock.

    Per-forward context (`set_context`), fed by the model_function_wrapper:
        spans  -> (span_axis0, span_h, span_w) in patch units
        sigma  -> normalised timestep, t in [0, 1] (1 = pure noise)
    """

    def __init__(self, theta=10000.0, axes_dim=(16, 56, 56), base_patches=None,
                 method="yarn", dype=True, lambda_t=2.0, mscale=True, verbose=False,
                 span_mode="always"):
        super().__init__()
        self.theta = float(theta)
        self.axes_dim = [int(d) for d in axes_dim]
        n_img_axes = max(0, len(self.axes_dim) - 1)
        if base_patches is None:
            base_patches = (64.0,) * n_img_axes
        elif isinstance(base_patches, (int, float)):
            base_patches = (float(base_patches),) * n_img_axes
        self.base_patches = tuple(float(b) for b in base_patches)
        self.method = method if method in ("yarn", "ntk", "base") else "yarn"
        self.dype = bool(dype) and self.method != "base"
        self.lambda_t = float(lambda_t)
        self.use_mscale = bool(mscale)
        self.verbose = bool(verbose)
        # "always": the grid from the latent is authoritative (flux / qwen, whose
        #           text ids sit outside the image range and would corrupt a
        #           max-min heuristic)
        # "guarded": only when the axis really spans several positions - z-image
        #           calls the rope 3x per forward (cap / siglip / image) and the
        #           constant-position streams must keep span 1
        self.span_mode = span_mode

        self.current_timestep = 1.0
        self.spans = None
        self._freq_cache = {}
        self._stock_cache = {}
        self.last_debug = {}
        self.trace = []

    # -- context -----------------------------------------------------------------
    def set_context(self, spans=None, sigma=None):
        if sigma is not None:
            self.current_timestep = float(sigma)
        if spans is not None:
            self.spans = tuple(float(s) for s in spans)

    def _span(self, i, pos):
        ids_span = float(pos[..., i].max() - pos[..., i].min()) + 1.0
        if self.spans is not None and i < len(self.spans):
            declared = self.spans[i]
            if declared and declared > 0:
                if self.span_mode == "always":
                    return declared
                if ids_span > 1:  # this axis is a real grid (not a constant stream)
                    return declared
        return ids_span

    # -- rope --------------------------------------------------------------------
    def _stock(self, i, dim, phase_device, dtype):
        key = (i, dim, dtype)
        f = self._stock_cache.get(key)
        if f is None or f.device != phase_device:
            f = _stock_freqs(dim, self.theta, dtype, phase_device)
            self._stock_cache[key] = f
        return f

    def _yarn(self, i, dim, scale, kappa, phase_device, dtype):
        key = (i, dim, round(scale, 6), round(kappa, 6), dtype)
        hit = self._freq_cache.get(key)
        if hit is None:
            hit = dype_freqs(dim, self.theta, scale, self.base_patches[i - 1], kappa=kappa,
                             dtype=dtype, device=phase_device)
            if len(self._freq_cache) > 512:
                self._freq_cache.clear()
            self._freq_cache[key] = hit
        return hit

    def forward(self, ids):
        n_axes = ids.shape[-1]
        pos = ids.float()
        device = pos.device
        can_fp64 = comfy.model_management.supports_fp64(device)
        phase_device = device if can_fp64 else torch.device("cpu")
        dtype = torch.float64 if can_fp64 else torch.float32

        out = []
        dbg = {"sigma": self.current_timestep, "axes": []}
        for i in range(n_axes):
            dim = self.axes_dim[i]
            span = self._span(i, pos)
            base = self.base_patches[i - 1] if i > 0 else 1.0
            extrapolate = (i > 0) and self.method != "base" and span > base
            p = pos[..., i].to(dtype=torch.float32, device=phase_device)

            if extrapolate:
                kappa = (self.current_timestep ** self.lambda_t) if self.dype else 1.0
                if self.method == "ntk":
                    base_ntk = (span / base) ** (dim / (dim - 2))
                    factor = base_ntk ** (self.lambda_t * kappa) if self.dype else base_ntk
                    factor = max(1.0, float(factor))
                    expo = torch.arange(0, dim, 2, dtype=dtype, device=phase_device) / dim
                    freqs = 1.0 / ((self.theta * factor) ** expo)
                    mscale = 1.0
                else:
                    freqs, mscale = self._yarn(i, dim, span / base, kappa, phase_device, dtype)
                phase = torch.einsum("...n,d->...nd", p, freqs)
                # upstream order: float64 phase -> float32 -> multiply mscale
                mat = _to_rope_matrix(phase, dim, float(mscale) if self.use_mscale else 1.0)
                dbg["axes"].append((i, span, round(span / base, 4), round(kappa, 5), round(float(mscale), 5)))
            else:
                freqs = self._stock(i, dim, phase_device, dtype)
                phase = torch.einsum("...n,d->...nd", p, freqs)
                mat = _to_rope_matrix(phase, dim)
                dbg["axes"].append((i, span, 1.0, 1.0, 1.0))

            out.append(mat.to(dtype=torch.float32, device=device))

        emb = torch.cat(out, dim=-3).unsqueeze(1)
        self.last_debug = dbg
        if self.verbose:
            self.trace.append(dbg)
            if len(self.trace) > 4096:
                del self.trace[:2048]
        return emb


# --------------------------------------------------------------------------------------
# family detection / grid math
# --------------------------------------------------------------------------------------

FAMILY_DEFAULTS = {
    # family: (base_resolution, display name)
    "flux": (1024, "flux1 (dev / krea / schnell)"),
    "qwen": (1328, "qwen-image"),
    "zimage": (1024, "z-image / lumina2"),
}


def detect_family(base_model):
    """-> (family, patch_size, default_base_resolution)"""
    unet_config = getattr(base_model.model_config, "unet_config", {}) or {}
    image_model = str(unet_config.get("image_model", "")).lower()
    dm = base_model.diffusion_model
    patch_size = int(getattr(dm, "patch_size", 2) or 2)
    if "qwen_image" in image_model:
        fam = "qwen"
    elif image_model == "lumina2":
        fam = "zimage"
    elif image_model == "flux":
        fam = "flux"
    elif image_model.startswith("flux"):
        # flux2 / flux3 use a different rope (patch 1, 4-axis ids) and the klein
        # variant needs diffusers-only weights - not supported by this build.
        raise RuntimeError(
            f"[DyPE] {image_model!r} is not supported by this build "
            "(only flux1 / qwen-image / z-image(lumina2))")
    else:
        raise RuntimeError(
            f"[DyPE] unsupported model family image_model={image_model!r}; "
            "supported: flux1 / qwen-image / z-image(lumina2)")
    return fam, patch_size, FAMILY_DEFAULTS[fam][0]


def latent_grid(latent, family, patch_size):
    """(span_t, span_h, span_w) in patch units, mirrored from each DiT's own
    `process_img` (flux / qwen round, z-image floors)."""
    h, w = int(latent.shape[-2]), int(latent.shape[-1])
    if family == "zimage":
        h_tok, w_tok = max(1, h // patch_size), max(1, w // patch_size)
    else:
        h_tok = max(1, (h + patch_size // 2) // patch_size)
        w_tok = max(1, (w + patch_size // 2) // patch_size)
    t_tok = 1
    if latent.ndim == 5 and family == "qwen":
        t_tok = 1  # single image frame; reference latents keep their own t index
    return float(t_tok), float(h_tok), float(w_tok)


def token_count(latent, family, patch_size):
    """image_seq_len used by diffusers' calculate_shift."""
    _, h_tok, w_tok = latent_grid(latent, family, patch_size)
    return int(h_tok * w_tok)


# --------------------------------------------------------------------------------------
# patch injection
# --------------------------------------------------------------------------------------


def _make_wrapper(embedder, family, patch_size, inner=None, verbose=False):
    """model_function_wrapper: feed sigma + image grid to the rope, then run the DiT.

    `inner` keeps a previously installed wrapper (other plugins / TeaCache style
    nodes) alive instead of silently replacing it.
    """
    def dype_model_function_wrapper(model_function, kwargs):
        x = kwargs["input"]
        t = kwargs.get("timestep")
        sigma = None
        if t is not None:
            tt = t if torch.is_tensor(t) else torch.tensor([t])
            sigma = float(tt.reshape(-1)[0].item())
        embedder.set_context(spans=latent_grid(x, family, patch_size), sigma=sigma)
        if inner is not None:
            return inner(model_function, kwargs)
        return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

    return dype_model_function_wrapper


def apply_dype_patch(patcher, method="yarn", dype=True, base_resolution=0,
                     lambda_t=2.0, mscale=True, verbose=False):
    """Attach the DyPE RoPE to any comfy MODEL (stock 'Load Diffusion Model' output).

    Returns (patched_model, info dict). Everything else - loading, sampling,
    VAE, LoRA - stays on the stock comfy path.
    """
    patched = patcher.clone()
    base_model = patched.model
    dm = getattr(base_model, "diffusion_model", None)
    family, patch_size, default_base = detect_family(base_model)

    old = getattr(dm, "pe_embedder", None) or getattr(dm, "rope_embedder", None)
    if old is None:
        raise RuntimeError("[DyPE] model has no rope embedder (pe_embedder/rope_embedder)")
    attr_name = "pe_embedder" if hasattr(dm, "pe_embedder") else "rope_embedder"
    theta = float(getattr(old, "theta", 10000.0))
    axes_dim = list(getattr(old, "axes_dim", []) or [])
    if not axes_dim:
        raise RuntimeError("[DyPE] could not read axes_dim from the rope embedder")

    base_res = int(base_resolution) if base_resolution else default_base
    # 8 px latent * patch_size px per token
    base_patches = max(1.0, base_res / (8.0 * patch_size))

    embedder = DyPEEmbedND(theta=theta, axes_dim=axes_dim,
                           base_patches=(base_patches,) * (len(axes_dim) - 1),
                           method=method, dype=dype, lambda_t=lambda_t,
                           mscale=mscale, verbose=verbose,
                           span_mode="guarded" if family == "zimage" else "always")
    patched.add_object_patch(f"diffusion_model.{attr_name}", embedder)
    prev_wrapper = patched.model_options.get("model_function_wrapper", None)
    patched.set_model_unet_function_wrapper(
        _make_wrapper(embedder, family, patch_size, inner=prev_wrapper, verbose=verbose))

    info = {
        "family": family,
        "family_name": FAMILY_DEFAULTS[family][1],
        "attr": attr_name,
        "theta": theta,
        "axes_dim": axes_dim,
        "base_resolution": base_res,
        "base_patches": base_patches,
        "method": method,
        "dype": bool(dype),
        "lambda_t": lambda_t,
        "embedder": embedder,
    }
    if verbose:
        print(f"[DyPE] {info['family_name']} | rope={attr_name} theta={theta} axes={axes_dim} "
              f"base={base_res}px -> {base_patches:g} patches | method={method} dype={dype} "
              f"lambda_t={lambda_t}")
    return patched, info
