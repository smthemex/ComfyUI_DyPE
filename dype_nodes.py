# -*- coding: UTF-8 -*-

import comfy.sample
import comfy.samplers
import comfy.utils
from comfy_api.latest import io
from . import dype_core, dype_sampling


def _set_cond_values(conditioning, values):
    out = []
    for t, d in conditioning:
        d = d.copy()
        d.update(values)
        out.append([t, d])
    return out


class DyPEPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DyPEPatch",
            display_name="DyPE Patch (rope extrapolation)",
            category="DyPE",
            description="Attach DyPE (Dynamic Position Extrapolation) to a MODEL loaded by any stock "
                        "loader (Load Diffusion Model / GGUF loader). Only the RoPE table is replaced - "
                        "weights, VAE and sampler stay 100% native. Inside the training window "
                        "(<= base_resolution) the rope is bit-identical to stock comfy, so the node is "
                        "safe to leave in place at any resolution.\n"
                        "Supports: flux1 (base 1024), qwen-image (base 1328), z-image / lumina2 (base 1024).\n"
                        "Note: DyPE only fixes the *position* extrapolation. The flow shift must also "
                        "follow the official schedule - use DyPESampler, or for flux1 the stock "
                        "'ModelSamplingFlux' node with matching width/height.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("method", options=["yarn", "ntk", "base"], default="yarn",
                               tooltip="yarn = DyPE paper (YaRN bands, recommended) / ntk = NTK-aware "
                                       "theta scaling / base = stock comfy rope (A/B switch)"),
                io.Boolean.Input("dype", default=True,
                                 tooltip="kappa = t^lambda timestep modulation. False = plain static YaRN."),
                io.Int.Input("base_resolution", default=0, min=0, max=8192, step=16,
                             tooltip="Training resolution in pixels (0 = auto: flux 1024, qwen 1328, "
                                     "z-image 1024). Extrapolation starts above this."),
                io.Float.Input("lambda_t", default=2.0, min=0.0, max=8.0, step=0.1,
                               tooltip="DyPE exponent lambda (paper: 2.0)"),
                io.Boolean.Input("mscale", default=True,
                                 tooltip="YaRN magnitude scaling 0.1*ln(scale)+1 on cos/sin"),
                io.Boolean.Input("verbose", default=False),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, method, dype, base_resolution, lambda_t, mscale, verbose) -> io.NodeOutput:
        patched, info = dype_core.apply_dype_patch(
            model, method=method, dype=dype, base_resolution=base_resolution,
            lambda_t=lambda_t, mscale=mscale, verbose=verbose)
        print(f"[DyPE] patched {info['family_name']}: rope {info['attr']} "
              f"theta={info['theta']} axes={info['axes_dim']} "
              f"base={info['base_resolution']}px ({info['base_patches']:g} patches) "
              f"method={method} dype={dype} lambda_t={lambda_t}")
        return io.NodeOutput(patched)


class DyPESampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DyPESampler",
            display_name="DyPE Sampler (official sigma grid)",
            category="DyPE",
            description="Euler sampler driven by the official diffusers sigma grid for the model family, "
                        "with the resolution dependent flow shift computed from the latent itself:\n"
                        "  flux1      dynamic mu (256/4096, base_shift .5 / max_shift 1.15)\n"
                        "  qwen-image dynamic mu (256/8192, .5 / .9) + shift_terminal 0.02\n"
                        "  z-image    static shift 3.0\n"
                        "Uses comfy's native sampler core (sample_custom + euler), so LoRA / patches / "
                        "model management all still apply. For flux1 put the guidance value in "
                        "`guidance` and keep cfg=1.0 (guidance-distilled); for qwen-image use "
                        "`cfg` (true_cfg_scale, 4.0) and connect a negative conditioning.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent_image"),
                io.Int.Input("steps", default=28, min=1, max=200),
                io.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1,
                               tooltip="true CFG scale. flux1 / z-image: keep 1.0. qwen-image: 4.0"),
                io.Float.Input("guidance", default=3.5, min=0.0, max=30.0, step=0.1,
                               tooltip="flux1 guidance embedding (official 3.5). Ignored by other families"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Combo.Input("sampler", options=comfy.samplers.SAMPLER_NAMES, default="euler"),
                io.Combo.Input("family", options=["auto", "flux", "qwen", "zimage"], default="auto",
                               tooltip="Only needed if auto-detection fails"),
                io.Boolean.Input("verbose", default=False),
                io.Conditioning.Input("negative", optional=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, model, positive, latent_image, steps, cfg, guidance, denoise, seed,
                sampler, family, verbose, negative=None) -> io.NodeOutput:
        base_model = model.model
        detected_family, patch_size, _ = dype_core.detect_family(base_model)
        if family == "auto":
            family = detected_family
        # exactly what stock KSampler/common_ksampler does: fix empty-latent channel count
        # and promote 4D -> 5D for latent formats with latent_dimensions == 3 (qwen-image)
        latent = comfy.sample.fix_empty_latent_channels(
            model, latent_image["samples"],
            latent_image.get("downscale_ratio_spacial", None),
            latent_image.get("downscale_ratio_temporal", None))
        tokens = dype_core.token_count(latent, family, patch_size)
        sigmas = dype_sampling.official_sigmas(family, tokens, steps, denoise=denoise)

        # guidance-distilled families (flux1): guidance is an embedding, not CFG
        cond = positive
        cfg_eff = float(cfg)
        unet_config = getattr(base_model.model_config, "unet_config", {}) or {}
        if family == "flux" and unet_config.get("guidance_embed"):
            cond = _set_cond_values(positive, {"guidance": float(guidance)})
            if negative is None:
                cfg_eff = 1.0
        if cfg_eff > 1.0 and negative is None:
            print("[DyPE] cfg > 1 but no negative conditioning connected -> running without CFG")
            cfg_eff = 1.0

        noise = comfy.sample.prepare_noise(latent, seed, latent_image.get("batch_index"))
        neg = negative if negative is not None else []
        noise_mask = latent_image.get("noise_mask", None)

        pbar = comfy.utils.ProgressBar(len(sigmas) - 1)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, len(sigmas) - 1)

        if verbose:
            print(f"[DyPE] {dype_sampling.describe(family, tokens, steps)} | cfg={cfg_eff} "
                  f"guidance={guidance} denoise={denoise} latent={tuple(latent.shape)}")
            print("[DyPE] sigmas: " + ", ".join(f"{float(s):.4f}" for s in sigmas))

        samples = comfy.sample.sample_custom(
            model, noise, cfg_eff, comfy.samplers.sampler_object(sampler), sigmas,
            cond, neg, latent, noise_mask=noise_mask, seed=seed, callback=callback)
        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return io.NodeOutput(out)


