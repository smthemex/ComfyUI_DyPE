# -*- coding: UTF-8 -*-
"""ComfyUI_DyPE - DyPE (Dynamic Position Extrapolation) on the native comfy DiT stack.

The legacy diffusers-wrapper nodes (DyPE_node.py / model_loader_utils.py plus the
bundled `DyPE/`, `qwen/`, `zimage/`, `Flux/`, `flux2/` pipelines) are kept for
reference only and are NOT imported: they die on transformers>=5 (peft HybridCache)
and are replaced by dype_core.py + dype_sampling.py + dype_nodes.py, which need
nothing but torch/numpy.

Nodes:
    DyPEPatch     MODEL -> MODEL          DyPE rope (yarn/ntk) + timestep propagation
    DyPESampler   LATENT -> LATENT        official diffusers sigma grid + comfy euler
"""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .dype_nodes import  DyPEPatch, DyPESampler


class DyPE_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DyPEPatch, DyPESampler]


async def comfy_entrypoint() -> DyPE_Extension:
    return DyPE_Extension()
