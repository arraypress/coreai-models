# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""FLUX.2 transformer with LoRA matrices as runtime inputs (``--lora-slot RANK``).

Core AI has no adapter API: a compiled asset's weights are fixed, so baking a LoRA in
costs a whole re-export (7.2 GB for Klein 4B) per LoRA. This export wraps every
LoRA-targeted Linear layer as ``W·x + B·(A·x)`` and takes ``A`` (``[rank, in]``) and
``B`` (``[out, rank]``) as ordinary inputs named ``lora_a_<i>`` / ``lora_b_<i>``, the
layers in the order ``slot.json`` lists them. One slot asset then serves any LoRA of
rank ≤ RANK (smaller ranks zero-padded, the LoRA's alpha/rank scale folded into B);
all-zero matrices reproduce the plain export bit for bit, and the runtime fills any
slot input a caller leaves out with zeros (Swift: ``LoRASlot``).

Measured on FLUX.2 Klein 4B, rank 16, 1024², M3 Max: 80 layers, 166 inputs, exported
in 102 s at 16.9 GB peak RAM; same 7.2 GB asset as the plain export; transformer
output bit-identical with zero matrices; a LoRA through the slot vs merged in PyTorch
fp16: relative L2 0.0024 (below the plain Core AI vs PyTorch error, 0.0039); +2 %
per step.

The targeted layers (``TARGETS``) are the attention projections of the double-stream
blocks and the fused projections of the single-stream blocks — the PEFT target list
FLUX.2 LoRA trainers use. Text-to-image at the exported size only: the img2img
entrypoints are not wrapped.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: Suffixes of the Linear layers a slot wraps, matched against ``named_modules`` names.
TARGETS: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_qkv_mlp_proj",
    "attn.to_out",
)


def slot_targets(model: torch.nn.Module) -> list[tuple[str, torch.nn.Linear]]:
    """The Linear layers a slot wraps, in ``named_modules`` order — the inputs' order."""
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and any(name.endswith(t) for t in TARGETS)
    ]


class SlotLinear(torch.nn.Module):
    """``W·x + B·(A·x)``, with A and B read from the wrapper's per-call context."""

    def __init__(self, base: torch.nn.Linear, ctx: list[Any], index: int):
        super().__init__()
        self.base, self.ctx, self.index = base, ctx, index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.ctx[2 * self.index], self.ctx[2 * self.index + 1]
        return self.base(x) + torch.matmul(torch.matmul(x, a.t()), b.t())


class SlotTransformer(torch.nn.Module):
    """The FLUX.2 transformer with every target Linear swapped for a :class:`SlotLinear`.

    ``forward`` takes the plain transformer's six inputs followed by A0, B0, A1, B1, …
    """

    def __init__(self, transformer: torch.nn.Module):
        super().__init__()
        self.model = transformer
        layers = slot_targets(transformer)
        self.layers = [(name, mod.in_features, mod.out_features) for name, mod in layers]
        self.ctx: list[Any] = [None] * (2 * len(layers))
        for i, (name, mod) in enumerate(layers):
            parent_name, _, attr = name.rpartition(".")
            parent = transformer.get_submodule(parent_name) if parent_name else transformer
            setattr(parent, attr, SlotLinear(mod, self.ctx, i))

    def forward(self, hidden_states, encoder_hidden_states, timestep, guidance, img_ids, txt_ids, *lora):
        for i, tensor in enumerate(lora):
            self.ctx[i] = tensor
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

    def layout(self, rank: int) -> dict[str, Any]:
        """What ``slot.json`` holds: the rank and every layer's name and shape, in input order."""
        return {"rank": rank, "layers": [{"name": n, "in": i, "out": o} for n, i, o in self.layers]}


def slot_input_names(count: int) -> tuple[str, ...]:
    """``lora_a_0``, ``lora_b_0``, ``lora_a_1``, … for ``count`` layers."""
    return tuple(name for i in range(count) for name in (f"lora_a_{i}", f"lora_b_{i}"))


def export_lora_slot(
    hf_model_id: str,
    output_dir: str | Path,
    rank: int,
    compute_precision: str = "float16",
    include_debug_info: bool = False,
) -> Path:
    """Export the text-to-image transformer of ``hf_model_id`` with a rank-``rank`` slot.

    Writes ``Transformer.aimodel`` and ``slot.json`` into ``output_dir`` and returns the
    asset's path. Only the transformer is loaded (the full pipeline would also load the
    8 GB text encoder); the text encoder, VAEs and tokenizer of a plain export are shared.
    """
    from types import SimpleNamespace

    from diffusers import Flux2Transformer2DModel

    from coreai_models.diffusion.flux2 import dummy_flux2_transformer
    from coreai_models.diffusion.gpu import export_stateless
    from coreai_models.export.metadata import build_aimodel_metadata

    if rank < 1:
        raise ValueError(f"--lora-slot rank must be at least 1, got {rank}")
    dtype = getattr(torch, compute_precision)
    started = time.time()
    transformer = Flux2Transformer2DModel.from_pretrained(hf_model_id, subfolder="transformer", torch_dtype=dtype)
    wrapper = SlotTransformer(transformer)
    base_inputs = dummy_flux2_transformer(SimpleNamespace(transformer=transformer))
    # Small random matrices so tracing sees real data flow through every slot.
    generator = torch.Generator().manual_seed(0)
    extra: list[torch.Tensor] = []
    for _, fan_in, fan_out in wrapper.layers:
        extra.append((torch.randn(rank, fan_in, generator=generator) * 1e-3).to(dtype))
        extra.append((torch.randn(fan_out, rank, generator=generator) * 1e-3).to(dtype))
    input_names = (
        "hidden_states",
        "encoder_hidden_states",
        "timestep",
        "guidance",
        "img_ids",
        "txt_ids",
        *slot_input_names(len(wrapper.layers)),
    )
    logger.info(f"{len(wrapper.layers)} slot layers, {len(input_names)} inputs, rank {rank}")
    program = export_stateless(
        wrapper, tuple(base_inputs) + tuple(extra), input_names, ("output",), include_debug_info=include_debug_info
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset = out / "Transformer.aimodel"
    # save_asset needs a Path: a str fails only after the whole conversion.
    program.save_asset(asset, build_aimodel_metadata(hf_model_id, component="Transformer"))
    (out / "slot.json").write_text(json.dumps(wrapper.layout(rank), indent=1))
    logger.info(f"LoRA-slot transformer exported in {time.time() - started:.0f}s: {asset}")
    return asset
