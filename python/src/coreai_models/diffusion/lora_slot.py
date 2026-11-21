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
    height: int = 1024,
    width: int = 1024,
    img2img: bool = False,
) -> Path:
    """Export one transformer function of ``hf_model_id`` with a rank-``rank`` slot.

    Text to image (``Transformer.aimodel``) or, with ``img2img``, the full-grid img2img
    function (``Transformer_img2img_full.aimodel``: noise tokens plus one reference grid
    the same size), traced at ``height`` × ``width``; writes ``slot.json`` beside it
    (``"img2img": true`` for the editing one) and returns the asset's path. Only the
    transformer is loaded (the full pipeline would also load the 8 GB text encoder); the
    text encoder, VAEs and tokenizer of a plain export are shared.
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
    if (height, width) == (1024, 1024) and not img2img:
        base_inputs = dummy_flux2_transformer(SimpleNamespace(transformer=transformer))
    else:
        base_inputs = _shape_inputs(transformer, height // 16, width // 16, img2img, dtype)
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
    asset = out / ("Transformer_img2img_full.aimodel" if img2img else "Transformer.aimodel")
    # save_asset needs a Path: a str fails only after the whole conversion.
    program.save_asset(asset, build_aimodel_metadata(hf_model_id, component="Transformer"))
    layout = wrapper.layout(rank) | ({"img2img": True} if img2img else {})
    (out / "slot.json").write_text(json.dumps(layout, indent=1))
    logger.info(f"LoRA-slot transformer exported in {time.time() - started:.0f}s: {asset}")
    return asset


def _shape_inputs(transformer: torch.nn.Module, rows: int, cols: int, reference: bool,
                  dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Dummy inputs for one transformer function on a ``rows`` × ``cols`` token grid;
    with ``reference``, a second grid the same size marked by the reference time offset."""
    from coreai_models.diffusion.flux2 import REFERENCE_TOKEN_TIME_OFFSET, _rect_position_ids, _text_position_ids

    cfg = transformer.config
    axes = len(list(cfg.axes_dims_rope))
    ids = _rect_position_ids(rows, cols, axes)
    if reference:
        ids = torch.cat([ids, _rect_position_ids(rows, cols, axes, time_offset=REFERENCE_TOKEN_TIME_OFFSET)])
    return (
        torch.randn(1, ids.shape[0], cfg.in_channels, dtype=dtype),
        torch.randn(1, 512, cfg.joint_attention_dim, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        ids.unsqueeze(0),
        _text_position_ids(512, axes),
    )


# ---------------------------------------------------------------------------
# Several shapes in one asset
# ---------------------------------------------------------------------------


def shape_function_names(height: int, width: int) -> tuple[str, str]:
    """The text-to-image and full-grid img2img entrypoints for one shape.

    1024×1024 keeps the names Apple's pipeline enters by default (``main``,
    ``img2img_full``); any other shape is named by its pixels, height first, so a
    folder's ``metadata.json`` can point the pipeline at it (``transformer_function``,
    ``img2img_function_prefix``).
    """
    if (height, width) == (1024, 1024):
        return "main", "img2img_full"
    return f"t2i_{height}x{width}", f"img2img_{height}x{width}_full"


def _rewrap(wrapper: SlotTransformer) -> None:
    """Fresh :class:`SlotLinear` modules over the same base weights, with a fresh context.

    A second ``torch.export`` through the first trace's slot modules lifts that trace's
    fake tensors as constants (".numpy() is not supported for tensor subclasses" at
    conversion). New wrappers per trace avoid it; the base ``Linear`` weights are the same
    objects, so the asset still holds one copy.
    """
    ctx: list[Any] = [None] * len(wrapper.ctx)
    wrapper.ctx = ctx
    for i, (name, _, _) in enumerate(wrapper.layers):
        parent_name, _, attr = name.rpartition(".")
        parent = wrapper.model.get_submodule(parent_name) if parent_name else wrapper.model
        setattr(parent, attr, SlotLinear(getattr(parent, attr).base, ctx, i))


def export_lora_slot_shapes(
    hf_model_id: str,
    output_dir: str | Path,
    rank: int,
    shapes: list[tuple[int, int]],
    compute_precision: str = "float16",
) -> Path:
    """One transformer asset holding every shape, text-to-image AND img2img, each with the slot.

    For each ``(height, width)`` this traces a text-to-image function and a full-grid
    img2img function (noise tokens plus one reference grid the same size, marked by the
    reference time offset), named by :func:`shape_function_names`. All share one copy of
    the weights: measured on Klein 4B, two shapes made a 7.2 GB asset, the size of one,
    and the 1:1 function's output was bit-identical to the single-shape slot export.

    TRAP, measured: Core AI compiles EVERY function of the asset when it loads it, and its
    cache keeps the compiled weights per function — ≈7.5 GB each for Klein 4B, 116 GB for
    sixteen. The download is shared; the disk is not. For a model this size, export one
    function per asset (:func:`export_lora_slot` with ``height``/``width``/``img2img``)
    and fetch only the ones a caller uses.

    Writes ``TransformerShapes.aimodel`` and ``slot.json`` (which also lists the
    functions and marks img2img as slotted) into ``output_dir``.
    """
    from diffusers import Flux2Transformer2DModel

    import coreai_torch

    from coreai_models.diffusion.flux2 import (
        REFERENCE_TOKEN_TIME_OFFSET,
        _rect_position_ids,
        _text_position_ids,
    )
    from coreai_models.diffusion.gpu import _decomp_empty_permuted
    from coreai_models.export.metadata import build_aimodel_metadata

    if rank < 1:
        raise ValueError(f"--lora-slot rank must be at least 1, got {rank}")
    for height, width in shapes:
        if height % 32 or width % 32:
            raise ValueError(f"{height}x{width}: both sides must be multiples of 32 (several references "
                             "share the grid in quarters)")
    dtype = getattr(torch, compute_precision)
    started = time.time()
    transformer = Flux2Transformer2DModel.from_pretrained(hf_model_id, subfolder="transformer", torch_dtype=dtype)
    wrapper = SlotTransformer(transformer)
    cfg = transformer.config
    axes = len(list(cfg.axes_dims_rope))
    text_len = 512
    generator = torch.Generator().manual_seed(0)
    lora: list[torch.Tensor] = []
    for _, fan_in, fan_out in wrapper.layers:
        lora.append((torch.randn(rank, fan_in, generator=generator) * 1e-3).to(dtype))
        lora.append((torch.randn(fan_out, rank, generator=generator) * 1e-3).to(dtype))

    def inputs(rows: int, cols: int, reference: bool) -> tuple[torch.Tensor, ...]:
        ids = _rect_position_ids(rows, cols, axes)
        if reference:
            ids = torch.cat([ids, _rect_position_ids(rows, cols, axes, time_offset=REFERENCE_TOKEN_TIME_OFFSET)])
        return (
            torch.randn(1, ids.shape[0], cfg.in_channels, dtype=dtype),
            torch.randn(1, text_len, cfg.joint_attention_dim, dtype=dtype),
            torch.tensor([0.5], dtype=dtype),
            torch.tensor([1.0], dtype=dtype),
            ids.unsqueeze(0),
            _text_position_ids(text_len, axes),
        ) + tuple(t.clone() for t in lora)

    input_names = (
        "hidden_states", "encoder_hidden_states", "timestep", "guidance", "img_ids", "txt_ids",
        *slot_input_names(len(wrapper.layers)),
    )
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    table = coreai_torch.get_decomp_table()
    table[torch.ops.aten.empty_permuted.default] = _decomp_empty_permuted
    functions: list[str] = []
    for height, width in shapes:
        for name, reference in zip(shape_function_names(height, width), (False, True)):
            _rewrap(wrapper)
            wrapper.eval()
            with torch.no_grad():
                exported = torch.export.export(wrapper, args=inputs(height // 16, width // 16, reference))
            converter.add_exported_program(
                exported.run_decompositions(table), input_names=input_names,
                output_names=("output",), entrypoint_name=name)
            functions.append(name)
            logger.info(f"traced {name} ({time.time() - started:.0f}s)")
    _rewrap(wrapper)
    program = converter.to_coreai()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset = out / "TransformerShapes.aimodel"
    program.save_asset(asset, build_aimodel_metadata(hf_model_id, component="Transformer"))
    layout = wrapper.layout(rank) | {"img2img": True, "functions": functions}
    (out / "slot.json").write_text(json.dumps(layout, indent=1))
    logger.info(f"{len(functions)}-function LoRA-slot transformer exported in {time.time() - started:.0f}s: {asset}")
    return asset


def export_vae_shape(hf_model_id: str, output_dir: str | Path, height: int, width: int,
                     compute_precision: str = "float16", decoder: bool = True, encoder: bool = True) -> None:
    """The FLUX.2 VAE decoder and/or encoder traced at one ``height`` × ``width``.

    Each shape folder needs its own: Core AI traces them at fixed sizes. Small (≈95 and
    66 MB for Klein 4B), so one per shape costs little.
    """
    from diffusers import AutoencoderKLFlux2

    from coreai_models.diffusion.flux2 import Flux2VAEDecoderWrapper, Flux2VAEEncoderWrapper
    from coreai_models.diffusion.gpu import export_stateless
    from coreai_models.export.metadata import build_aimodel_metadata

    dtype = getattr(torch, compute_precision)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if decoder:
        vae = AutoencoderKLFlux2.from_pretrained(hf_model_id, subfolder="vae", torch_dtype=dtype)
        latent = torch.randn(1, vae.config.latent_channels, height // 8, width // 8, dtype=dtype)
        program = export_stateless(Flux2VAEDecoderWrapper(vae), (latent,), ("z",), ("image",))
        program.save_asset(out / "VAEDecoder.aimodel", build_aimodel_metadata(hf_model_id, component="VAEDecoder"))
    if encoder:
        vae = AutoencoderKLFlux2.from_pretrained(hf_model_id, subfolder="vae", torch_dtype=dtype)
        image = torch.randn(1, 3, height, width, dtype=dtype)
        program = export_stateless(Flux2VAEEncoderWrapper(vae), (image,), ("image",), ("latent_params",))
        program.save_asset(out / "VAEEncoder.aimodel", build_aimodel_metadata(hf_model_id, component="VAEEncoder"))
