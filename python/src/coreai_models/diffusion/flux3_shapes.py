"""FLUX.3 [klein] in every shape: one transformer and one autoencoder per shape, from one load.

    python -m coreai_models.diffusion.flux3_shapes BUNDLE --shapes 16x9,9x16,4x5,4x3,3x2 [--compression 4bit]
    python -m coreai_models.diffusion.flux3_shapes BUNDLE --edit --shapes 1x1,16x9,9x16,4x3,3x4,3x2,2x3

`BUNDLE` is a finished 1:1 export (its `metadata.json`, `tokenizer` and batch-norm statistics are
copied into each shape). Each shape lands in `BUNDLE/shapes/<W>x<H ratio>/` the way FLUX.2's do:
`Transformer.aimodel` (text to image at that size), `VAEDecoder.aimodel`, `metadata.json` naming
`image_width` / `image_height` and the shared text encoder as `../../TextEncoder.aimodel`.
With `--edit`, each shape instead gains the editing transformer — `Transformer_img2img_full.aimodel`,
traced with one full grid of reference tokens (several references share it in quarters at run
time) — and a `VAEEncoder.aimodel` to read references at that size. 1:1's folder holds only its
editor and encoder, pointing at the 1:1 transformer and decoder above it.

The weights are the same at every shape, so they are loaded — and, for 4-bit, quantized — once
and traced once per shape: the quantization pass alone took 18 minutes for one shape, a trace
20 seconds. Every shape is its own asset, never one multi-function asset: Core AI compiles every
function of an asset when it loads it, and sixteen FLUX.2 functions compiled to 116 GB.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import torch

from coreai_models.diffusion.flux3 import (TEXT_PAD_MULTIPLE, Flux3ImageCondTransformerWrapper, Flux3Parts,
                                           Flux3TransformerWrapper, flux3_transformer_dynamic_shapes,
                                           flux3_transformer_image_cond_dynamic_shapes, image_position_ids,
                                           reference_position_ids, text_position_ids)

logger = logging.getLogger(__name__)

MODEL_ID = "black-forest-labs/flux-3-image-klein"

# The shapes imagine draws, in pixels (width, height), by folder name — FLUX.2's sizes, every
# side a multiple of 32 so several references can share a grid in halves.
SHAPES: dict[str, tuple[int, int]] = {
    "1x1": (1024, 1024),
    "16x9": (1344, 768),
    "9x16": (768, 1344),
    "4x5": (896, 1120),
    "4x3": (1152, 864),
    "3x4": (864, 1152),
    "3x2": (1248, 832),
    "2x3": (832, 1248),
}

PATCH = 16

TRANSFORMER_INPUTS = ("hidden_states", "encoder_hidden_states", "timestep", "target_timestep", "img_ids", "txt_ids")
EDIT_INPUTS = TRANSFORMER_INPUTS + ("image_cond_hidden_states", "image_cond_ids")
EDITOR = "Transformer_img2img_full.aimodel"


def transformer_trace_inputs(width: int, height: int, cfg: dict, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Inputs at one shape, two text windows (torch.export fixes a dimension traced at size 1)."""
    rows, cols = height // PATCH, width // PATCH
    text_len = 2 * TEXT_PAD_MULTIPLE
    return (
        torch.randn(1, rows * cols, cfg["in_channels"], dtype=dtype),
        torch.randn(1, text_len, cfg["joint_attention_dim"], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        torch.tensor([0.990472], dtype=dtype),
        image_position_ids(rows, cols),
        text_position_ids(text_len),
    )


def edit_trace_inputs(width: int, height: int, cfg: dict, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Picture inputs plus one reference filling the image's own grid."""
    rows, cols = height // PATCH, width // PATCH
    return transformer_trace_inputs(width, height, cfg, dtype) + (
        torch.randn(1, rows * cols, cfg["cond_in_channels"], dtype=dtype),
        reference_position_ids([(rows, cols)]),
    )


def edit_metadata(existing: dict | None, top: dict, name: str, width: int, height: int) -> dict:
    """A shape folder's metadata once it edits: its own encoder; 1:1's transformer and decoder
    are the ones above it, and an editing-only shape loads its editor as the transformer."""
    meta = existing if existing is not None else shape_metadata(top, width, height)
    assets = meta["assets"]
    assets["vae_encoder"] = "VAEEncoder.aimodel"
    if name == "1x1":
        assets["transformer"] = "../../Transformer.aimodel"
        assets["vae_decoder"] = "../../VAEDecoder.aimodel"
    elif existing is None:
        assets["transformer"] = EDITOR
    return meta


def shape_metadata(top: dict, width: int, height: int) -> dict:
    """The 1:1 bundle's metadata for one shape folder: local transformer and autoencoder, the
    shared text encoder two folders up, and the shape's size."""
    meta = json.loads(json.dumps(top))
    assets = meta.setdefault("assets", {})
    assets["transformer"] = "Transformer.aimodel"
    assets["vae_decoder"] = "VAEDecoder.aimodel"
    assets["text_encoder"] = "../../TextEncoder.aimodel"
    assets.pop("vae_encoder", None)
    meta["diffusion"]["image_width"] = width
    meta["diffusion"]["image_height"] = height
    return meta


def copy_sidecars(bundle: Path, folder: Path) -> None:
    """The tokenizer and batch-norm statistics, as FLUX.2's shape folders carry them."""
    if (folder / "tokenizer").exists():
        shutil.rmtree(folder / "tokenizer")
    shutil.copytree(bundle / "tokenizer", folder / "tokenizer")
    for name in ("vae_bn_mean.npy", "vae_bn_var.npy"):
        shutil.copy2(bundle / name, folder / name)


def main() -> None:
    from coreai_models.diffusion.components import quant_weight_owner
    from coreai_models.diffusion.flux2 import Flux2VAEDecoderWrapper, Flux2VAEEncoderWrapper
    from coreai_models.diffusion.gpu import export_stateless
    from coreai_models.diffusion.pipeline import _quantize_component_weights, _resolve_compression
    from coreai_models.export.metadata import build_aimodel_metadata

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path, help="A finished 1:1 FLUX.3 export")
    parser.add_argument("--shapes", required=True, help="Comma-separated: " + ",".join(SHAPES))
    parser.add_argument("--compression", default=None, help="As the standard export's, e.g. 4bit")
    parser.add_argument("--edit", action="store_true", help="The editing transformer and encoder, not drawing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    names = [name.strip() for name in args.shapes.split(",") if name.strip()]
    unknown = [name for name in names if name not in SHAPES]
    if unknown:
        raise SystemExit(f"unknown shapes {unknown}; known: {list(SHAPES)}")
    if not args.edit and "1x1" in names:
        raise SystemExit("1:1 drawing is the bundle itself; only --edit has a 1x1 folder")
    top = json.loads((args.bundle / "metadata.json").read_text())
    dtype = torch.bfloat16
    parts = Flux3Parts(MODEL_ID, dtype)
    cfg = parts.transformer_config
    inputs = edit_trace_inputs if args.edit else transformer_trace_inputs

    start = time.time()
    transformer = parts.load_transformer()
    wrapper = (Flux3ImageCondTransformerWrapper if args.edit else Flux3TransformerWrapper)(transformer).eval()
    logger.info("transformer loaded in %.0f s", time.time() - start)
    quant_config = _resolve_compression(args.compression) if args.compression else None
    if quant_config is not None:
        start = time.time()
        _quantize_component_weights(wrapper, quant_weight_owner(wrapper), inputs(*SHAPES[names[0]], cfg, dtype),
                                    quant_config, set())
        logger.info("quantized once in %.0f s", time.time() - start)

    def save(module: torch.nn.Module, trace: tuple, names_in: tuple, names_out: tuple, path: Path,
             component: str, dynamic_shapes: tuple | None = None) -> None:
        program = export_stateless(module, trace, names_in, names_out, dynamic_shapes=dynamic_shapes,
                                   include_debug_info=False)
        if path.exists():
            shutil.rmtree(path)
        program.save_asset(path, build_aimodel_metadata(MODEL_ID, component=component))

    for name in names:
        width, height = SHAPES[name]
        folder = args.bundle / "shapes" / name
        folder.mkdir(parents=True, exist_ok=True)
        start = time.time()
        metadata_path = folder / "metadata.json"
        existing = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        if args.edit:
            save(wrapper, inputs(width, height, cfg, dtype), EDIT_INPUTS, ("output",), folder / EDITOR,
                 "Transformer", flux3_transformer_image_cond_dynamic_shapes())
            image = torch.randn(1, 3, height, width, dtype=dtype)
            save(Flux2VAEEncoderWrapper(parts.vae).eval(), (image,), ("image",), ("latent_params",),
                 folder / "VAEEncoder.aimodel", "VAEEncoder")
            meta = edit_metadata(existing, top, name, width, height)
        else:
            save(wrapper, inputs(width, height, cfg, dtype), TRANSFORMER_INPUTS, ("output",),
                 folder / "Transformer.aimodel", "Transformer", flux3_transformer_dynamic_shapes())
            meta = shape_metadata(top, width, height)
            if existing is not None and "vae_encoder" in existing["assets"]:
                meta["assets"]["vae_encoder"] = existing["assets"]["vae_encoder"]
        # Drawing writes its decoder; an editor needs one only where the shape has none (1:1's is above).
        if not args.edit or (name != "1x1" and not (folder / "VAEDecoder.aimodel").exists()):
            latent = torch.randn(1, parts.vae.config.latent_channels, height // 8, width // 8, dtype=dtype)
            save(Flux2VAEDecoderWrapper(parts.vae).eval(), (latent,), ("z",), ("image",),
                 folder / "VAEDecoder.aimodel", "VAEDecoder")
        copy_sidecars(args.bundle, folder)
        metadata_path.write_text(json.dumps(meta, indent=2))
        logger.info("%s (%d × %d) %s exported in %.0f s", name, width, height,
                    "editor" if args.edit else "picture", time.time() - start)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
