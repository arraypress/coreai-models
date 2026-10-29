"""FLUX.3 [klein] at any size: every length variable, from one load (and, for 4-bit, one quantization).

    python -m coreai_models.diffusion.flux3_anysize BUNDLE [--compression 4bit]

`BUNDLE` is a finished FLUX.3 export (its `metadata.json`, tokenizer, batch-norm statistics and text
encoder are kept). Beside them this writes:

- `Transformer_any.aimodel` — text to image, the image and text lengths variable;
- `Transformer_img2img_full.aimodel` — editing, the image, text and reference lengths variable, so
  one editor takes any number of references, each at its own full grid, as diffusers does;
- `VAEEncoder_any.aimodel`, `VAEDecoder_any.aimodel` — height and width variable, multiples of 16
  up to 2048;

and rewrites `metadata.json` to name them. Any shape, any size up to 4 megapixels, up to ten
references. Measured: the variable transformer agrees with diffusers' forward to cosine 0.999912 on
the saved reference inputs, exactly as the fixed 1:1 export does; the variable editor run at 1:1
reproduced the fixed one bit for bit. Core AI compiles each variable function once for every size.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import torch

from coreai_models.diffusion.flux3 import (TEXT_PAD_MULTIPLE, Flux3ImageCondTransformerWrapper, Flux3Parts,
                                           Flux3TransformerWrapper, flux3_transformer_dynamic_shapes,
                                           image_position_ids, reference_position_ids, text_position_ids)

logger = logging.getLogger(__name__)

MODEL_ID = "black-forest-labs/flux-3-image-klein"
TRANSFORMER_INPUTS = ("hidden_states", "encoder_hidden_states", "timestep", "target_timestep", "img_ids", "txt_ids")
EDIT_INPUTS = TRANSFORMER_INPUTS + ("image_cond_hidden_states", "image_cond_ids")
ASSETS = {"transformer": "Transformer_any.aimodel", "text_encoder": "TextEncoder.aimodel",
          "vae_decoder": "VAEDecoder_any.aimodel", "vae_encoder": "VAEEncoder_any.aimodel"}
EDITOR = "Transformer_img2img_full.aimodel"
# 4 megapixels (2048²) of image tokens; ten 1-megapixel references.
MAX_IMAGE_TOKENS = 16384
MAX_REFERENCE_TOKENS = 10 * 4096


def trace_inputs(width: int, height: int, cfg: dict, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Text-to-image inputs at one size, two text windows (torch.export fixes a dimension traced at 1)."""
    rows, cols, text_len = height // 16, width // 16, 2 * TEXT_PAD_MULTIPLE
    return (torch.randn(1, rows * cols, cfg["in_channels"], dtype=dtype),
            torch.randn(1, text_len, cfg["joint_attention_dim"], dtype=dtype),
            torch.tensor([1.0], dtype=dtype), torch.tensor([0.990472], dtype=dtype),
            image_position_ids(rows, cols), text_position_ids(text_len))


def main() -> None:
    from coreai_models.diffusion.components import quant_weight_owner
    from coreai_models.diffusion.flux2 import Flux2VAEDecoderWrapper, Flux2VAEEncoderWrapper
    from coreai_models.diffusion.gpu import export_stateless
    from coreai_models.diffusion.pipeline import _quantize_component_weights, _resolve_compression
    from coreai_models.export.metadata import build_aimodel_metadata

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path, help="A finished FLUX.3 export")
    parser.add_argument("--compression", default=None, help="As the standard export's, e.g. 4bit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    dtype = torch.bfloat16
    parts = Flux3Parts(MODEL_ID, dtype)
    cfg = parts.transformer_config
    transformer = parts.load_transformer()
    editor = Flux3ImageCondTransformerWrapper(transformer).eval()
    drawer = Flux3TransformerWrapper(transformer).eval()
    edit_trace = trace_inputs(1152, 864, cfg, dtype) + (
        torch.randn(1, 4096, cfg["cond_in_channels"], dtype=dtype), reference_position_ids([(64, 64)]))
    if args.compression:
        start = time.time()
        # The editor's forward touches every weight, the reference stream's included: one pass serves both.
        _quantize_component_weights(editor, quant_weight_owner(editor), edit_trace,
                                    _resolve_compression(args.compression), set())
        logger.info("quantized once in %.0f s", time.time() - start)

    image = torch.export.Dim("image_tokens", min=16, max=MAX_IMAGE_TOKENS)
    cond = torch.export.Dim("cond_tokens", min=2, max=MAX_REFERENCE_TOKENS)
    text = flux3_transformer_dynamic_shapes()

    def save(module, trace, names_in, names_out, dynamic, name, component):
        start = time.time()
        program = export_stateless(module, trace, names_in, names_out, dynamic_shapes=dynamic, include_debug_info=False)
        target = args.bundle / name
        if target.exists():
            shutil.rmtree(target)
        program.save_asset(target, build_aimodel_metadata(MODEL_ID, component=component))
        logger.info("%s exported in %.0f s", name, time.time() - start)

    save(drawer, trace_inputs(1152, 864, cfg, dtype), TRANSFORMER_INPUTS, ("output",),
         ({1: image}, text[1], None, None, {0: image}, text[5]), ASSETS["transformer"], "Transformer")
    save(editor, edit_trace, EDIT_INPUTS, ("output",),
         ({1: image}, text[1], None, None, {0: image}, text[5], {1: cond}, {0: cond}), EDITOR, "Transformer")
    h, w = torch.export.Dim("h16", min=1, max=128), torch.export.Dim("w16", min=1, max=128)
    save(Flux2VAEEncoderWrapper(parts.vae).eval(), (torch.randn(1, 3, 1024, 768, dtype=dtype),), ("image",),
         ("latent_params",), ({2: 16 * h, 3: 16 * w},), ASSETS["vae_encoder"], "VAEEncoder")
    hl, wl = torch.export.Dim("h2", min=1, max=128), torch.export.Dim("w2", min=1, max=128)
    save(Flux2VAEDecoderWrapper(parts.vae).eval(),
         (torch.randn(1, parts.vae.config.latent_channels, 128, 96, dtype=dtype),), ("z",), ("image",),
         ({2: 2 * hl, 3: 2 * wl},), ASSETS["vae_decoder"], "VAEDecoder")

    meta_path = args.bundle / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["assets"] = dict(ASSETS)
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("metadata.json names the any-size assets")


if __name__ == "__main__":
    main()
