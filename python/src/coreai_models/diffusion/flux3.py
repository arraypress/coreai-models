# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
FLUX.3 component specifications and torch wrappers for Core AI export.

FLUX.3 [klein] is a DiT that uses:
- a Qwen3-VL text encoder: hidden states of layers 4, 8, …, 32 stacked into 8 × 2560 = 20480
  channels per token. Only its language model runs for a text prompt, and only up to layer 32,
  so the export drops the vision tower, layers 33–36 and the output head.
- per-mode blocks (5 for the image, 5 for the text, 5 for reference images) then 28 joint
  blocks, with 4D RoPE over (t, h, w, l).
- the FLUX.2 VAE (AutoencoderKLFlux2) unchanged.
- a step-distilled "flow map": every step also embeds the timestep it jumps to
  (`target_timestep`).

Two things differ from FLUX.2 and are part of the model recipe, so the export keeps them:
- the text window is the prompt's length rounded UP to a multiple of 80 tokens, and the
  transformer attends to the padding. The text length is therefore a dynamic dimension in
  steps of 80, not a fixed 512.
- image position ids are scaled by 0.5 on the h and w axes (`IMAGE_ROPE_SCALE`); text tokens
  carry their index on the last axis; padding tokens take position 1 inside Qwen3-VL, as in
  the training environment (the runtime passes `position_ids`).
"""

from typing import Any, cast

import torch

# The recipe constants, from diffusers' modular FLUX.3 pipeline.
TEXT_PAD_MULTIPLE = 80
TEXT_ENCODER_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
IMAGE_ROPE_SCALE = 0.5
NUM_INFERENCE_STEPS = 16
TIMESTEP_SHIFT = 6.93

# The longest text window exported: 13 × 80 = 1040 tokens (a full scene caption is 150–400).
MAX_TEXT_WINDOWS = 13


# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------


def _joint_block_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb_mod: torch.Tensor,
    temb_mod_context: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flux3SingleTransformerBlock.forward with the text/content split taken from the content
    end. diffusers splits at the text length, which is the dynamic dimension, and the Core AI
    converter cannot split at a symbolic size; the content length is fixed by the image grid.
    Same arithmetic, same result."""
    mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)
    c_mod_shift, c_mod_scale, c_mod_gate = torch.chunk(temb_mod_context, 3, dim=-1)
    norm_hidden_states = (1 + mod_scale) * self.norm(hidden_states) + mod_shift
    norm_encoder_hidden_states = (1 + c_mod_scale) * self.norm(encoder_hidden_states) + c_mod_shift
    combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
    output = self.attn(hidden_states=combined, image_rotary_emb=image_rotary_emb) + self.ff(combined)
    content_len = hidden_states.shape[1]
    context_output, content_output = output[:, :-content_len], output[:, -content_len:]
    return encoder_hidden_states + c_mod_gate * context_output, hidden_states + mod_gate * content_output


class Flux3TransformerWrapper(torch.nn.Module):
    """FLUX.3 [klein] text-to-image for export: Flux3Transformer2DModel.forward without the
    reference-image path, line for line, with two export changes that keep the arithmetic:

    - the timestep embedding of 0 (what the text tokens are modulated by) is a constant, so it
      is computed once and stored; diffusers builds it from `torch.zeros(..., dtype)`, a
      `full` the Core AI converter types as float32 in a bfloat16 graph;
    - the joint blocks take `_joint_block_forward` (the split at the content end), so the
      dynamic text window exports.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        import types

        for block in transformer.single_transformer_blocks:
            block.forward = types.MethodType(_joint_block_forward, block)
        self.model = transformer
        dtype = next(transformer.parameters()).dtype
        with torch.no_grad():
            zero = transformer.time_proj(torch.zeros(1, dtype=dtype))
            self.register_buffer("zero_temb", transformer.timestep_embedder(zero.to(dtype))[:, None, :].clone())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        target_timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        dtype = hidden_states.dtype
        batch_size, image_seq_len = hidden_states.shape[:2]

        image_temb = m.timestep_embedder(m.time_proj(timestep.to(dtype) * 1000).to(dtype))[:, None, :]
        target_temb = m.target_timestep_embedder(m.time_proj(target_timestep.to(dtype) * 1000).to(dtype))[:, None, :]
        image_temb = image_temb + target_temb
        zero_temb = self.zero_temb.to(dtype) + target_temb
        image_temb = image_temb.expand(batch_size, image_seq_len, -1)

        zero_temb_act = torch.nn.functional.silu(zero_temb)
        txt_temb_act = zero_temb_act.expand(batch_size, encoder_hidden_states.shape[1], -1)
        txt_early_mod = m.early_stream_modulations["txt"](txt_temb_act)
        txt_single_mod = m.single_stream_modulations["txt"](txt_temb_act)
        encoder_hidden_states = m.context_embedder(encoder_hidden_states)
        txt_rotary_emb = m.pos_embed(txt_ids)

        image_temb_act = torch.nn.functional.silu(image_temb)
        image_early_mod = m.early_stream_modulations["image"](image_temb_act)
        image_single_mod = m.single_stream_modulations["image"](image_temb_act)
        hidden_states = m.x_embedders["image"](hidden_states)
        image_rotary_emb = m.pos_embed(img_ids)

        for block in m.txt_mode_blocks:
            encoder_hidden_states = block(encoder_hidden_states, txt_early_mod, txt_rotary_emb)
        for block in m.mode_blocks["image"]:
            hidden_states = block(hidden_states, image_early_mod, image_rotary_emb)

        joint_rotary_emb = (
            torch.cat([txt_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([txt_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )
        content_states = hidden_states
        for block in m.single_transformer_blocks:
            encoder_hidden_states, content_states = block(
                hidden_states=content_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod=image_single_mod,
                temb_mod_context=txt_single_mod,
                image_rotary_emb=joint_rotary_emb,
            )
        hidden_states = content_states[:, :image_seq_len]
        hidden_states = m.norm_out(hidden_states, image_temb)
        return cast(torch.Tensor, m.proj_out(hidden_states))


class Flux3TextEncoderWrapper(torch.nn.Module):
    """Runs Qwen3-VL's language model up to the last stacked layer and stacks the hidden states.

    The runtime passes 2-D `position_ids` (padding at position 1); the language model expands them
    itself to four rows, one for the causal mask and three for its multimodal RoPE, exactly as the
    full model does with given positions. Layers past the last one stacked never run, so they are
    dropped, and so is the final norm, which in the full model only ever follows layer 36.
    """

    def __init__(self, text_encoder: torch.nn.Module, hidden_states_layers: tuple[int, ...] = TEXT_ENCODER_LAYERS) -> None:
        super().__init__()
        language_model = text_encoder.model.language_model
        keep = max(hidden_states_layers)
        language_model.layers = language_model.layers[:keep]
        # The last hidden state a model returns is after its final norm. With the stack cut at
        # the last stacked layer, that norm would land on it; in the full model it never does.
        language_model.norm = torch.nn.Identity()
        self.model = language_model
        self.hidden_states_layers = hidden_states_layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        stacked = torch.stack([outputs.hidden_states[k] for k in self.hidden_states_layers], dim=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)


# ---------------------------------------------------------------------------
# Position ids and dummy inputs
# ---------------------------------------------------------------------------


def image_position_ids(grid_h: int, grid_w: int, l_coord: float = 0.0) -> torch.Tensor:
    """`[grid_h*grid_w, 4]` ids (t, h, w, l): h and w scaled by IMAGE_ROPE_SCALE, l constant
    (0 for the generated image, -1, -2, … for references). Row-major over h then w."""
    hh, ww = torch.meshgrid(torch.arange(grid_h, dtype=torch.float32), torch.arange(grid_w, dtype=torch.float32), indexing="ij")
    ids = torch.zeros(grid_h * grid_w, 4)
    ids[:, 1] = hh.reshape(-1) * IMAGE_ROPE_SCALE
    ids[:, 2] = ww.reshape(-1) * IMAGE_ROPE_SCALE
    ids[:, 3] = l_coord
    return ids


def text_position_ids(text_seq_len: int) -> torch.Tensor:
    """`[text_seq_len, 4]` ids: the token index on the last axis."""
    ids = torch.zeros(text_seq_len, 4)
    ids[:, 3] = torch.arange(text_seq_len, dtype=torch.float32)
    return ids


def text_window(prompt_tokens: int) -> int:
    """The text window for a prompt: its length rounded up to a multiple of 80, capped."""
    windows = min(max(1, -(-prompt_tokens // TEXT_PAD_MULTIPLE)), MAX_TEXT_WINDOWS)
    return windows * TEXT_PAD_MULTIPLE


def dummy_flux3_transformer(pipe: Any, grid_h: int = 64, grid_w: int = 64, text_seq_len: int = 160) -> tuple[torch.Tensor, ...]:
    cfg = pipe.transformer.config
    dtype = next(pipe.transformer.parameters()).dtype
    return (
        torch.randn(1, grid_h * grid_w, cfg.in_channels, dtype=dtype),
        torch.randn(1, text_seq_len, cfg.joint_attention_dim, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([0.4], dtype=dtype),
        image_position_ids(grid_h, grid_w),
        text_position_ids(text_seq_len),
    )


def flux3_transformer_dynamic_shapes() -> tuple[dict[int, Any] | None, ...]:
    """The text window is dynamic in steps of 80 (1 … MAX_TEXT_WINDOWS windows)."""
    windows = torch.export.Dim("text_windows", min=1, max=MAX_TEXT_WINDOWS)
    text_len = TEXT_PAD_MULTIPLE * windows
    return (None, {1: text_len}, None, None, None, {0: text_len})


def dummy_flux3_text_encoder(pipe: Any, text_seq_len: int = 160) -> tuple[torch.Tensor, ...]:
    ids = torch.zeros(1, text_seq_len, dtype=torch.int32)
    mask = torch.ones(1, text_seq_len, dtype=torch.int32)
    positions = torch.arange(text_seq_len, dtype=torch.int32).unsqueeze(0)
    return (ids, mask, positions)


def flux3_text_encoder_dynamic_shapes() -> tuple[dict[int, Any] | None, ...]:
    windows = torch.export.Dim("text_windows", min=1, max=MAX_TEXT_WINDOWS)
    text_len = TEXT_PAD_MULTIPLE * windows
    return ({1: text_len}, {1: text_len}, {1: text_len})
