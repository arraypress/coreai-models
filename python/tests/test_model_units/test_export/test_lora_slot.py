# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The LoRA slot's wrapping, on a toy module (no FLUX download)."""

import torch

from coreai_models.diffusion.lora_slot import SlotLinear, slot_input_names, slot_targets


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(8, 4)])
        self.norm = torch.nn.LayerNorm(8)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        self.ff = torch.nn.Linear(8, 8)


def test_targets_are_attention_projections_in_module_order():
    block = _Block()
    names = [name for name, _ in slot_targets(block)]
    assert names == ["attn.to_q", "attn.to_out.0"]


def test_zero_matrices_reproduce_the_base_layer_exactly():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 4)
    ctx = [torch.zeros(2, 8), torch.zeros(4, 2)]
    x = torch.randn(3, 8)
    assert torch.equal(SlotLinear(base, ctx, 0)(x), base(x))


def test_slot_matches_a_merged_lora():
    torch.manual_seed(1)
    base = torch.nn.Linear(8, 4)
    a, b = torch.randn(2, 8), torch.randn(4, 2)
    merged = torch.nn.Linear(8, 4)
    with torch.no_grad():
        merged.weight.copy_(base.weight + b @ a)
        merged.bias.copy_(base.bias)
    x = torch.randn(3, 8)
    assert torch.allclose(SlotLinear(base, [a, b], 0)(x), merged(x), atol=1e-5)


def test_input_names_pair_a_and_b_per_layer():
    assert slot_input_names(2) == ("lora_a_0", "lora_b_0", "lora_a_1", "lora_b_1")


def test_shape_function_names_keep_the_square_defaults():
    from coreai_models.diffusion.lora_slot import shape_function_names

    assert shape_function_names(1024, 1024) == ("main", "img2img_full")
    assert shape_function_names(864, 1152) == ("t2i_864x1152", "img2img_864x1152_full")


def test_rewrap_gives_fresh_slot_modules_over_the_same_weights():
    from coreai_models.diffusion.lora_slot import SlotTransformer, _rewrap

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.to_q = torch.nn.Linear(4, 4)

    model = Tiny()
    wrapper = SlotTransformer(model)
    before = model.attn.to_q
    _rewrap(wrapper)
    after = model.attn.to_q
    assert after is not before and after.base is before.base
    assert after.ctx is wrapper.ctx and all(v is None for v in wrapper.ctx)


def test_shape_inputs_double_the_grid_for_a_reference():
    from types import SimpleNamespace

    from coreai_models.diffusion.lora_slot import _shape_inputs

    transformer = SimpleNamespace(config=SimpleNamespace(axes_dims_rope=[32, 32, 32, 32], in_channels=128,
                                                         joint_attention_dim=64))
    plain = _shape_inputs(transformer, 54, 72, False, torch.float32)
    edit = _shape_inputs(transformer, 54, 72, True, torch.float32)
    assert plain[0].shape == (1, 54 * 72, 128) and edit[0].shape == (1, 2 * 54 * 72, 128)
    ids = edit[4][0]
    assert ids[0, 0] == 0 and ids[54 * 72, 0] == 10
    assert ids[54 * 72 - 1, 1] == 53 and ids[54 * 72 - 1, 2] == 71
