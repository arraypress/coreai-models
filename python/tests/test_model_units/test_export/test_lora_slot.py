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
