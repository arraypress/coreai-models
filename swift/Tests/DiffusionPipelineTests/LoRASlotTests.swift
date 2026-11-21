// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Testing

@testable import CoreAIDiffusionPipeline

@Suite("LoRA slot inputs")
struct LoRASlotTests {
    private let declared = [
        "hidden_states", "encoder_hidden_states", "timestep", "guidance", "img_ids", "txt_ids",
        "lora_a_0", "lora_b_0", "lora_a_1", "lora_b_1",
    ]

    @Test("Every slot input left out defaults to zeros, in declared order")
    func missingDefaultsToZeros() {
        #expect(
            LoRASlot.missingInputs(declared: declared, provided: [])
                == ["lora_a_0", "lora_b_0", "lora_a_1", "lora_b_1"])
    }

    @Test("Provided slot inputs are not replaced")
    func providedAreKept() {
        let provided: Set = [LoRASlot.aName(0), LoRASlot.bName(0)]
        #expect(LoRASlot.missingInputs(declared: declared, provided: provided) == ["lora_a_1", "lora_b_1"])
    }

    @Test("A missing ordinary input never becomes zeros")
    func ordinaryInputsStayRequired() {
        #expect(LoRASlot.missingInputs(declared: ["hidden_states", "timestep"], provided: []).isEmpty)
    }

    @Test("A plain export declares no slot inputs, so nothing is added")
    func plainExportUntouched() {
        #expect(LoRASlot.missingInputs(declared: Array(declared.prefix(6)), provided: []).isEmpty)
    }

    @Test("Configurations default to no transformer inputs and compare by their names")
    func configurationDefault() {
        var a = PipelineConfiguration(prompt: "x")
        #expect(a.transformerInputs.isEmpty)
        let b = a
        a.transformerInputs = [LoRASlot.aName(0): NDArray(shape: [1, 1], scalarType: .float16)]
        #expect(a != b)
    }
}
