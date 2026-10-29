// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing
@testable import CoreAIDiffusionPipeline

/// FLUX.3 [klein]'s recipe, checked against diffusers' modular FLUX.3 pipeline.
struct Flux3RecipeTests {

    @Test("Sigmas match diffusers' FlowMatchEulerDiscreteScheduler(shift: 6.93) to 1e-6")
    func sigmasMatchDiffusers() {
        // FlowMatchEulerDiscreteScheduler(shift=6.93).set_timesteps(sigmas=linspace(1, 1/16, 16)), diffusers 0.42 dev.
        let reference: [Float] = [1.0, 0.990472, 0.979802, 0.967773, 0.954107, 0.938446, 0.920319, 0.899092,
                                  0.873897, 0.843506, 0.806126, 0.759036, 0.697885, 0.615271, 0.497487, 0.316005]
        let sigmas = FlowTransformerPipeline.flux3Sigmas(steps: 16)
        #expect(sigmas.count == 16)
        for (got, want) in zip(sigmas, reference) { #expect(abs(got - want) < 1e-6) }
        let scheduler = DiscreteFlowScheduler(sigmas: sigmas)
        #expect(scheduler.sigmas.last == 0, "the last step jumps to 0")
    }

    @Test("The text window rounds up to a multiple of 80, capped at 13 windows")
    func textWindow() {
        #expect(FlowTransformerPipeline.flux3TextWindow(tokens: 26) == 80)
        #expect(FlowTransformerPipeline.flux3TextWindow(tokens: 80) == 80)
        #expect(FlowTransformerPipeline.flux3TextWindow(tokens: 81) == 160)
        #expect(FlowTransformerPipeline.flux3TextWindow(tokens: 329) == 400)
        #expect(FlowTransformerPipeline.flux3TextWindow(tokens: 5000) == 1040)
        #expect(FlowTransformerPipeline.flux3TextWindow(tokens: 0) == 80)
    }

    @Test("Padding sits at position 1 inside Qwen3-VL")
    func textEncoderPositions() {
        #expect(FlowTransformerPipeline.flux3TextEncoderPositions(window: 6, realTokens: 3) == [0, 1, 2, 1, 1, 1])
    }

    @Test("Image ids scale h and w by 0.5; references take a negative l")
    func imageIds() {
        let ids = FlowTransformerPipeline.flux3ImageIds(rows: 2, cols: 3)
        #expect(ids.count == 2 * 3 * 4)
        #expect(Array(ids[(1 * 3 + 2) * 4..<(1 * 3 + 2) * 4 + 4]) == [0, 0.5, 1.0, 0], "row 1, column 2")
        let reference = FlowTransformerPipeline.flux3ImageIds(rows: 1, cols: 1, lCoord: -1)
        #expect(reference == [0, 0, 0, -1])
        #expect(FlowTransformerPipeline.flux3TextIds(textSeqLen: 3) == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2])
    }

    @Test("Reference slots: phases for a reference with several, block means for one; one reference is its full grid")
    func referenceLayout() {
        let one = FlowTransformerPipeline.flux3SlotLayout(slots: FlowTransformerPipeline.referenceSlots(count: 1))
        #expect(one.map(\.reference) == [0, 0, 0, 0])
        // Every full-grid position exactly once across the four phases.
        var covered = Set<[Int]>()
        for slot in one {
            let (py, px) = slot.phase!
            for h in 0..<3 { for w in 0..<4 { covered.insert([2 * h + py, 2 * w + px]) } }
        }
        #expect(covered.count == 6 * 8, "four phases of a 6 × 8 grid are the grid")
        let two = FlowTransformerPipeline.flux3SlotLayout(slots: FlowTransformerPipeline.referenceSlots(count: 2))
        #expect(two.map(\.reference) == [0, 1, 0, 1])
        #expect(two[0].phase! == (0, 0) && two[2].phase! == (1, 1), "a reference's two slots are two different phases")
        let three = FlowTransformerPipeline.flux3SlotLayout(slots: FlowTransformerPipeline.referenceSlots(count: 3))
        #expect(three.map(\.reference) == [0, 1, 2, 0])
        #expect(three[1].phase == nil && three[2].phase == nil, "one slot: block means")
        #expect(three[0].phase != nil && three[3].phase != nil)
        #expect(FlowTransformerPipeline.flux3Id(row: 2.5, col: 4, l: -2) == [0, 1.25, 2, -2], "block centres scale like any position")
    }
}
