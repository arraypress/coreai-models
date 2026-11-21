// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

/// Reference-token subsampling and CFG interpolation for FLUX.2.
///
/// Neither had coverage, and both are the kind of index-and-sign arithmetic whose
/// failure mode is a subtly degraded image rather than a crash, so these pin the
/// documented properties instead of golden values.
@Suite("Flux2 token math")
struct Flux2TokenMathTests {
    /// Deterministic filler — a fixed pattern keeps failures reproducible.
    private func ramp(_ count: Int) -> [Float] {
        (0..<count).map { Float(($0 * 37) % 211) / 211.0 - 0.5 }
    }

    @Test("subsampleTokens at stride 1 is an identity copy")
    func subsampleStrideOneIsIdentity() {
        let tokens = ramp(8 * 8 * 4)
        let actual = FlowTransformerPipeline.subsampleTokens(
            tokens, fromSide: 8, toSide: 8, channels: 4)
        #expect(actual == tokens)
    }

    /// Averaging is per channel because the channel index encodes intra-patch position;
    /// mixing channels would misalign sub-positions. A per-channel constant must survive.
    @Test("subsampleTokens preserves per-channel constants")
    func subsamplePreservesChannelConstants() {
        let channels = 4
        let fromSide = 8
        var tokens = [Float](repeating: 0, count: fromSide * fromSide * channels)
        for token in 0..<(fromSide * fromSide) {
            for c in 0..<channels { tokens[token * channels + c] = Float(c) * 10 }
        }
        let actual = FlowTransformerPipeline.subsampleTokens(
            tokens, fromSide: fromSide, toSide: 2, channels: channels)
        for token in 0..<4 {
            for c in 0..<channels {
                #expect(abs(actual[token * channels + c] - Float(c) * 10) < 1e-5)
            }
        }
    }

    /// Block means, not point samples: subsampling a grid whose every 2×2 block averages
    /// to a known value must reproduce that value.
    @Test("subsampleTokens area-averages rather than point-sampling")
    func subsampleAreaAverages() {
        // 4×4 grid, 1 channel, value = token index. Each 2×2 block of a 4-wide row-major
        // grid holds {r, r+1, r+4, r+5}, so its mean is r + 2.5.
        let tokens = (0..<16).map { Float($0) }
        let actual = FlowTransformerPipeline.subsampleTokens(
            tokens, fromSide: 4, toSide: 2, channels: 1)
        #expect(actual.count == 4)
        for (i, origin) in [0, 2, 8, 10].enumerated() {
            #expect(abs(actual[i] - (Float(origin) + 2.5)) < 1e-5, "block \(i)")
        }
    }

    @Test(
        "CFG interpolation matches uncond + g*(cond - uncond)",
        arguments: [Float(0.0), 1.0, 1.5, 3.5, 7.0, 30.0])
    func cfgMatchesFormula(guidanceScale: Float) {
        let count = 512
        let cond = ramp(count)
        let uncond = ramp(count).map { $0 * -0.5 + 0.25 }
        var destination = [Float](repeating: 0, count: count)

        FlowTransformerPipeline.applyClassifierFreeGuidance(
            cond: cond[0..<count], uncond: uncond[0..<count],
            guidanceScale: guidanceScale, into: &destination)

        for i in 0..<count {
            // Same expression in the same order, so this is exact, not approximate.
            #expect(destination[i] == uncond[i] + guidanceScale * (cond[i] - uncond[i]))
        }
    }

    @Test("CFG at guidance 1 returns the conditional pass unchanged")
    func cfgAtUnitGuidanceIsConditional() {
        let cond = ramp(64)
        let uncond = ramp(64).map { -$0 }
        var destination = [Float](repeating: 0, count: 64)
        FlowTransformerPipeline.applyClassifierFreeGuidance(
            cond: cond[0..<64], uncond: uncond[0..<64], guidanceScale: 1.0,
            into: &destination)
        #expect(destination == cond)
    }

    /// The buffer is reused across denoising steps, so a stale step must not bleed through.
    @Test("CFG buffer reuse does not carry state across steps")
    func cfgBufferReuseIsClean() {
        let cond = ramp(64)
        let uncond = ramp(64).map { -$0 }
        var destination = [Float](repeating: 0, count: 64)

        FlowTransformerPipeline.applyClassifierFreeGuidance(
            cond: cond[0..<64], uncond: uncond[0..<64], guidanceScale: 25.0,
            into: &destination)
        FlowTransformerPipeline.applyClassifierFreeGuidance(
            cond: cond[0..<64], uncond: uncond[0..<64], guidanceScale: 2.0,
            into: &destination)

        for i in 0..<64 {
            #expect(destination[i] == uncond[i] + 2.0 * (cond[i] - uncond[i]))
        }
    }

    /// Slices handed in by the pipeline are prefixes of a longer transformer output;
    /// only the noise-token span may be read.
    @Test("CFG reads only the requested slice")
    func cfgHonorsSliceBounds() {
        let full = ramp(256)
        let half = 128
        var destination = [Float](repeating: 0, count: half)
        FlowTransformerPipeline.applyClassifierFreeGuidance(
            cond: full[0..<half], uncond: full[0..<half], guidanceScale: 4.0,
            into: &destination)
        // cond == uncond collapses to uncond regardless of g, so any leakage past the
        // slice bound would show up as a mismatch here.
        #expect(destination == Array(full[0..<half]))
    }
}

@Suite("Flux2 multi-reference ids")
struct Flux2MultiReferenceTests {
    @Test("four quarter slots are dealt to the references in turn")
    func slotsAreDealt() {
        #expect(FlowTransformerPipeline.referenceSlots(count: 2) == [0, 1, 0, 1])
        #expect(FlowTransformerPipeline.referenceSlots(count: 3) == [0, 1, 2, 0])
        #expect(FlowTransformerPipeline.referenceSlots(count: 4) == [0, 1, 2, 3])
    }

    /// The ids fill the traced full-grid length exactly, and each reference keeps its own T.
    @Test("multi-reference ids fill the full grid with one T per reference")
    func idsCarryOneTPerReference() {
        let axes = 4
        let noiseSide = 8
        let slotSide = 4
        let ids = FlowTransformerPipeline.buildImageIdsWithReferences(
            noiseRows: noiseSide, noiseCols: noiseSide, slots: [0, 1, 0, 1],
            slotRows: slotSide, slotCols: slotSide, axisCount: axes)
        #expect(ids.count == 2 * noiseSide * noiseSide * axes)
        let noiseSeq = noiseSide * noiseSide
        let slotSeq = slotSide * slotSide
        let firstT = (0..<4).map { ids[(noiseSeq + $0 * slotSeq) * axes] }
        #expect(firstT == [10, 20, 10, 20])
        #expect(ids[0] == 0)
        let lastCorner = (noiseSeq + 3 * slotSeq + slotSeq - 1) * axes
        #expect(ids[lastCorner + 1] == Float(slotSide - 1))
        #expect(ids[lastCorner + 2] == Float(slotSide - 1))
    }

    /// A non-square grid: the noise is rows × cols, each quarter slot half of each, and
    /// the last token of a slot sits at (rows/2 − 1, cols/2 − 1).
    @Test("multi-reference ids on a non-square grid")
    func idsOnNonSquareGrid() {
        let axes = 4
        let rows = 6, cols = 8
        let ids = FlowTransformerPipeline.buildImageIdsWithReferences(
            noiseRows: rows, noiseCols: cols, slots: [0, 0, 0, 0],
            slotRows: rows / 2, slotCols: cols / 2, axisCount: axes)
        #expect(ids.count == 2 * rows * cols * axes)
        let noiseLast = (rows * cols - 1) * axes
        #expect(ids[noiseLast + 1] == Float(rows - 1))
        #expect(ids[noiseLast + 2] == Float(cols - 1))
        let slotSeq = (rows / 2) * (cols / 2)
        let slotLast = (rows * cols + slotSeq - 1) * axes
        #expect(ids[slotLast + 1] == Float(rows / 2 - 1))
        #expect(ids[slotLast + 2] == Float(cols / 2 - 1))
    }

    /// Rectangular subsampling equals the square one on a square grid, and on a
    /// non-square grid averages each 2 × 2 block.
    @Test("subsampleTokens on rows × cols")
    func subsampleRectangular() {
        let tokens: [Float] = (0..<192).map { index -> Float in Float((index * 37) % 211) / 211 - 0.5 }
        let rectangular = FlowTransformerPipeline.subsampleTokens(
            tokens, fromRows: 8, fromCols: 8, toRows: 4, toCols: 4, channels: 3)
        let square = FlowTransformerPipeline.subsampleTokens(tokens, fromSide: 8, toSide: 4, channels: 3)
        #expect(rectangular == square)
        let grid: [Float] = (0..<24).map { index -> Float in Float(index) }
        let half = FlowTransformerPipeline.subsampleTokens(
            grid, fromRows: 4, fromCols: 6, toRows: 2, toCols: 3, channels: 1)
        #expect(half.count == 6)
        #expect(half[0] == Float(0 + 1 + 6 + 7) / 4)
        #expect(half[5] == Float(16 + 17 + 22 + 23) / 4)
    }

    /// A shape folder's `metadata.json` names its functions.
    @Test("metadata names a multi-shape transformer's functions")
    func metadataFunctionNames() throws {
        let json = #"{"type":"flux2","image_width":1152,"image_height":864,"transformer_function":"t2i_864x1152","img2img_function_prefix":"img2img_864x1152_"}"#
        let config = try JSONDecoder().decode(DiffusionConfig.self, from: Data(json.utf8))
        #expect(config.transformerFunction == "t2i_864x1152")
        #expect(config.img2imgFunctionPrefix == "img2img_864x1152_")
        let plain = try JSONDecoder().decode(DiffusionConfig.self, from: Data(#"{"type":"flux2"}"#.utf8))
        #expect(plain.transformerFunction == nil && plain.img2imgFunctionPrefix == nil)
    }
}
