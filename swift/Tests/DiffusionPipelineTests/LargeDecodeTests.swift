// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing
@testable import CoreAIDiffusionPipeline

/// Tiling a decode too large to run whole.
struct LargeDecodeTests {

    @Test("A 3840 × 2160 picture decodes in 3 × 5 tiles of 1024 pixels, overlapping by at least 256")
    func fourKTiles() {
        let tile = FlowTransformerPipeline.largeDecodeTile, overlap = FlowTransformerPipeline.largeDecodeOverlap
        #expect(tile * 8 == 1024 && overlap * 8 == 256)
        let rows = FlowTransformerPipeline.tileSpans(length: 270, tile: tile, overlap: overlap)
        let cols = FlowTransformerPipeline.tileSpans(length: 480, tile: tile, overlap: overlap)
        #expect(rows == [0..<128, 71..<199, 142..<270])
        #expect(cols == [0..<128, 88..<216, 176..<304, 264..<392, 352..<480])
        #expect(2048 * 2048 == FlowTransformerPipeline.largeDecodePixels, "4 megapixels still decode whole")
    }

    @Test("Spans cover the axis from 0 to its end, overlap enough and stay within the tile size")
    func spansCover() {
        for length in [257, 300, 480, 512, 513, 700, 1000] {
            let spans = FlowTransformerPipeline.tileSpans(length: length, tile: 256, overlap: 32)
            #expect(spans.first?.lowerBound == 0)
            #expect(spans.last?.upperBound == length)
            #expect(spans.allSatisfy { $0.count == 256 })
            for (a, b) in zip(spans, spans.dropFirst()) { #expect(a.upperBound - b.lowerBound >= 32, "length \(length)") }
        }
        #expect(FlowTransformerPipeline.tileSpans(length: 200, tile: 256, overlap: 32) == [0..<200], "small enough: one span")
    }

    @Test("Weights are 1 at the picture's edges and inside, ramping only toward a neighbour")
    func weights() {
        let first = 0..<2048, second = 1792..<3840
        #expect(FlowTransformerPipeline.tileWeight(0, in: first, length: 3840, ramp: 256) == 1, "the picture's edge")
        #expect(FlowTransformerPipeline.tileWeight(1000, in: first, length: 3840, ramp: 256) == 1)
        #expect(FlowTransformerPipeline.tileWeight(2047, in: first, length: 3840, ramp: 256) < 0.01, "the shared edge")
        #expect(FlowTransformerPipeline.tileWeight(3839, in: second, length: 3840, ramp: 256) == 1)
        let rising = (1792..<2048).map { FlowTransformerPipeline.tileWeight($0, in: second, length: 3840, ramp: 256) }
        #expect(zip(rising, rising.dropFirst()).allSatisfy { $0 <= $1 }, "it rises into the tile")
        #expect(rising.allSatisfy { $0 > 0 }, "never zero, so every pixel is covered")
    }
}
