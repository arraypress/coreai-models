// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Decoding a picture larger than a variable-size VAE decoder can hold at once.
///
/// Measured on a 36 GB Mac: FLUX.3's decoder takes a 2048 × 2048 picture whole, but a
/// 3840 × 2160 one alone took free memory from 57% to 4%. So past
/// ``largeDecodePixels`` the latents are decoded in overlapping rectangular tiles no
/// larger than that, and the overlaps are blended with linear ramps.
extension FlowTransformerPipeline {

    /// The most output pixels decoded in one call: a 2048 × 2048 picture.
    static let largeDecodePixels = 2048 * 2048

    /// A tile's side, in latent pixels (1024 output pixels for an 8× decoder): with the
    /// transformer still loaded, one 2048-pixel tile took free memory under 10%.
    static let largeDecodeTile = 128

    /// The overlap between neighbouring tiles, in latent pixels (256 output pixels).
    static let largeDecodeOverlap = 32

    /// The tiles' spans along one axis of `length`: as few tiles of at most `tile` as
    /// cover it with at least `overlap` shared between neighbours, spread evenly, the
    /// first at 0 and the last ending at `length`.
    static func tileSpans(length: Int, tile: Int, overlap: Int) -> [Range<Int>] {
        guard length > tile else { return [0..<length] }
        let count = Int((Double(length - overlap) / Double(tile - overlap)).rounded(.up))
        return (0..<count).map { index in
            let start = Int((Double(index) * Double(length - tile) / Double(count - 1)).rounded())
            return start..<(start + tile)
        }
    }

    /// The blending weight at `position` inside `span` of an axis of `length`: rising
    /// across `ramp` pixels from each edge shared with a neighbour, 1 at the picture's
    /// own edges and in the middle.
    static func tileWeight(_ position: Int, in span: Range<Int>, length: Int, ramp: Int) -> Float {
        let fromStart = span.lowerBound == 0 ? Int.max : position - span.lowerBound
        let fromEnd = span.upperBound == length ? Int.max : span.upperBound - 1 - position
        let distance = min(fromStart, fromEnd)
        return distance >= ramp ? 1 : Float(distance + 1) / Float(ramp + 1)
    }

    /// Decodes `latents` (`[1, channels, height, width]`) tile by tile when the picture is
    /// larger than ``largeDecodePixels``, else whole.
    func decodeLarge(
        latents: [Float], channels: Int, height: Int, width: Int, scale: Int,
        decoder: CoreAIDiffusionModelFunction
    ) async throws -> [Float] {
        guard height * width * scale * scale > Self.largeDecodePixels else {
            return try await decoder.run(floatInputs: [(latents, [1, channels, height, width])])
        }
        let rowsSpans = Self.tileSpans(length: height, tile: Self.largeDecodeTile, overlap: Self.largeDecodeOverlap)
        let colsSpans = Self.tileSpans(length: width, tile: Self.largeDecodeTile, overlap: Self.largeDecodeOverlap)
        let outH = height * scale, outW = width * scale, plane = outH * outW
        let ramp = Self.largeDecodeOverlap * scale
        var output = [Float](repeating: 0, count: 3 * plane)
        var weights = [Float](repeating: 0, count: plane)

        for rows in rowsSpans {
            for cols in colsSpans {
                var tile = [Float](repeating: 0, count: channels * rows.count * cols.count)
                var index = 0
                for c in 0..<channels {
                    for y in rows {
                        let base = c * height * width + y * width
                        for x in cols { tile[index] = latents[base + x]; index += 1 }
                    }
                }
                let decoded = try await decoder.run(floatInputs: [(tile, [1, channels, rows.count, cols.count])])

                let outRows = (rows.lowerBound * scale)..<(rows.upperBound * scale)
                let outCols = (cols.lowerBound * scale)..<(cols.upperBound * scale)
                let tileW = outCols.count, tilePlane = outRows.count * tileW
                let columnWeights = outCols.map { Self.tileWeight($0, in: outCols, length: outW, ramp: ramp) }
                for (ty, y) in outRows.enumerated() {
                    let rowWeight = Self.tileWeight(y, in: outRows, length: outH, ramp: ramp)
                    for (tx, x) in outCols.enumerated() {
                        let w = rowWeight * columnWeights[tx]
                        let at = y * outW + x, from = ty * tileW + tx
                        for c in 0..<3 { output[c * plane + at] += w * decoded[c * tilePlane + from] }
                        weights[at] += w
                    }
                }
            }
        }
        for c in 0..<3 {
            for i in 0..<plane where weights[i] > 0 { output[c * plane + i] /= weights[i] }
        }
        return output
    }
}
