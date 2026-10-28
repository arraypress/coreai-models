// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import Tokenizers

/// FLUX.3 [klein]: tokenize → Qwen3-VL encode (a window of 80, 160, … tokens) → noise →
/// pack → flow-map Euler steps (the bundle's `default_steps`) → unpack → BN denorm → unpatchify → VAE decode.
/// An edit adds reference pictures, encoded once and attended to at every step.
///
/// The latent packing, BN statistics and VAE are FLUX.2's, unchanged. What differs is the
/// recipe, mirrored from diffusers' modular FLUX.3 pipeline (diffusers 0.41/0.42 dev,
/// `modular_pipelines/flux3`):
/// - the text window is the prompt's length rounded UP to a multiple of 80, and the
///   transformer attends to the padding, so the window is part of the recipe;
/// - Qwen3-VL receives explicit position ids, with padding at position 1 (the training
///   environment's behaviour);
/// - image position ids are scaled by 0.5 on the h and w axes;
/// - sigmas are `linspace(1, 1/16, 16)` shifted by a constant 6.93, and the transformer
///   takes the exact sigma plus the sigma the step jumps to (`target_timestep`).
extension FlowTransformerPipeline {
    // MARK: - Recipe Constants

    private static let flux3PatchSize = 16
    private static let flux3LatentChannels = 128
    static let flux3TextPadMultiple = 80
    static let flux3MaxTextWindows = 13
    static let flux3ImageRopeScale: Float = 0.5
    static let flux3TimestepShift: Float = 6.93
    private static let flux3AxisCount = 4

    // MARK: - Denoising Plan

    func makeFlux3Plan(_ configuration: PipelineConfiguration) async throws -> DenoisingPlan {
        let steps = configuration.stepCount

        // 1. Encode text into a window of 80, 160, … tokens.
        let (textEmbeddings, textSeqLen) = try await encodeFlux3Text(configuration.prompt)
        if configuration.lazyModelLoading { await textEncoder.unloadResources() }
        let textDim = textEmbeddings.count / textSeqLen

        // 2. Latent grid from the exported image size.
        let rows = defaultImageSize.height / Self.flux3PatchSize
        let cols = defaultImageSize.width / Self.flux3PatchSize
        let inChannels = Self.flux3LatentChannels
        let seqLen = rows * cols

        // 3. Schedule: evenly spaced sigmas shifted by a constant.
        let scheduler = DiscreteFlowScheduler(sigmas: Self.flux3Sigmas(steps: steps))

        // 4. Noise [1, 128, rows, cols], packed to [1, rows*cols, 128].
        let latentShape = [1, inChannels, rows, cols]
        let noise = generateNoise(count: latentShape.reduce(1, *), seed: configuration.seed, sourceType: .torch)
        let noisePacked = packLatentsSpatialFlatten(noise, channels: inChannels, height: rows, width: cols)

        let imageIds = Self.flux3ImageIds(rows: rows, cols: cols)
        let textIds = Self.flux3TextIds(textSeqLen: textSeqLen)
        let axisCount = Self.flux3AxisCount

        // 5. Reference pictures, when editing: tokens the transformer attends to beside the
        // image's, at timestep 0, never denoised. The generation still starts from noise.
        let reference = try await flux3References(configuration, rows: rows, cols: cols, inChannels: inChannels)
        let denoiser: CoreAIDiffusionModelFunction
        let fnName: String
        if reference != nil {
            guard let route = img2imgRoutes[.full] else {
                throw PipelineLoadError.unsupportedConfiguration(
                    "this FLUX.3 bundle has no editing transformer (Transformer_img2img_full.aimodel).")
            }
            denoiser = route.function
            fnName = route.entrypoint
        } else {
            denoiser = transformer
            fnName = transformerFunctionName
        }

        // The flow map predicts the velocity for a jump from σ to the next σ (0 after the last).
        let predict: (_ latents: [Float], _ step: Int, _ sigma: Float) async throws -> [Float] = { latents, step, sigma in
            let target = scheduler.sigmas[step + 1]
            var inputs: [([Float], [Int])] = [
                (latents, [1, seqLen, inChannels]),
                (textEmbeddings, [1, textSeqLen, textDim]),
                ([sigma], [1]),
                ([target], [1]),
                (imageIds, [seqLen, axisCount]),
                (textIds, [textSeqLen, axisCount]),
            ]
            if let reference {
                inputs.append((reference.tokens, [1, reference.count, inChannels]))
                inputs.append((reference.ids, [reference.count, axisCount]))
            }
            return try await denoiser.run(floatInputs: inputs, functionName: fnName, namedInputs: [:])
        }

        // Unpack → BN denorm → unpatchify: (1, rows*cols, 128) → (1, 32, rows*2, cols*2).
        let vaeLatents: (_ latents: [Float]) -> [Float] = { packed in
            let spatial = self.unpackLatentsSpatialFlatten(packed, channels: inChannels, height: rows, width: cols)
            let denormed = self.applyBatchNormDenorm(spatial, channels: inChannels, height: rows, width: cols)
            return Self.unpatchifyLatents(denormed, channels: inChannels, height: rows, width: cols)
        }

        return DenoisingPlan(
            latents: noisePacked,
            scheduler: scheduler,
            renoise: nil,
            predict: predict,
            vaeLatents: vaeLatents,
            vaeShape: [1, inChannels / 4, rows * 2, cols * 2],
            denoiser: denoiser,
            initialNoise: noise,
            noiseShape: latentShape)
    }

    // MARK: - References

    /// The reference tokens of an edit and their position ids, or nil without references.
    ///
    /// The editing transformer is traced with one full grid of reference tokens. One
    /// reference fills it. Several share it as FLUX.2's do: each is encoded at the full
    /// grid and area-averaged to a quarter, and the four quarters are dealt out in turn
    /// (two references get two each, the first of three gets two). Each reference keeps
    /// diffusers' FLUX.3 ids — its own grid from (0, 0), told apart by l = −1, −2, … — so
    /// the slots of one reference carry identical ids.
    func flux3References(
        _ configuration: PipelineConfiguration, rows: Int, cols: Int, inChannels: Int
    ) async throws -> (tokens: [Float], ids: [Float], count: Int)? {
        let references = configuration.referenceImages
        guard !references.isEmpty else { return nil }
        guard let encoder else {
            throw PipelineLoadError.unsupportedConfiguration("this FLUX.3 bundle has no VAE encoder to read references with.")
        }
        guard references.count <= Self.maxReferenceImages else {
            throw PipelineLoadError.unsupportedConfiguration(
                "FLUX.3 takes at most \(Self.maxReferenceImages) reference images here; got \(references.count).")
        }
        var encoded: [[Float]] = []
        for image in references {
            encoded.append(try await encodeReferenceImage(
                encoder: encoder, srcImage: image, rows: rows, cols: cols, inChannels: inChannels))
        }
        if configuration.lazyModelLoading { await encoder.unloadResources() }
        if encoded.count == 1 {
            return (encoded[0], Self.flux3ImageIds(rows: rows, cols: cols, lCoord: -1), rows * cols)
        }
        let (slotRows, slotCols) = (rows / 2, cols / 2)
        let quarters = encoded.map {
            Self.subsampleTokens($0, fromRows: rows, fromCols: cols, toRows: slotRows, toCols: slotCols, channels: inChannels)
        }
        let slots = Self.referenceSlots(count: encoded.count)
        return (slots.flatMap { quarters[$0] },
                Self.flux3ReferenceIds(slots: slots, slotRows: slotRows, slotCols: slotCols),
                slots.count * slotRows * slotCols)
    }

    /// Position ids for references dealt into quarter slots: each slot its reference's grid
    /// from (0, 0) with l = −(1 + reference).
    static func flux3ReferenceIds(slots: [Int], slotRows: Int, slotCols: Int) -> [Float] {
        slots.flatMap { flux3ImageIds(rows: slotRows, cols: slotCols, lCoord: -Float(1 + $0)) }
    }

    // MARK: - Schedule

    /// diffusers' FLUX.3 schedule: `linspace(1, 1/steps, steps)` through the constant shift
    /// `s·σ / (1 + (s−1)·σ)` of FlowMatchEulerDiscreteScheduler.
    static func flux3Sigmas(steps: Int, shift: Float = flux3TimestepShift) -> [Float] {
        let first: Float = 1, last = 1 / Float(steps)
        return (0..<steps).map { index in
            let sigma = steps == 1 ? first : first + (last - first) * Float(index) / Float(steps - 1)
            return shift * sigma / (1 + (shift - 1) * sigma)
        }
    }

    // MARK: - Text

    /// The text window for a prompt of `tokens` tokens: rounded up to a multiple of 80, capped.
    static func flux3TextWindow(tokens: Int) -> Int {
        let windows = min(max(1, (tokens + flux3TextPadMultiple - 1) / flux3TextPadMultiple), flux3MaxTextWindows)
        return windows * flux3TextPadMultiple
    }

    /// Position ids for Qwen3-VL: the token index, with padding at 1 (as in training).
    static func flux3TextEncoderPositions(window: Int, realTokens: Int) -> [Int32] {
        (0..<window).map { $0 < realTokens ? Int32($0) : 1 }
    }

    /// Must match diffusers' `Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds`: the chat
    /// template (Qwen3-VL's has no thinking block), right-padded with the pad token to the
    /// window, an attention mask over the real tokens, and explicit position ids.
    private func encodeFlux3Text(_ text: String) async throws -> (embeddings: [Float], window: Int) {
        let maxWindow = Self.flux3TextPadMultiple * Self.flux3MaxTextWindows
        let messages: [[String: String]] = [["role": "user", "content": text]]
        var ids = try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: nil, addGenerationPrompt: true,
            truncation: true, maxLength: maxWindow, tools: nil, additionalContext: nil)
        if ids.count > maxWindow { ids = Array(ids.prefix(maxWindow)) }

        let realTokens = ids.count
        let window = Self.flux3TextWindow(tokens: realTokens)
        let padTokenId = tokenizer.convertTokenToId("<|endoftext|>") ?? 151643
        while ids.count < window { ids.append(padTokenId) }

        let mask = (0..<window).map { Int32($0 < realTokens ? 1 : 0) }
        let embeddings = try await textEncoder.run(intInputs: [
            (ids.map { Int32($0) }, [1, window]),
            (mask, [1, window]),
            (Self.flux3TextEncoderPositions(window: window, realTokens: realTokens), [1, window]),
        ])
        return (embeddings, window)
    }

    // MARK: - RoPE Position IDs

    /// `img_ids` `[rows*cols, 4]` as (t, h, w, l): h and w scaled by 0.5, l = `lCoord`
    /// (0 for the generated image, −1, −2, … for references). Row-major over h then w.
    static func flux3ImageIds(rows: Int, cols: Int, lCoord: Float = 0) -> [Float] {
        var ids = [Float](repeating: 0, count: rows * cols * flux3AxisCount)
        for h in 0..<rows {
            for w in 0..<cols {
                let base = (h * cols + w) * flux3AxisCount
                ids[base + 1] = Float(h) * flux3ImageRopeScale
                ids[base + 2] = Float(w) * flux3ImageRopeScale
                ids[base + 3] = lCoord
            }
        }
        return ids
    }

    /// `txt_ids` `[textSeqLen, 4]`: the token index on the last axis.
    static func flux3TextIds(textSeqLen: Int) -> [Float] {
        var ids = [Float](repeating: 0, count: textSeqLen * flux3AxisCount)
        for s in 0..<textSeqLen { ids[s * flux3AxisCount + 3] = Float(s) }
        return ids
    }
}
