// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// The naming convention of a transformer exported with `--lora-slot RANK`.
///
/// Core AI has no adapter API: a compiled asset's weights are fixed. A slot export
/// instead wraps each LoRA-targeted Linear layer as `W·x + B·(A·x)` and takes A
/// (`[rank, in]`) and B (`[out, rank]`) as ordinary inputs named `lora_a_<i>` and
/// `lora_b_<i>`, in the order the export's `slot.json` lists the layers. All-zero
/// matrices reproduce the plain export bit for bit, and one shared asset then serves
/// any number of LoRAs — each a few tens of megabytes instead of a re-export.
///
/// Pass the matrices through ``PipelineConfiguration/transformerInputs``; any the
/// caller leaves out are zeros.
public enum LoRASlot {
    /// The prefix every slot input's name starts with.
    public static let inputPrefix = "lora_"

    /// The name of layer `index`'s down-projection input.
    public static func aName(_ index: Int) -> String { "lora_a_\(index)" }

    /// The name of layer `index`'s up-projection input.
    public static func bName(_ index: Int) -> String { "lora_b_\(index)" }

    /// The slot inputs a function declares that `provided` does not cover, in declared
    /// order — the ones a run fills with zeros. Never names a non-slot input: a missing
    /// latent or embedding stays an error rather than turning silently into zeros.
    public static func missingInputs(declared: [String], provided: Set<String>) -> [String] {
        declared.filter { $0.hasPrefix(inputPrefix) && !provided.contains($0) }
    }
}
