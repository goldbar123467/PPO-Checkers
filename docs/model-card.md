# Model card: Red House Checkers Policy v1

## Model details

- **Artifact:** `checkers-practice-update-004608.pt`
- **Task:** policy/value inference for American checkers
- **Architecture:** 470,410-parameter residual convolutional policy/value network
- **Input:** actor-canonical float tensor shaped `8 × 8 × 8`
- **Output:** 128 unmasked action logits and one actor-relative value in `[-1, 1]`
- **Training:** PPO self-play, seed 0, 4,608 selected updates / 37,748,736 transitions at the exported boundary
- **Framework:** PyTorch 2.13.0 source stack; CPU runtime bundle
- **License:** MIT

## Intended use

The bundle is intended for education, reproducibility, evaluation, and playing American checkers through this repository's symbolic engine. It is not a standalone rules engine. Consumers must apply the exact legal-action mapping/mask before action selection.

## Not intended for

- Selecting an unmasked action directly from logits.
- Claims of expert, tournament, solved-game, or calibrated human strength.
- Loading from an unverified source or without matching the published SHA-256.
- Safety-critical or high-stakes decisions; this is a board-game model.

## Evaluation

Update 4608 scored 1.0000 against random (432/432 wins) and 0.9005 against the project's Minimax-2 proxy (354 wins, 70 draws, 8 losses) across 216 ballots and both colors. Approximate 95% intervals were 0.9912–1.0000 and 0.8686–0.9253. The same suite was used for checkpoint selection, so this is not sealed-test evidence. See [evaluation.md](evaluation.md).

## Artifact integrity

- Bundle SHA-256: `5d6c5c8392f7fb6897a596f5eb204f7d958f6f828d1cf56cfce98b3fcfec34fe`
- Bundle size: 1,905,669 bytes
- Source checkpoint SHA-256: `5ae84a2e8e376cfa7c5864d2ad11955b065f9c1c6b1f79244d5321f3e6a13762`
- Source Git revision: `495ff829e15373c3bb5117dd13933b3a8cdfa492`
- Export parity: exact logits, values, and greedy actions on 12 deterministic positions

The loader verifies the sidecar before `torch.load(weights_only=True)`, validates the closed schema and tensor metadata, rejects non-finite tensors, loads strictly onto CPU, and switches the network to evaluation mode.

## Limitations

One full-budget seed was used for this artifact. Its evaluation baseline is internal and shallow. Training and selection used fixed ballots, no sealed external test was run, and human performance is unknown. Web sampled mode adds stochastic action selection but does not create a different or weaker calibrated model.

## Future ONNX release

ONNX/WebAssembly export is intentionally deferred until action/value parity and browser rules conformance are tested. That future Hugging Face artifact will be released separately under Apache-2.0; it is not part of this MIT PyTorch bundle.
