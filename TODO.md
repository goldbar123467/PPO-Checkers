# Red House release checklist

Updated: 2026-07-29
Live: <https://checkers.upsidedownatlas.com>

## Release invariants

- The symbolic Python engine is authoritative for legality and state transitions.
- The web opponent is PPO checkpoint update 4608; Minimax-2 is evaluation-only.
- Full checkpoints, run histories, datasets, caches, and secrets never enter Git or an image.
- Public claims must trace to retained run evidence or a reproducible production measurement.

## Completed

- [x] Exported a 1,905,669-byte, 470,410-parameter CPU bundle and SHA-256 sidecar.
- [x] Proved exact source/export logits, value, and greedy-action parity on 12 positions.
- [x] Implemented Vite + React + TypeScript play with server-authoritative rules.
- [x] Labeled greedy and sampled modes as the same neural policy; removed seed input and generated fresh displayed seeds per game.
- [x] Added original generated art with provenance and a clean-room MIT UI reference study.
- [x] Added bounded sessions, request workers/timeouts, structured normalized logs, security headers, and frontend accessibility structure.
- [x] Packaged a pinned CPU-only image with an unprivileged/read-only runtime and model checksum gate.
- [x] Provisioned the approved Hetzner host, non-root key-only operator, official Docker packages, updates, UFW, and hardened SSH.
- [x] Deployed through Caddy and proxied Cloudflare DNS with HTTPS; origin web ports accept only Cloudflare networks.
- [x] Verified desktop/mobile rendering, two random seeds, a human/neural exchange, edge caching, headers, direct-origin timeout, and no browser console errors.
- [x] Restarted production: ready in 2.381 s, old ephemeral session returned 404, fresh neural reply succeeded, and 32 concurrent health probes returned 200.
- [x] Measured 308 MB image, about 160 MiB steady app RSS, and 7 ms origin move/reply latency.
- [x] Chose MIT for source/current PyTorch policy; reserved Apache-2.0 for the future Hugging Face ONNX release.
- [x] Added professional README, architecture/training/evaluation/results/deployment/model-card docs, security policy, contributing guide, and frontend CI.
- [x] Generated a compact public report from authoritative retained metrics rather than editing results by hand.

## Final publication gates

- [x] Run the complete Python and frontend gates after the documentation/release edits.
- [x] Run tracked-tree and full-history credential scans; verify ignored secrets and model/run artifacts remain untracked.
- [x] Commit and publish the intentional repository state.
- [x] Deploy the immutable publication commit and record its image/source/model manifest.
- [x] Exercise an image rollback and roll forward, including model checksum and legal-move smoke tests.
- [x] Create `checkers-policy-v1`, attach the bundle/sidecar/model card/deployment manifest, download it cleanly, verify its hash, load it, and select a legal action.
- [ ] Confirm the public GitHub Actions run is green.

## Known limitations (accepted, not hidden)

- Checkpoint selection reused the fixed ballot suite; there is no sealed external test.
- The web model comes from one long practice-run seed; human strength is not evaluated.
- Sessions are intentionally in-memory and disappear on restart.
- Cloudflare Tunnel was unavailable to the scoped token, so ingress uses proxied DNS plus a Cloudflare-only origin firewall.
- There is bounded concurrency/body/session retention and Cloudflare DDoS protection, but no per-user application rate limiter or external pager.
- Automated accessibility found low-contrast decorative square numbers/hover text; the approved visual treatment was preserved after structural critical ARIA issues were fixed.

## Next research milestones

- [ ] Export ONNX, prove PyTorch/ONNX policy/value/action parity, then publish separately to Hugging Face under Apache-2.0.
- [ ] Add a sealed opening suite, multiple full-budget seeds, and an independent stronger engine.
- [ ] Compare neural-only play with search-guided policy/value play at fixed compute.
- [ ] Add calibrated difficulty levels, then pursue camera board recognition and physical robot control.
