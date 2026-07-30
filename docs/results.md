# Results

## Deployed update 4608

| Match | Games | Wins | Draws | Losses | Score | Approx. 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| vs random | 432 | 432 | 0 | 0 | 1.0000 | 0.9912–1.0000 |
| vs Minimax-2 | 432 | 354 | 70 | 8 | 0.9005 | 0.8686–0.9253 |

The evaluation is balanced across 216 opening ballots and both colors. The exact selected evaluation JSON has SHA-256 `809566af8ae420d5e50d11afc540a6d239fd98de534d3ef3ae2fe3158d11b5c0` in retained run storage.

## Final update 6144

| Match | Games | Wins | Draws | Losses | Score | Approx. 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| vs random | 432 | 432 | 0 | 0 | 1.0000 | 0.9912–1.0000 |
| vs Minimax-2 | 432 | 327 | 90 | 15 | 0.8611 | 0.8253–0.8906 |

The final policy remained perfect in this random sample but lost 0.0394 score against Minimax-2 relative to the selected checkpoint. The run therefore demonstrates non-monotonic checkpoint quality; more updates were not automatically better.

## Resource results

Across all 6,144 training metric rows:

| Metric | Mean | Maximum |
|---|---:|---:|
| GPU memory used | 7,442.6 MiB | 11,923 MiB |
| GPU utilization | 72.36% | 100% |
| GPU power | 61.85 W | 115.28 W |
| GPU temperature | 44.88 °C | 60 °C |
| Process RSS | 3.95 GB | 4.59 GB |
| Environment steps/s | 828.57 | 1,354.03 |

The self-play run took 77,845.005 measured training seconds. The pause-at-1024 and resumed invocation wall counters sum to 82,170.568 seconds. Evaluation, checkpoint serialization, W&B, and the explicit approval pause explain why elapsed human time is not identical to the inner training timer.

## Artifact results

| Artifact | Size | SHA-256 |
|---|---:|---|
| Source full checkpoint | 735,110,559 bytes | `5ae84a2e8e376cfa7c5864d2ad11955b065f9c1c6b1f79244d5321f3e6a13762` |
| Model-only CPU bundle | 1,905,669 bytes | `5d6c5c8392f7fb6897a596f5eb204f7d958f6f828d1cf56cfce98b3fcfec34fe` |

The bundle strictly reloaded and matched the source checkpoint's logits, value, and masked greedy action exactly on 12 fixed positions.

## Production measurements

On a one-vCPU, 2 GB x86-64 Hetzner server, the app image is 307,862,430 bytes. The running app used about 155–160 MiB RSS, a real human-move plus neural reply completed in 7 ms at the origin, and 32 health requests at concurrency eight all returned HTTP 200. A container restart returned ready in 2.381 seconds; the pre-restart game correctly returned 404 because sessions are explicitly ephemeral.

## Honest interpretation

This is convincing evidence that PPO self-play produced a compact policy that plays legal checkers and strongly beats the declared internal baselines under this exact evaluation. It is not evidence of solved checkers, expert human strength, a standard Elo, or generalization to sealed openings. The strongest reported checkpoint was chosen on the repeated evaluation suite and comes from one long-run seed.
