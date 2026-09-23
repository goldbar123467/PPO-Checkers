/**
 * Masked action selection, matching `checkers.agents.policy_agent.PolicyAgent`.
 *
 * Illegal actions are removed before selection: greedy play takes the highest legal logit
 * (ties resolve to the lowest action ID, like `argmax`), and sampled play draws from the
 * softmax over legal logits only.
 */
import type { LegalAction } from "./encoding";

export type PolicyMode = "greedy" | "sampled";

/** Softmax probabilities over the legal actions only, keyed by action ID. */
export function legalProbabilities(
  logits: ArrayLike<number>,
  legal: readonly LegalAction[],
): Map<number, number> {
  if (legal.length === 0) throw new Error("state has no legal action");
  const maximum = Math.max(...legal.map(({ action }) => logits[action]));
  const weights = legal.map(({ action }) => Math.exp(logits[action] - maximum));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  return new Map(legal.map(({ action }, index) => [action, weights[index] / total]));
}

export function greedyAction(logits: ArrayLike<number>, legal: readonly LegalAction[]): number {
  if (legal.length === 0) throw new Error("state has no legal action");
  let best = legal[0].action;
  for (const { action } of legal) {
    const logit = logits[action];
    if (logit > logits[best] || (logit === logits[best] && action < best)) best = action;
  }
  return best;
}

export function sampleAction(
  logits: ArrayLike<number>,
  legal: readonly LegalAction[],
  random: () => number,
): number {
  const probabilities = legalProbabilities(logits, legal);
  const ordered = [...probabilities.entries()].sort(([a], [b]) => a - b);
  let threshold = random();
  for (const [action, probability] of ordered) {
    threshold -= probability;
    if (threshold < 0) return action;
  }
  return ordered[ordered.length - 1][0];
}

/** Small deterministic 32-bit PRNG (mulberry32) for seeded sampled play. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}
