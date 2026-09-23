// @vitest-environment node
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { OBSERVATION_SIZE, encodeObservation, legalActions } from "@/engine/encoding";
import { CheckersGame } from "@/engine/game";
import { CheckersNetwork, type PolicyManifest } from "@/engine/network";
import { greedyAction } from "@/engine/policy";
import { applyStep, initialState, legalSteps, type Player, type State } from "@/engine/rules";
import manifest from "@/model/policy.json";

import fixture from "./fixtures/parity.json";

interface StateRecord {
  men: number[];
  kings: number[];
  side: number;
  captureInProgress: boolean;
  movingSquare: number | null;
  sequenceOrigin: number | null;
  capturedPending: number;
  noProgress: number[];
  ply: number;
}

const LOGIT_TOLERANCE = 1e-4;
const PERFT = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../../../tests/golden/data/external_perft.json", import.meta.url)), "utf8"),
) as { leaf_nodes: Record<string, number> };
const weights = readFileSync(fileURLToPath(new URL("../model/policy.bin", import.meta.url)));

function toState(record: StateRecord): State {
  return {
    men: [record.men[0], record.men[1]],
    kings: [record.kings[0], record.kings[1]],
    side: record.side as Player,
    captureInProgress: record.captureInProgress,
    movingSquare: record.movingSquare,
    sequenceOrigin: record.sequenceOrigin,
    capturedPending: record.capturedPending,
    noProgress: [record.noProgress[0], record.noProgress[1]],
    ply: record.ply,
  };
}

function network(): CheckersNetwork {
  const buffer = weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength);
  return CheckersNetwork.fromBuffer(buffer, manifest as PolicyManifest);
}

function perft(state: State, depth: number): number {
  if (depth === 0) return 1;
  let leaves = 0;
  for (const step of legalSteps(state)) {
    const transition = applyStep(state, step);
    leaves += perft(transition.after, depth - (transition.moveCompleted ? 1 : 0));
  }
  return leaves;
}

function replay(game: (typeof fixture.games)[number]) {
  return new CheckersGame({
    maxPlies: game.maxPlies,
    repetitionDraws: game.repetitionDraws,
    initialState: toState(game.initialState),
  });
}

describe("TypeScript engine parity with the Python reference", () => {
  it("matches Bik's published completed-move perft counts", () => {
    for (let depth = 0; depth <= 6; depth += 1) {
      expect(perft(initialState(), depth)).toBe(PERFT.leaf_nodes[String(depth)]);
    }
  });

  it("replays every recorded game with identical legal actions, notation, and outcomes", () => {
    const reasons = new Set<string>();
    for (const game of fixture.games) {
      const environment = replay(game);
      const notation: string[] = [];
      game.actions.forEach((action, index) => {
        expect(environment.legalActions().map((legal) => legal.action)).toEqual(game.legal[index]);
        const result = environment.step(action);
        if (result.notation) notation.push(result.notation);
      });
      expect(notation).toEqual(game.notation);
      expect(environment.legalActions()).toEqual([]);
      const outcome = environment.outcome;
      expect(outcome?.reason).toBe(game.outcome.reason);
      expect(outcome?.winner === null ? null : outcome?.winner === 0 ? "red" : "white").toBe(
        game.outcome.winner,
      );
      expect(environment.state).toEqual(toState(game.finalState));
      reasons.add(game.outcome.reason);
    }
    expect(reasons).toEqual(new Set(["no_pieces", "no_legal_move", "no_progress", "repetition", "ply_cap"]));
  });

  it("encodes observations and legal actions exactly like the Python environment", () => {
    for (const record of fixture.network) {
      const state = toState(record.state);
      const observation = encodeObservation(state);
      expect(observation).toHaveLength(OBSERVATION_SIZE);
      const ones = Array.from(observation.subarray(0, 6 * 64).entries())
        .filter(([, value]) => value === 1)
        .map(([index]) => index);
      expect(ones).toEqual(record.observation.ones);
      expect(observation[6 * 64]).toBeCloseTo(record.observation.noProgress, 7);
      expect(observation[7 * 64]).toBeCloseTo(record.observation.ply, 7);
      expect(legalActions(state).map((legal) => legal.action)).toEqual(record.legal);
    }
  });

  it("loads the committed weights that the Python export describes", () => {
    expect(createHash("sha256").update(weights).digest("hex")).toBe(manifest.weightsSha256);
    expect(fixture.weightsSha256).toBe(manifest.weightsSha256);
    expect(network().parameterCount).toBe(470_410);
  });

  it("reproduces PyTorch logits, values, and greedy actions", () => {
    const model = network();
    for (const record of fixture.network) {
      const state = toState(record.state);
      const output = model.forward(encodeObservation(state));
      output.logits.forEach((logit, index) => {
        expect(Math.abs(logit - record.logits[index])).toBeLessThan(LOGIT_TOLERANCE);
      });
      expect(Math.abs(output.value - record.value)).toBeLessThan(LOGIT_TOLERANCE);
      expect(greedyAction(output.logits, legalActions(state))).toBe(record.greedy);
    }
  });

  it("plays every recorded greedy decision exactly as the Python policy did", () => {
    const model = network();
    let decisions = 0;
    for (const game of fixture.games) {
      const greedySides = new Set<Player>();
      if (game.red === "greedy") greedySides.add(0);
      if (game.white === "greedy") greedySides.add(1);
      if (greedySides.size === 0) continue;
      const environment = replay(game);
      for (const action of game.actions) {
        if (greedySides.has(environment.state.side)) {
          const { logits } = model.forward(encodeObservation(environment.state));
          expect(greedyAction(logits, environment.legalActions())).toBe(action);
          decisions += 1;
        }
        environment.step(action);
      }
    }
    expect(decisions).toBeGreaterThan(100);
  }, 120_000);
});
