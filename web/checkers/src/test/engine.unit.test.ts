// @vitest-environment node
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it, vi } from "vitest";

import { legalActions, stepToAction } from "@/engine/encoding";
import { CheckersGame, formatMove } from "@/engine/game";
import { CheckersNetwork, type PolicyManifest } from "@/engine/network";
import { greedyAction, legalProbabilities, sampleAction, seededRandom } from "@/engine/policy";
import {
  GEOMETRY,
  applyStep,
  initialState,
  legalSteps,
  positionKey,
  squareAt,
  terminalOutcome,
  type State,
} from "@/engine/rules";
import { loadPolicyNetwork } from "@/lib/loadPolicy";
import { GameSession } from "@/lib/session";
import manifestJson from "@/model/policy.json";

const manifest = manifestJson as PolicyManifest;
const weightBytes = readFileSync(fileURLToPath(new URL("../model/policy.bin", import.meta.url)));

function weights(): ArrayBuffer {
  return weightBytes.buffer.slice(weightBytes.byteOffset, weightBytes.byteOffset + weightBytes.byteLength);
}

function session(humanColor: "red" | "black" = "red", policyMode: "greedy" | "sampled" = "greedy") {
  return new GameSession({ id: "g", humanColor, policyMode, seed: 7, maxPlies: 512, repetitionDraws: true });
}

const zeros = { logits: new Float32Array(128), value: 0.1, inferenceMs: 2 };

describe("rules edge cases", () => {
  it("builds the frozen geometry table and board numbering", () => {
    expect(GEOMETRY[0]).toEqual([[5, 9], [4, null], [null, null], [null, null]]);
    expect(GEOMETRY[27]).toEqual([[null, null], [31, null], [null, null], [23, 18]]);
    expect(squareAt(0, 6)).toBe(0);
    expect(squareAt(0, 1)).toBeNull();
    expect(squareAt(8, 0)).toBeNull();
  });

  it("rejects illegal steps and mid-capture position keys", () => {
    expect(() => applyStep(initialState(), { origin: 8, destination: 16, captured: null })).toThrow(/not legal/);
    const capture: State = { ...initialState(), captureInProgress: true, movingSquare: 8, sequenceOrigin: 8 };
    expect(() => positionKey(capture)).toThrow(/completed-move/);
    expect(() => new CheckersGame({ initialState: capture })).toThrow(/completed-move/);
  });

  it("promotes a man that reaches the far row and ends its capture there", () => {
    const state: State = {
      ...initialState(),
      men: [1 << 21, 1 << 25],
      kings: [0, 0],
    };
    const [jump] = legalSteps(state);
    expect(jump).toEqual({ origin: 21, destination: 30, captured: 25 });
    const { after, moveCompleted } = applyStep(state, jump);
    expect(moveCompleted).toBe(true);
    expect(after.kings[0]).toBe(1 << 30);
    expect(terminalOutcome(after)).toEqual({ winner: 0, reason: "no_pieces" });
  });

  it("formats notation and rejects play after the game ends", () => {
    expect(formatMove([8, 12], false)).toBe("9-13");
    expect(formatMove([21, 14, 5], true)).toBe("22x15x6");
    const game = new CheckersGame({ maxPlies: 1 });
    expect(() => game.step(999)).toThrow(/illegal action/);
    game.step(game.legalActions()[0].action);
    expect(game.outcome).toEqual({ winner: null, reason: "ply_cap" });
    expect(game.legalActions()).toEqual([]);
    expect(() => game.step(0)).toThrow(/finished/);
  });

  it("encodes White's actions in the rotated canonical frame", () => {
    const state: State = { ...initialState(), side: 1 };
    const actions = legalActions(state);
    expect(actions.length).toBe(7);
    for (const { action, step } of actions) expect(stepToAction(state, step)).toBe(action);
  });
});

describe("masked policy selection", () => {
  const legal = legalActions(initialState());

  it("takes the best legal logit and breaks ties toward the lowest action", () => {
    const logits = new Float32Array(128).fill(5);
    expect(greedyAction(logits, legal)).toBe(Math.min(...legal.map(({ action }) => action)));
    logits[legal[3].action] = 9;
    logits[0] = 100;
    expect(greedyAction(logits, legal)).toBe(legal[3].action);
    expect(() => greedyAction(logits, [])).toThrow(/no legal action/);
  });

  it("normalizes over legal moves only and samples reproducibly", () => {
    const logits = new Float32Array(128);
    logits[legal[0].action] = 3;
    const probabilities = legalProbabilities(logits, legal);
    expect([...probabilities.values()].reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
    expect(probabilities.get(legal[0].action)).toBeGreaterThan(0.7);
    const first = Array.from({ length: 20 }, seededRandom(42));
    expect(Array.from({ length: 20 }, seededRandom(42))).toEqual(first);
    expect(first.every((value) => value >= 0 && value < 1)).toBe(true);
    const random = seededRandom(1);
    const picks = Array.from({ length: 200 }, () => sampleAction(logits, legal, random));
    expect(picks.filter((action) => action === legal[0].action).length).toBeGreaterThan(120);
    expect(sampleAction(logits, legal, () => 0.999999999)).toBe(Math.max(...legal.map(({ action }) => action)));
  });
});

describe("browser game session", () => {
  it("snapshots the opening for the human side", () => {
    const snapshot = session().snapshot();
    expect(snapshot.board).toHaveLength(64);
    expect(snapshot.pieces).toHaveLength(24);
    expect(snapshot.legalMoves).toHaveLength(7);
    expect(snapshot).toMatchObject({ humanColor: "red", modelColor: "black", isHumanTurn: true, outcome: null });
  });

  it("validates human moves and turn order", () => {
    const game = session();
    expect(() => game.applyModelDecision(zeros)).toThrow(/not the AI's turn/);
    expect(() => game.applyHumanMove(8, 20)).toThrow(/not a legal move/);
    game.applyHumanMove(8, 12);
    expect(game.snapshot()).toMatchObject({ isHumanTurn: false, legalMoves: [], lastStep: { origin: 8, destination: 12 } });
    expect(() => game.applyHumanMove(9, 13)).toThrow(/Wait for the AI/);
    const insight = game.applyModelDecision(zeros);
    expect(insight.legalMoveCount).toBe(7);
    expect(insight.confidence).toBeCloseTo(1 / 7);
    expect(game.snapshot().moves.map((move) => move.actor)).toEqual(["red", "black"]);
  });

  it("plays sampled policy moves and reports the outcome in site colors", () => {
    const game = session("black", "sampled");
    let guard = 0;
    while (guard < 600 && !game.snapshot().outcome) {
      if (game.isModelTurn) game.applyModelDecision(zeros);
      else {
        const [first] = game.snapshot().legalMoves;
        game.applyHumanMove(first.origin, first.destination);
      }
      guard += 1;
    }
    const { outcome } = game.snapshot();
    expect(outcome).not.toBeNull();
    expect(["red", "black", null]).toContain(outcome?.winner);
    expect(outcome?.isDraw).toBe(outcome?.winner === null);
    expect(() => game.applyHumanMove(0, 4)).toThrow(/already ended/);
  });
});

describe("policy weights", () => {
  it("rejects manifests and buffers that do not match the network", () => {
    expect(() => CheckersNetwork.fromBuffer(weights(), { ...manifest, schema: "OTHER" })).toThrow(/schema/);
    expect(() => CheckersNetwork.fromBuffer(new ArrayBuffer(8), manifest)).toThrow(/unexpected size/);
    expect(() => CheckersNetwork.fromBuffer(weights(), { ...manifest, tensors: [] })).toThrow(/layout does not match/);
    const renamed = manifest.tensors.map((record, index) => (index === 0 ? { ...record, name: "x" } : record));
    expect(() => CheckersNetwork.fromBuffer(weights(), { ...manifest, tensors: renamed })).toThrow(/mismatch for stem/);
    expect(() => CheckersNetwork.fromBuffer(weights(), { ...manifest, parameterCount: 1 })).toThrow(/parameter count/);
    const poisoned = weights();
    new Float32Array(poisoned)[0] = Number.NaN;
    expect(() => CheckersNetwork.fromBuffer(poisoned, manifest)).toThrow(/finite/);
    const network = CheckersNetwork.fromBuffer(weights(), manifest);
    expect(() => network.forward(new Float32Array(3))).toThrow(/8 × 8 × 8/);
  });

  it("downloads and checksums weights before building the network", async () => {
    const ok = vi.fn(async () => new Response(weights()));
    await expect(loadPolicyNetwork(ok as typeof fetch)).resolves.toBeInstanceOf(CheckersNetwork);
    expect(ok).toHaveBeenCalledWith(expect.stringContaining("policy"));

    const missing = vi.fn(async () => new Response(null, { status: 404 }));
    await expect(loadPolicyNetwork(missing as typeof fetch)).rejects.toThrow(/HTTP 404/);

    const tampered = weights();
    new Float32Array(tampered)[5] += 1;
    const corrupt = vi.fn(async () => new Response(tampered));
    await expect(loadPolicyNetwork(corrupt as typeof fetch)).rejects.toThrow(/integrity/);
  });
});
