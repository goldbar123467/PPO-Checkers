import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { GameSnapshot, ModelInfo } from "../types";

const model: ModelInfo = {
  ready: true,
  bundleId: "practice-update-004608",
  experimentId: "practice",
  update: 4608,
  globalStep: 37748736,
  sourceCheckpoint: "runs/checkpoint.pt",
  sourceCheckpointSha256: "a".repeat(64),
  bundleSha256: "b".repeat(64),
  bundleSizeBytes: 2_000_000,
  gitSha: "abc123",
  gitDirty: false,
  device: "cpu",
  actionCount: 128,
  maxPlies: 512,
  repetitionDraws: true,
  parameterCount: 470410,
};

const board = Array.from({ length: 64 }, (_, index) => {
  const row = Math.floor(index / 8);
  const column = index % 8;
  const playable = (row + column) % 2 === 0;
  return { row, column, playable, square: playable ? Math.floor(index / 2) : null };
});

const game: GameSnapshot = {
  id: "game-1",
  humanColor: "red",
  modelColor: "white",
  policyMode: "greedy",
  seed: 0,
  sideToMove: "red",
  isHumanTurn: true,
  captureInProgress: false,
  forcedSquare: null,
  ply: 0,
  board,
  pieces: [{ square: 8, row: 2, column: 6, color: "red", kind: "man" }],
  legalMoves: [{ action: 32, origin: 8, destination: 12, captured: null }],
  lastStep: null,
  moves: [],
  outcome: null,
};

afterEach(() => vi.restoreAllMocks());

describe("checkers web harness", () => {
  it("gates play on model readiness and submits a selected legal move", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(model), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(game), { status: 201 }))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            ...game,
            ply: 2,
            lastStep: { origin: 20, destination: 16 },
            moves: [
              { ply: 1, actor: "red", notation: "9-13" },
              { ply: 2, actor: "white", notation: "21-17" },
            ],
          }),
          { status: 200 },
        ),
      );

    render(<App />);
    expect(screen.getByText("Waiting for the neural policy")).toBeInTheDocument();
    expect(await screen.findByText("Set the table")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /begin match/i }));
    expect(await screen.findByRole("group", { name: /red's side/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /square 9, red man, movable/i }));
    fireEvent.click(screen.getByRole("button", { name: /square 13, empty, legal destination/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    expect(await screen.findByText("21-17")).toBeInTheDocument();
  });

  it("shows a useful policy-server error", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("connection refused"));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("connection refused");
    expect(screen.queryByRole("button", { name: /begin match/i })).not.toBeInTheDocument();
  });
});
