import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "@/App";
import { AppProviders } from "@/app/AppProviders";
import { createGame, fetchModel, submitMove } from "@/lib/api/checkers";
import type { BoardCell, GameSnapshot, ModelInfo } from "@/types";

vi.mock("@/lib/api/checkers", () => ({
  createGame: vi.fn(),
  fetchModel: vi.fn(),
  submitMove: vi.fn(),
}));

const model: ModelInfo = {
  ready: true,
  bundleId: "practice-update-4608",
  experimentId: "checkers-practice",
  update: 4608,
  globalStep: 37_748_736,
  sourceCheckpoint: "checkpoint-004608.pt",
  sourceCheckpointSha256: "a".repeat(64),
  bundleSha256: "b".repeat(64),
  bundleSizeBytes: 1_905_669,
  gitSha: "1234567",
  gitDirty: false,
  device: "cpu",
  actionCount: 128,
  maxPlies: 512,
  repetitionDraws: true,
  parameterCount: 470_410,
};

const board: BoardCell[] = Array.from({ length: 64 }, (_, index) => {
  const row = Math.floor(index / 8);
  const column = index % 8;
  const playable = (row + column) % 2 === 0;
  return { row, column, playable, square: playable ? row * 4 + Math.floor(column / 2) : null };
});

function game(humanColor: "red" | "white" = "red"): GameSnapshot {
  return {
    id: "game-1",
    humanColor,
    modelColor: humanColor === "red" ? "white" : "red",
    policyMode: "greedy",
    seed: 123,
    sideToMove: humanColor,
    isHumanTurn: true,
    captureInProgress: false,
    forcedSquare: null,
    ply: 0,
    board,
    pieces: [{ square: 8, row: 2, column: 0, color: humanColor, kind: "man" }],
    legalMoves: [{ action: 32, origin: 8, destination: 12, captured: null }],
    lastStep: null,
    moves: [],
    outcome: null,
  };
}

describe("IMSA West game-first application", () => {
  beforeEach(() => {
    vi.mocked(fetchModel).mockResolvedValue(model);
    vi.mocked(createGame).mockResolvedValue(game());
    vi.mocked(submitMove).mockResolvedValue(game());
  });

  it("shows the supplied school identity, real evidence, and simple game setup", async () => {
    render(<AppProviders><App /></AppProviders>);

    expect(await screen.findByText("Policy update 4,608 ready")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Indiana Math and Science Academy West logo" })).toHaveAttribute(
      "src",
      "/assets/imsa-west-logo.png",
    );
    expect(screen.getByRole("heading", { name: "Can you beat our checkers AI?" })).toBeInTheDocument();
    expect(screen.getByText("354–70–8")).toBeInTheDocument();
    expect(screen.getByText("37.7M")).toBeInTheDocument();
    expect(screen.getByText(/trained through 37,748,736 self-play transitions/i)).toBeInTheDocument();
    expect(screen.getByText(/These are project evaluation results, not a human skill rating/i)).toBeInTheDocument();
    expect(screen.getByText(/Eight actor-centered number layers/)).toBeInTheDocument();
    expect(screen.getByText(/separate 128-slot mask marks legal actions/)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Start game" })).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Start game" })).toBeEnabled();
  });

  it("starts deterministic play with the selected side and renders the accessible board", async () => {
    vi.mocked(createGame).mockResolvedValue(game("white"));
    render(<AppProviders><App /></AppProviders>);

    await screen.findByText("Policy update 4,608 ready");
    fireEvent.click(screen.getByRole("button", { name: /White AI moves first/i }));
    fireEvent.click(screen.getByRole("button", { name: "Start game" }));

    await waitFor(() => expect(createGame).toHaveBeenCalledWith("white", "greedy", expect.any(Number)));
    expect(await screen.findByRole("group", { name: /white's side/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Your turn/i })).toBeInTheDocument();
  });
});
