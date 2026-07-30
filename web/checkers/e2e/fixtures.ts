import type { Page } from "@playwright/test";
import type { BoardCell, GameSnapshot, ModelInfo, Piece } from "../src/types";

export const model: ModelInfo = {
  ready: true,
  bundleId: "practice-update-004608",
  experimentId: "checkers-practice",
  update: 4608,
  globalStep: 37_748_736,
  sourceCheckpoint: "runs/checkpoints/checkers/checkers-practice/checkpoint-004608.pt",
  sourceCheckpointSha256: "a".repeat(64),
  bundleSha256: "5d6c5c8392" + "b".repeat(54),
  bundleSizeBytes: 1_905_669,
  gitSha: "abc1234",
  gitDirty: false,
  device: "cpu",
  actionCount: 128,
  maxPlies: 512,
  repetitionDraws: true,
  parameterCount: 470_410,
};

function squareFor(row: number, column: number): number {
  return row * 4 + Math.floor((6 + (row % 2) - column) / 2);
}

export const board: BoardCell[] = Array.from({ length: 64 }, (_, index) => {
  const row = Math.floor(index / 8);
  const column = index % 8;
  const playable = (row + column) % 2 === 0;
  return { row, column, playable, square: playable ? squareFor(row, column) : null };
});

function piece(square: number, color: "red" | "white", kind: "man" | "king" = "man"): Piece {
  const cell = board.find((candidate) => candidate.square === square);
  if (!cell) throw new Error(`No board cell for square ${square}`);
  return { square, row: cell.row, column: cell.column, color, kind };
}

const openingPieces = [
  ...Array.from({ length: 12 }, (_, square) => piece(square, "red")),
  ...Array.from({ length: 12 }, (_, offset) => piece(20 + offset, "white")),
];

export function gameSnapshot(overrides: Partial<GameSnapshot> = {}): GameSnapshot {
  return {
    id: "fixture-game",
    humanColor: "red",
    modelColor: "white",
    policyMode: "greedy",
    seed: 424_242,
    sideToMove: "red",
    isHumanTurn: true,
    captureInProgress: false,
    forcedSquare: null,
    ply: 0,
    board,
    pieces: openingPieces,
    legalMoves: [
      { action: 32, origin: 8, destination: 12, captured: null },
      { action: 36, origin: 9, destination: 12, captured: null },
      { action: 37, origin: 9, destination: 13, captured: null },
      { action: 41, origin: 10, destination: 13, captured: null },
      { action: 42, origin: 10, destination: 14, captured: null },
      { action: 46, origin: 11, destination: 14, captured: null },
      { action: 47, origin: 11, destination: 15, captured: null },
    ],
    lastStep: null,
    moves: [],
    outcome: null,
    ...overrides,
  };
}

export const opening = gameSnapshot();

export const afterOpeningMove = gameSnapshot({
  ply: 2,
  pieces: [
    ...openingPieces.filter((candidate) => candidate.square !== 8 && candidate.square !== 20),
    piece(12, "red"),
    piece(16, "white"),
  ],
  legalMoves: [{ action: 52, origin: 13, destination: 17, captured: null }],
  lastStep: { origin: 20, destination: 16 },
  moves: [
    { ply: 1, actor: "red", notation: "9-13" },
    { ply: 2, actor: "white", notation: "21-17" },
  ],
});

export const forcedCapture = gameSnapshot({
  captureInProgress: true,
  forcedSquare: 12,
  pieces: [piece(12, "red"), piece(17, "white"), piece(26, "white")],
  legalMoves: [{ action: 50, origin: 12, destination: 21, captured: 17 }],
});

export const forcedContinuation = gameSnapshot({
  ply: 1,
  captureInProgress: true,
  forcedSquare: 21,
  pieces: [piece(21, "red"), piece(26, "white")],
  legalMoves: [{ action: 84, origin: 21, destination: 30, captured: 26 }],
  lastStep: { origin: 12, destination: 21 },
  moves: [{ ply: 1, actor: "red", notation: "13x22" }],
});

export const afterMultiJump = gameSnapshot({
  ply: 2,
  sideToMove: "white",
  isHumanTurn: false,
  pieces: [piece(30, "red", "king")],
  legalMoves: [],
  lastStep: { origin: 21, destination: 30 },
  moves: [{ ply: 1, actor: "red", notation: "13x22x31" }],
});

export const kingPosition = gameSnapshot({
  pieces: [piece(28, "red", "king"), piece(3, "white", "king")],
  legalMoves: [
    { action: 112, origin: 28, destination: 24, captured: null },
    { action: 113, origin: 28, destination: 25, captured: null },
  ],
});

export const gameOver = gameSnapshot({
  ply: 31,
  sideToMove: "white",
  isHumanTurn: false,
  pieces: [piece(28, "red", "king")],
  legalMoves: [],
  outcome: { winner: "red", reason: "no_pieces", isDraw: false },
  moves: [{ ply: 31, actor: "red", notation: "24x31" }],
});

interface MockOptions {
  initial?: GameSnapshot;
  moveResponses?: GameSnapshot[];
  moveDelayMs?: number;
}

export interface MockApi {
  createRequests: () => number;
  moveRequests: () => ReadonlyArray<{ origin: number; destination: number }>;
}

export async function installMockApi(page: Page, options: MockOptions = {}): Promise<MockApi> {
  const initial = options.initial ?? opening;
  const responses = options.moveResponses ?? [afterOpeningMove];
  const moves: Array<{ origin: number; destination: number }> = [];
  let creates = 0;

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (pathname === "/api/model" && request.method() === "GET") {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(model) });
      return;
    }
    if (pathname === "/api/games" && request.method() === "POST") {
      creates += 1;
      await route.fulfill({ status: 201, contentType: "application/json", body: JSON.stringify(initial) });
      return;
    }
    if (/^\/api\/games\/[^/]+\/moves$/.test(pathname) && request.method() === "POST") {
      const body = request.postDataJSON() as { origin: number; destination: number };
      moves.push(body);
      if (options.moveDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.moveDelayMs));
      }
      const response = responses[Math.min(moves.length - 1, responses.length - 1)];
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(response) });
      return;
    }
    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ error: `Unexpected fixture request: ${request.method()} ${pathname}` }),
    });
  });

  return {
    createRequests: () => creates,
    moveRequests: () => moves,
  };
}

export async function beginMatch(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByRole("button", { name: "Begin match" }).click();
  await page.getByRole("group", { name: /checkers board/i }).waitFor();
  await page.waitForFunction(() => {
    const board = document.querySelector(".board")?.getBoundingClientRect();
    return Boolean(board && board.top >= 0 && board.top < innerHeight);
  });
  await page.evaluate(
    () => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))),
  );
}
