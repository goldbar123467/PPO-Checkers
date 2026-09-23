import type { PolicyMode } from "@/engine/policy";
import type { TerminationReason } from "@/engine/rules";

export type { PolicyMode, TerminationReason };

/** Red moves first. The Python engine calls the second player White; the site shows Black. */
export type Color = "red" | "black";
export type PieceKind = "man" | "king";

export interface BoardCell {
  row: number;
  column: number;
  playable: boolean;
  square: number | null;
}

export interface Piece {
  square: number;
  row: number;
  column: number;
  color: Color;
  kind: PieceKind;
}

export interface LegalMove {
  action: number;
  origin: number;
  destination: number;
  captured: number | null;
}

export interface MoveRecord {
  ply: number;
  actor: Color;
  notation: string;
}

export interface Outcome {
  winner: Color | null;
  reason: TerminationReason;
  isDraw: boolean;
}

export interface GameSnapshot {
  id: string;
  humanColor: Color;
  modelColor: Color;
  policyMode: PolicyMode;
  seed: number;
  sideToMove: Color;
  isHumanTurn: boolean;
  captureInProgress: boolean;
  forcedSquare: number | null;
  ply: number;
  board: BoardCell[];
  pieces: Piece[];
  legalMoves: LegalMove[];
  lastStep: { origin: number; destination: number } | null;
  moves: MoveRecord[];
  outcome: Outcome | null;
}

export interface ModelInfo {
  update: number;
  globalStep: number;
  parameterCount: number;
  weightsSizeBytes: number;
  sourceBundleSha256: string;
}

/** What the network reported for the move it just played. */
export interface PolicyInsight {
  /** Value-head estimate from the model's side, in [-1, 1]. Not a calibrated probability. */
  value: number;
  /** Softmax probability of the chosen move among the legal moves. */
  confidence: number;
  legalMoveCount: number;
  inferenceMs: number;
}
