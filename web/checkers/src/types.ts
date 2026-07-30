export type Color = "red" | "white";
export type PolicyMode = "greedy" | "sampled";
export type PieceKind = "man" | "king";

export interface ModelInfo {
  ready: boolean;
  bundleId: string;
  experimentId: string;
  update: number;
  globalStep: number;
  sourceCheckpoint: string;
  sourceCheckpointSha256: string;
  bundleSha256: string;
  bundleSizeBytes: number;
  gitSha: string;
  gitDirty: boolean;
  device: "cpu";
  actionCount: number;
  maxPlies: number;
  repetitionDraws: boolean;
  parameterCount: number;
}

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
  reason: string;
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
