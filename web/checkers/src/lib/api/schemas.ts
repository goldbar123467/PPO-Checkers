import { z } from "zod";

const safeInteger = z.number().int().safe();
const square = z.number().int().min(0).max(31);

export const colorSchema = z.enum(["red", "white"]);
export const policyModeSchema = z.enum(["greedy", "sampled"]);

export const healthSchema = z.object({
  status: z.literal("ok"),
  modelReady: z.boolean(),
});

export const modelInfoSchema = z.object({
  ready: z.boolean(),
  bundleId: z.string().min(1),
  experimentId: z.string().min(1),
  update: safeInteger.nonnegative(),
  globalStep: safeInteger.nonnegative(),
  sourceCheckpoint: z.string().min(1),
  sourceCheckpointSha256: z.string().regex(/^[a-f0-9]{64}$/),
  bundleSha256: z.string().regex(/^[a-f0-9]{64}$/),
  bundleSizeBytes: safeInteger.positive(),
  gitSha: z.string().min(7),
  gitDirty: z.boolean(),
  device: z.literal("cpu"),
  actionCount: safeInteger.positive(),
  maxPlies: safeInteger.positive(),
  repetitionDraws: z.boolean(),
  parameterCount: safeInteger.positive(),
});

export const boardCellSchema = z.object({
  row: z.number().int().min(0).max(7),
  column: z.number().int().min(0).max(7),
  playable: z.boolean(),
  square: square.nullable(),
});

export const pieceSchema = z.object({
  square,
  row: z.number().int().min(0).max(7),
  column: z.number().int().min(0).max(7),
  color: colorSchema,
  kind: z.enum(["man", "king"]),
});

export const legalMoveSchema = z.object({
  action: z.number().int().min(0).max(127),
  origin: square,
  destination: square,
  captured: square.nullable(),
});

export const moveRecordSchema = z.object({
  ply: safeInteger.positive(),
  actor: colorSchema,
  notation: z.string().min(1),
});

export const outcomeSchema = z.object({
  winner: colorSchema.nullable(),
  reason: z.string().min(1),
  isDraw: z.boolean(),
});

export const gameSnapshotSchema = z.object({
  id: z.string().min(1),
  humanColor: colorSchema,
  modelColor: colorSchema,
  policyMode: policyModeSchema,
  seed: safeInteger.nonnegative(),
  sideToMove: colorSchema,
  isHumanTurn: z.boolean(),
  captureInProgress: z.boolean(),
  forcedSquare: square.nullable(),
  ply: safeInteger.nonnegative(),
  board: z.array(boardCellSchema).length(64),
  pieces: z.array(pieceSchema).max(24),
  legalMoves: z.array(legalMoveSchema),
  lastStep: z.object({ origin: square, destination: square }).nullable(),
  moves: z.array(moveRecordSchema),
  outcome: outcomeSchema.nullable(),
});

export const apiErrorSchema = z.object({
  error: z.object({
    code: z.string().min(1),
    message: z.string().min(1),
  }),
});

export type Color = z.infer<typeof colorSchema>;
export type PolicyMode = z.infer<typeof policyModeSchema>;
export type Health = z.infer<typeof healthSchema>;
export type ModelInfo = z.infer<typeof modelInfoSchema>;
export type BoardCell = z.infer<typeof boardCellSchema>;
export type Piece = z.infer<typeof pieceSchema>;
export type LegalMove = z.infer<typeof legalMoveSchema>;
export type MoveRecord = z.infer<typeof moveRecordSchema>;
export type Outcome = z.infer<typeof outcomeSchema>;
export type GameSnapshot = z.infer<typeof gameSnapshotSchema>;
