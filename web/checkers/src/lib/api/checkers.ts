import { requestJson } from "@/lib/api/client";
import {
  gameSnapshotSchema,
  healthSchema,
  modelInfoSchema,
  type Color,
  type GameSnapshot,
  type Health,
  type ModelInfo,
  type PolicyMode,
} from "@/lib/api/schemas";

export function fetchHealth(): Promise<Health> {
  return requestJson(healthSchema, "/api/health");
}

export function fetchModel(): Promise<ModelInfo> {
  return requestJson(modelInfoSchema, "/api/model");
}

export function createGame(
  humanColor: Color,
  policyMode: PolicyMode,
  seed: number,
): Promise<GameSnapshot> {
  return requestJson(gameSnapshotSchema, "/api/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ humanColor, policyMode, seed }),
  });
}

export function fetchGame(gameId: string): Promise<GameSnapshot> {
  return requestJson(gameSnapshotSchema, `/api/games/${encodeURIComponent(gameId)}`);
}

export function submitMove(
  gameId: string,
  origin: number,
  destination: number,
): Promise<GameSnapshot> {
  return requestJson(
    gameSnapshotSchema,
    `/api/games/${encodeURIComponent(gameId)}/moves`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin, destination }),
    },
  );
}
