import type { Color, GameSnapshot, ModelInfo, PolicyMode } from "./types";

interface ErrorPayload {
  error?: { code?: string; message?: string };
}

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let payload: ErrorPayload = {};
    try {
      payload = (await response.json()) as ErrorPayload;
    } catch {
      // Preserve the HTTP-level fallback when a nonconforming proxy responds.
    }
    throw new ApiError(
      payload.error?.message ?? `Policy server returned HTTP ${response.status}`,
      payload.error?.code ?? "http_error",
      response.status,
    );
  }
  return (await response.json()) as T;
}

export function fetchModel(): Promise<ModelInfo> {
  return request<ModelInfo>("/api/model");
}

export function createGame(
  humanColor: Color,
  policyMode: PolicyMode,
  seed: number,
): Promise<GameSnapshot> {
  return request<GameSnapshot>("/api/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ humanColor, policyMode, seed }),
  });
}

export function submitMove(
  gameId: string,
  origin: number,
  destination: number,
): Promise<GameSnapshot> {
  return request<GameSnapshot>(`/api/games/${encodeURIComponent(gameId)}/moves`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ origin, destination }),
  });
}
