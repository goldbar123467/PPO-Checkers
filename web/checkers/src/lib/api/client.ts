import type { ZodType } from "zod";

import { apiErrorSchema } from "@/lib/api/schemas";

const CLIENT_TIMEOUT_MS = 16_000;

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

export class ApiProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiProtocolError";
  }
}

export async function requestJson<T>(
  schema: ZodType<T>,
  url: string,
  init?: RequestInit,
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), CLIENT_TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const raw: unknown = await response.json().catch(() => undefined);
    if (!response.ok) {
      const parsedError = apiErrorSchema.safeParse(raw);
      throw new ApiError(
        parsedError.success
          ? parsedError.data.error.message
          : `The policy server returned HTTP ${response.status}.`,
        parsedError.success ? parsedError.data.error.code : "http_error",
        response.status,
      );
    }
    const parsed = schema.safeParse(raw);
    if (!parsed.success) {
      throw new ApiProtocolError(
        `The policy server response did not match its documented schema: ${parsed.error.issues[0]?.message ?? "unknown mismatch"}`,
      );
    }
    return parsed.data;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(
        "The local policy took too long to answer. You can retry without losing your setup.",
        "client_timeout",
        408,
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}
