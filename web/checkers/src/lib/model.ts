import {
  globalStep,
  maxPlies,
  parameterCount,
  repetitionDraws,
  sourceBundleSha256,
  update,
  weightsSizeBytes,
} from "@/model/policy.json";
import type { ModelInfo } from "@/types";

/** Provenance of the policy shipped with the site, read from the exported manifest. */
export const MODEL_INFO: ModelInfo = {
  update,
  globalStep,
  parameterCount,
  weightsSizeBytes,
  sourceBundleSha256,
};

/** Game rules the policy was trained and evaluated under. */
export const GAME_RULES = { maxPlies, repetitionDraws } as const;
