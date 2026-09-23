export type WorkerRequest =
  | { type: "load" }
  | { type: "evaluate"; id: number; observation: Float32Array };

export type WorkerResponse =
  | { type: "ready" }
  | { type: "result"; id: number; logits: Float32Array; value: number; inferenceMs: number }
  | { type: "error"; id: number | null; message: string };
