/// <reference lib="webworker" />
import type { CheckersNetwork } from "@/engine/network";
import { loadPolicyNetwork } from "@/lib/loadPolicy";
import type { WorkerRequest, WorkerResponse } from "@/lib/policyProtocol";

declare const self: DedicatedWorkerGlobalScope;

let network: CheckersNetwork | null = null;

function reply(message: WorkerResponse, transfer: Transferable[] = []) {
  self.postMessage(message, transfer);
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : "The policy worker failed.";
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === "load") {
    try {
      network = await loadPolicyNetwork();
      reply({ type: "ready" });
    } catch (error) {
      reply({ type: "error", id: null, message: describe(error) });
    }
    return;
  }
  try {
    if (!network) throw new Error("The policy is not loaded yet.");
    const started = performance.now();
    const { logits, value } = network.forward(request.observation);
    const inferenceMs = performance.now() - started;
    reply({ type: "result", id: request.id, logits, value, inferenceMs }, [logits.buffer]);
  } catch (error) {
    reply({ type: "error", id: request.id, message: describe(error) });
  }
};
