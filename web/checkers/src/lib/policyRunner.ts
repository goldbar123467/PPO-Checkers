import type { WorkerRequest, WorkerResponse } from "@/lib/policyProtocol";
import type { Evaluation } from "@/lib/session";

/** Runs the policy network somewhere that will not block the page. */
export interface PolicyRunner {
  evaluate(observation: Float32Array): Promise<Evaluation>;
  dispose(): void;
}

interface Pending {
  resolve: (evaluation: Evaluation) => void;
  reject: (error: Error) => void;
}

function startWorker(): Promise<PolicyRunner> {
  const worker = new Worker(new URL("./policy.worker.ts", import.meta.url), { type: "module" });
  const pending = new Map<number, Pending>();
  let nextId = 0;

  return new Promise<PolicyRunner>((resolve, reject) => {
    const fail = (error: Error) => {
      reject(error);
      pending.forEach(({ reject: rejectPending }) => rejectPending(error));
      pending.clear();
    };
    worker.onerror = (event) => {
      event.preventDefault();
      fail(new Error(event.message || "The policy worker stopped unexpectedly."));
    };
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.type === "ready") {
        resolve({
          evaluate(observation) {
            const id = nextId++;
            return new Promise<Evaluation>((resolveEvaluation, rejectEvaluation) => {
              pending.set(id, { resolve: resolveEvaluation, reject: rejectEvaluation });
              const request: WorkerRequest = { type: "evaluate", id, observation };
              worker.postMessage(request, [observation.buffer]);
            });
          },
          dispose() {
            worker.terminate();
            fail(new Error("The policy worker was stopped."));
          },
        });
        return;
      }
      if (message.type === "result") {
        pending.get(message.id)?.resolve(message);
        pending.delete(message.id);
        return;
      }
      if (message.id === null) {
        worker.terminate();
        fail(new Error(message.message));
        return;
      }
      pending.get(message.id)?.reject(new Error(message.message));
      pending.delete(message.id);
    };
    const load: WorkerRequest = { type: "load" };
    worker.postMessage(load);
  });
}

async function startInPage(): Promise<PolicyRunner> {
  const { loadPolicyNetwork } = await import("@/lib/loadPolicy");
  const network = await loadPolicyNetwork();
  return {
    async evaluate(observation) {
      const started = performance.now();
      const { logits, value } = network.forward(observation);
      return { logits, value, inferenceMs: performance.now() - started };
    },
    dispose() {},
  };
}

/** Load the policy in a Web Worker, falling back to the page thread where workers are unavailable. */
export function createPolicyRunner(): Promise<PolicyRunner> {
  if (typeof Worker === "undefined") return startInPage();
  try {
    return startWorker();
  } catch {
    return startInPage();
  }
}
