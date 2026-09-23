import { CheckersNetwork, type PolicyManifest } from "@/engine/network";
import weightsUrl from "@/model/policy.bin?url";
import manifest from "@/model/policy.json";

function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Download, checksum, and validate the exported policy weights. */
export async function loadPolicyNetwork(fetchImpl: typeof fetch = fetch): Promise<CheckersNetwork> {
  const response = await fetchImpl(weightsUrl);
  if (!response.ok) throw new Error(`The model weights could not be downloaded (HTTP ${response.status}).`);
  const buffer = await response.arrayBuffer();
  const subtle = globalThis.crypto?.subtle;
  if (subtle && hex(await subtle.digest("SHA-256", buffer)) !== manifest.weightsSha256) {
    throw new Error("The downloaded model weights failed their integrity check.");
  }
  return CheckersNetwork.fromBuffer(buffer, manifest as PolicyManifest);
}
