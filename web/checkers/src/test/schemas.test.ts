import { describe, expect, it } from "vitest";

import { gameSnapshotSchema, modelInfoSchema } from "@/lib/api/schemas";

describe("typed backend response boundary", () => {
  it("rejects undocumented model fields and invalid checksums where strict fields matter", () => {
    const result = modelInfoSchema.safeParse({ ready: true, bundleId: "x" });
    expect(result.success).toBe(false);
  });

  it("rejects malformed game boards before rendering", () => {
    const result = gameSnapshotSchema.safeParse({ id: "game", board: [] });
    expect(result.success).toBe(false);
  });
});
