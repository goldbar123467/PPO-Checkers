import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { GameStatus, MatchLedger } from "@/components/MatchLedger";
import { GameSession } from "@/lib/session";
import type { GameSnapshot } from "@/types";

function snapshot(overrides: Partial<GameSnapshot> = {}): GameSnapshot {
  const base = new GameSession({
    id: "g",
    humanColor: "red",
    policyMode: "greedy",
    seed: 1,
    maxPlies: 512,
    repetitionDraws: true,
  }).snapshot();
  return { ...base, ...overrides };
}

describe("game status and history", () => {
  it.each([
    [{ winner: "red", reason: "no_pieces", isDraw: false }, "You win! · no pieces remain"],
    [{ winner: "black", reason: "no_legal_move", isDraw: false }, "The AI wins · no legal move"],
    [{ winner: null, reason: "repetition", isDraw: true }, "Draw · threefold repetition"],
  ] as const)("announces %o", (outcome, text) => {
    render(<GameStatus game={snapshot({ outcome: { ...outcome } })} busy={false} />);
    expect(screen.getByRole("heading", { name: text })).toBeInTheDocument();
    expect(screen.getByText(/Game over after/)).toBeInTheDocument();
  });

  it("describes forced continuations and the AI's turn", () => {
    const { rerender } = render(<GameStatus game={snapshot({ captureInProgress: true })} busy={false} />);
    expect(screen.getByRole("heading", { name: /Keep jumping/ })).toBeInTheDocument();
    rerender(<GameStatus game={snapshot({ isHumanTurn: false })} busy={false} />);
    expect(screen.getByRole("heading", { name: /AI is choosing/ })).toBeInTheDocument();
  });

  it("lists moves with the mover's color", () => {
    render(
      <MatchLedger
        game={snapshot({ moves: [{ ply: 1, actor: "red", notation: "11-15" }] })}
      />,
    );
    expect(screen.getByText("1 move")).toBeInTheDocument();
    expect(screen.getByText("11-15")).toBeInTheDocument();
    expect(screen.getByText("Red")).toBeInTheDocument();
  });
});
