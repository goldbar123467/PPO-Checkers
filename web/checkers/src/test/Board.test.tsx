import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Board } from "../components/Board";
import type { BoardCell, GameSnapshot } from "../types";

function squareFor(row: number, column: number): number {
  return row * 4 + Math.floor((6 + (row % 2) - column) / 2);
}

const board: BoardCell[] = Array.from({ length: 64 }, (_, index) => {
  const row = Math.floor(index / 8);
  const column = index % 8;
  const playable = (row + column) % 2 === 0;
  return { row, column, playable, square: playable ? squareFor(row, column) : null };
});

function snapshot(overrides: Partial<GameSnapshot> = {}): GameSnapshot {
  return {
    id: "game-1",
    humanColor: "red",
    modelColor: "white",
    policyMode: "greedy",
    seed: 123,
    sideToMove: "red",
    isHumanTurn: true,
    captureInProgress: false,
    forcedSquare: null,
    ply: 0,
    board,
    pieces: [
      { square: 8, row: 2, column: 6, color: "red", kind: "man" },
      { square: 20, row: 5, column: 6, color: "white", kind: "man" },
    ],
    legalMoves: [{ action: 32, origin: 8, destination: 12, captured: null }],
    lastStep: null,
    moves: [],
    outcome: null,
    ...overrides,
  };
}

function pointer(
  target: Element,
  type: "pointerdown" | "pointerup" | "pointercancel",
  init: { pointerId: number; clientX?: number; clientY?: number },
) {
  const event = new MouseEvent(type, {
    bubbles: true,
    button: 0,
    clientX: init.clientX ?? 0,
    clientY: init.clientY ?? 0,
  });
  Object.defineProperties(event, {
    pointerId: { value: init.pointerId },
    isPrimary: { value: true },
  });
  fireEvent(target, event);
}

describe("Board interaction", () => {
  it("exposes and submits the same legal move through a non-pointer move list", () => {
    const onMove = vi.fn();
    render(<Board game={snapshot()} busy={false} onMove={onMove} />);

    expect(screen.getByRole("heading", { name: "Legal move list" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /ACF 9 to 13.*legal step/i }));
    expect(onMove).toHaveBeenCalledOnce();
    expect(onMove).toHaveBeenCalledWith(8, 12);
  });

  it("selects legal pieces, rejects illegal choices, clears selection, and submits a legal destination", () => {
    const onMove = vi.fn();
    render(
      <Board
        game={snapshot({
          pieces: [
            { square: 8, row: 2, column: 6, color: "red", kind: "man" },
            { square: 9, row: 2, column: 4, color: "red", kind: "man" },
            { square: 20, row: 5, column: 6, color: "white", kind: "man" },
          ],
          legalMoves: [
            { action: 32, origin: 8, destination: 12, captured: null },
            { action: 37, origin: 9, destination: 13, captured: null },
          ],
        })}
        busy={false}
        onMove={onMove}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /square 21, white man/i }));
    expect(onMove).not.toHaveBeenCalled();
    expect(screen.getByText("Square 21 cannot move in this position.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /square 9, orange man, movable/i }));
    expect(screen.getByRole("button", { name: /square 9, orange man, selected/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    fireEvent.click(screen.getByRole("button", { name: /square 14, empty/i }));
    expect(onMove).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /square 10, orange man, movable/i }));
    expect(screen.getByRole("button", { name: /square 10, orange man, selected/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    fireEvent.click(screen.getByRole("button", { name: /clear selection/i }));
    expect(screen.queryByRole("button", { name: /clear selection/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /square 9, orange man, movable/i }));
    fireEvent.click(screen.getByRole("button", { name: /square 13, empty, legal destination/i }));
    expect(onMove).toHaveBeenCalledOnce();
    expect(onMove).toHaveBeenCalledWith(8, 12);
  });

  it("keeps the forced piece selected through a multi-jump continuation", () => {
    const onMove = vi.fn();
    const forced = snapshot({
      captureInProgress: true,
      forcedSquare: 12,
      pieces: [{ square: 12, row: 3, column: 7, color: "red", kind: "man" }],
      legalMoves: [{ action: 50, origin: 12, destination: 21, captured: 17 }],
    });
    const { rerender } = render(<Board game={forced} busy={false} onMove={onMove} />);

    const forcedPiece = screen.getByRole("button", { name: /square 13, orange man, selected/i });
    fireEvent.click(forcedPiece);
    expect(forcedPiece).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("button", { name: /clear selection/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /square 22, empty, legal destination/i }));
    expect(onMove).toHaveBeenCalledWith(12, 21);

    rerender(
      <Board
        game={snapshot({
          ply: 1,
          captureInProgress: true,
          forcedSquare: 21,
          pieces: [{ square: 21, row: 5, column: 4, color: "red", kind: "man" }],
          legalMoves: [{ action: 84, origin: 21, destination: 30, captured: 26 }],
        })}
        busy={false}
        onMove={onMove}
      />,
    );
    expect(screen.getByRole("button", { name: /square 22, orange man, selected/i })).toBeInTheDocument();
  });

  it("exposes promotion, turn lock, and game-over lock without accepting input", () => {
    const onMove = vi.fn();
    const { rerender } = render(
      <Board
        game={snapshot({
          pieces: [{ square: 28, row: 7, column: 7, color: "red", kind: "king" }],
          legalMoves: [{ action: 112, origin: 28, destination: 24, captured: null }],
        })}
        busy={false}
        onMove={onMove}
      />,
    );
    expect(screen.getByRole("button", { name: /square 29, orange king, movable/i })).toBeInTheDocument();

    rerender(
      <Board
        game={snapshot({ isHumanTurn: false, sideToMove: "white" })}
        busy={false}
        onMove={onMove}
      />,
    );
    expect(screen.getByRole("button", { name: /square 9, orange man, movable/i })).toBeDisabled();

    rerender(
      <Board
        game={snapshot({ outcome: { winner: "red", reason: "no_moves", isDraw: false } })}
        busy={false}
        onMove={onMove}
      />,
    );
    expect(screen.getByRole("button", { name: /square 9, orange man, movable/i })).toBeDisabled();
    expect(onMove).not.toHaveBeenCalled();
  });

  it("supports roving keyboard focus, selection, move confirmation, and Escape", () => {
    const onMove = vi.fn();
    render(<Board game={snapshot()} busy={false} onMove={onMove} />);
    const origin = screen.getByRole("button", { name: /square 9, orange man, movable/i });
    origin.focus();
    fireEvent.keyDown(origin, { key: "Home" });
    expect(document.activeElement).toHaveAttribute("data-square-index", "31");
    origin.focus();
    fireEvent.keyDown(origin, { key: "Enter" });
    fireEvent.click(origin, { detail: 0 });
    expect(screen.getByRole("button", { name: /clear selection/i })).toBeInTheDocument();
    fireEvent.keyDown(origin, { key: "Escape" });
    expect(screen.queryByRole("button", { name: /clear selection/i })).not.toBeInTheDocument();

    fireEvent.click(origin, { detail: 0 });
    fireEvent.keyDown(origin, { key: "ArrowUp" });
    expect(document.activeElement).toHaveAttribute("data-square-index", "13");
    fireEvent.keyDown(document.activeElement as HTMLElement, { key: "ArrowRight" });
    expect(document.activeElement).toHaveAttribute("data-square-index", "12");
    const destination = document.activeElement as HTMLButtonElement;
    fireEvent.click(destination, { detail: 0 });
    expect(onMove).toHaveBeenCalledOnce();
  });

  it("rejects pointer cancellation, outside release, and the synthetic click after a valid tap", () => {
    const onMove = vi.fn();
    render(<Board game={snapshot()} busy={false} onMove={onMove} />);
    const group = screen.getByRole("group", { name: /orange's side/i });
    vi.spyOn(group, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 320,
      bottom: 320,
      width: 320,
      height: 320,
      toJSON: () => ({}),
    });

    pointer(group, "pointerdown", { pointerId: 1, clientX: 260, clientY: 220 });
    pointer(group, "pointercancel", { pointerId: 1 });
    pointer(group, "pointerup", { pointerId: 1, clientX: 260, clientY: 220 });
    expect(screen.queryByRole("button", { name: /square 9, orange man, selected/i })).not.toBeInTheDocument();

    pointer(group, "pointerdown", { pointerId: 2, clientX: 260, clientY: 220 });
    pointer(group, "pointerup", { pointerId: 2, clientX: -1, clientY: 220 });
    expect(screen.queryByRole("button", { name: /square 9, orange man, selected/i })).not.toBeInTheDocument();

    pointer(group, "pointerdown", { pointerId: 3, clientX: 260, clientY: 220 });
    pointer(group, "pointerup", { pointerId: 3, clientX: 260, clientY: 220 });
    pointer(group, "pointerdown", { pointerId: 4, clientX: 300, clientY: 180 });
    pointer(group, "pointerup", { pointerId: 4, clientX: 300, clientY: 180 });
    const destination = screen.getByRole("button", { name: /square 13, empty, legal destination/i });
    fireEvent.click(destination, { detail: 1 });
    expect(onMove).toHaveBeenCalledOnce();
  });

  it("rejects pointer activation while the board is busy", () => {
    const onMove = vi.fn();
    render(<Board game={snapshot()} busy onMove={onMove} />);
    const group = screen.getByRole("group", { name: /orange's side/i });
    vi.spyOn(group, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 320,
      bottom: 320,
      width: 320,
      height: 320,
      toJSON: () => ({}),
    });
    pointer(group, "pointerdown", { pointerId: 1, clientX: 260, clientY: 220 });
    pointer(group, "pointerup", { pointerId: 1, clientX: 260, clientY: 220 });
    expect(onMove).not.toHaveBeenCalled();
  });

  it("rejects a stale pointer press after the position advances", () => {
    const onMove = vi.fn();
    const { rerender } = render(<Board game={snapshot()} busy={false} onMove={onMove} />);
    const group = screen.getByRole("group", { name: /orange's side/i });
    vi.spyOn(group, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 320,
      bottom: 320,
      width: 320,
      height: 320,
      toJSON: () => ({}),
    });

    pointer(group, "pointerdown", { pointerId: 11, clientX: 260, clientY: 220 });
    rerender(<Board game={snapshot({ ply: 2 })} busy={false} onMove={onMove} />);
    pointer(group, "pointerup", { pointerId: 11, clientX: 260, clientY: 220 });

    expect(screen.queryByRole("button", { name: /square 9, orange man, selected/i })).not.toBeInTheDocument();
    expect(onMove).not.toHaveBeenCalled();
  });
});
