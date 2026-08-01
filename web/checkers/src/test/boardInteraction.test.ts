import { describe, expect, it } from "vitest";
import {
  beginPointerPress,
  boardOrder,
  cancelPointerPress,
  finishPointerPress,
  pointToBoardSquare,
} from "../boardInteraction";
import type { BoardCell } from "../types";

const board: BoardCell[] = Array.from({ length: 64 }, (_, index) => {
  const row = Math.floor(index / 8);
  const column = index % 8;
  const playable = (row + column) % 2 === 0;
  return {
    row,
    column,
    playable,
    square: playable ? boardSquare(row, column) : null,
  };
});

function boardSquare(row: number, column: number): number {
  return row * 4 + Math.floor((6 + (row % 2) - column) / 2);
}

describe("board pointer geometry", () => {
  it("uses one canonical ACF orientation for red and white perspectives", () => {
    expect(boardOrder("red")).toEqual({
      rows: [7, 6, 5, 4, 3, 2, 1, 0],
      columns: [0, 1, 2, 3, 4, 5, 6, 7],
    });
    expect(boardOrder("white")).toEqual({
      rows: [0, 1, 2, 3, 4, 5, 6, 7],
      columns: [7, 6, 5, 4, 3, 2, 1, 0],
    });
  });
  it.each([
    [320, 10, 10, 296],
    [512, 20, 40, 464],
    [800, 100, 200, 640],
  ])("maps the centers of playable cells at a %ipx board size", (size, left, top, width) => {
    const bounds = { left, top, width, height: width };
    const cell = width / 8;
    expect(pointToBoardSquare(left + cell * 1.5, top + cell / 2, bounds, board, "white")).toBe(0);
    expect(pointToBoardSquare(left + cell / 2, top + cell * 7.5, bounds, board, "white")).toBe(28);
    expect(pointToBoardSquare(left + cell * 1.5, top + cell / 2, bounds, board, "red")).toBe(31);
  });

  it("uses the destination side of an exact internal square boundary", () => {
    const bounds = { left: 0, top: 0, width: 320, height: 320 };
    expect(pointToBoardSquare(39.999, 0, bounds, board, "white")).toBeNull();
    expect(pointToBoardSquare(40, 0, bounds, board, "white")).toBe(0);
  });

  it("accepts coordinates just inside the board edges and rejects every outside edge", () => {
    const bounds = { left: 10, top: 20, width: 320, height: 320 };
    expect(pointToBoardSquare(50, 20, bounds, board, "white")).toBe(0);
    expect(pointToBoardSquare(10, 339.999, bounds, board, "white")).toBe(28);
    expect(pointToBoardSquare(9.999, 20, bounds, board, "white")).toBeNull();
    expect(pointToBoardSquare(10, 19.999, bounds, board, "white")).toBeNull();
    expect(pointToBoardSquare(330, 20, bounds, board, "white")).toBeNull();
    expect(pointToBoardSquare(10, 340, bounds, board, "white")).toBeNull();
  });

  it("rejects light, non-playable cells", () => {
    expect(
      pointToBoardSquare(20, 20, { left: 0, top: 0, width: 320, height: 320 }, board, "white"),
    ).toBeNull();
  });
});

describe("pointer press lifecycle", () => {
  it("activates exactly once when a pointer ends on the square where it began", () => {
    const pressed = beginPointerPress(null, 7, 12, false);
    expect(finishPointerPress(pressed, 7, 12, false)).toEqual({ next: null, activation: 12 });
    expect(finishPointerPress(null, 7, 12, false).activation).toBeNull();
  });

  it("rejects duplicate pointerdown events while a press is active", () => {
    const pressed = beginPointerPress(null, 7, 12, false);
    expect(beginPointerPress(pressed, 8, 13, false)).toBe(pressed);
  });

  it("rejects cancellation, release outside, release on another square, and stale pointer ids", () => {
    const pressed = beginPointerPress(null, 7, 12, false);
    expect(cancelPointerPress(pressed, 8)).toBe(pressed);
    expect(cancelPointerPress(pressed, 7)).toBeNull();
    expect(finishPointerPress(pressed, 8, 12, false)).toEqual({ next: pressed, activation: null });
    expect(finishPointerPress(pressed, 7, null, false).activation).toBeNull();
    expect(finishPointerPress(pressed, 7, 13, false).activation).toBeNull();
  });

  it("rejects presses and releases while input is locked", () => {
    expect(beginPointerPress(null, 7, 12, true)).toBeNull();
    expect(finishPointerPress({ pointerId: 7, square: 12 }, 7, 12, true).activation).toBeNull();
  });
});
