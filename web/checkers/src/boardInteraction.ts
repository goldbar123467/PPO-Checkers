import type { BoardCell, Color } from "./types";

export interface BoardBounds {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface PointerPress {
  pointerId: number;
  square: number;
}

export interface PointerRelease {
  next: PointerPress | null;
  activation: number | null;
}

export function boardOrder(color: Color): { rows: number[]; columns: number[] } {
  const ascending = [0, 1, 2, 3, 4, 5, 6, 7];
  const descending = [...ascending].reverse();
  return color === "red"
    ? { rows: descending, columns: ascending }
    : { rows: ascending, columns: descending };
}

export function pointToBoardSquare(
  clientX: number,
  clientY: number,
  bounds: BoardBounds,
  board: BoardCell[],
  perspective: Color,
): number | null {
  if (
    bounds.width <= 0 ||
    bounds.height <= 0 ||
    clientX < bounds.left ||
    clientY < bounds.top ||
    clientX >= bounds.left + bounds.width ||
    clientY >= bounds.top + bounds.height
  ) {
    return null;
  }

  const displayColumn = Math.min(7, Math.floor(((clientX - bounds.left) / bounds.width) * 8));
  const displayRow = Math.min(7, Math.floor(((clientY - bounds.top) / bounds.height) * 8));
  const order = boardOrder(perspective);
  const row = order.rows[displayRow];
  const column = order.columns[displayColumn];
  return board.find((cell) => cell.row === row && cell.column === column)?.square ?? null;
}

export function beginPointerPress(
  current: PointerPress | null,
  pointerId: number,
  square: number | null,
  locked: boolean,
): PointerPress | null {
  if (current || locked || square === null) return current;
  return { pointerId, square };
}

export function cancelPointerPress(
  current: PointerPress | null,
  pointerId: number,
): PointerPress | null {
  return current?.pointerId === pointerId ? null : current;
}

export function finishPointerPress(
  current: PointerPress | null,
  pointerId: number,
  square: number | null,
  locked: boolean,
): PointerRelease {
  if (!current || current.pointerId !== pointerId) {
    return { next: current, activation: null };
  }
  return {
    next: null,
    activation: !locked && square === current.square ? square : null,
  };
}
