import { useEffect, useMemo, useState } from "react";
import type { BoardCell, Color, GameSnapshot, LegalMove, Piece } from "../types";

interface BoardProps {
  game: GameSnapshot;
  busy: boolean;
  onMove: (origin: number, destination: number) => void;
}

function cellKey(cell: BoardCell): string {
  return `${cell.row}:${cell.column}`;
}

function pieceName(piece: Piece): string {
  return `${piece.color} ${piece.kind}`;
}

function boardOrder(color: Color): { rows: number[]; columns: number[] } {
  const ascending = [0, 1, 2, 3, 4, 5, 6, 7];
  const descending = [...ascending].reverse();
  return color === "red"
    ? { rows: descending, columns: ascending }
    : { rows: ascending, columns: descending };
}

export function Board({ game, busy, onMove }: BoardProps) {
  const [selected, setSelected] = useState<number | null>(null);
  const legalOrigins = useMemo(
    () => new Set(game.legalMoves.map((move) => move.origin)),
    [game.legalMoves],
  );
  const destinations = useMemo(
    () =>
      new Map(
        game.legalMoves
          .filter((move) => move.origin === selected)
          .map((move) => [move.destination, move]),
      ),
    [game.legalMoves, selected],
  );
  const pieces = useMemo(
    () => new Map(game.pieces.map((piece) => [piece.square, piece])),
    [game.pieces],
  );
  const cells = useMemo(
    () => new Map(game.board.map((cell) => [cellKey(cell), cell])),
    [game.board],
  );
  const order = boardOrder(game.humanColor);

  useEffect(() => {
    if (game.forcedSquare !== null) {
      setSelected(game.forcedSquare);
    } else if (selected !== null && !legalOrigins.has(selected)) {
      setSelected(null);
    }
  }, [game.forcedSquare, legalOrigins, selected]);

  function chooseSquare(square: number) {
    if (busy || !game.isHumanTurn || game.outcome) return;
    const destination = destinations.get(square);
    if (selected !== null && destination) {
      onMove(selected, destination.destination);
      return;
    }
    if (legalOrigins.has(square)) {
      setSelected(square === selected ? null : square);
    }
  }

  const displayCells = order.rows.flatMap((row) =>
    order.columns.map((column) => cells.get(`${row}:${column}`)),
  );

  return (
    <div className="board-shell">
      <div
        className={`board board--${game.humanColor}`}
        role="group"
        aria-label={`Checkers board from ${game.humanColor}'s side`}
        aria-busy={busy}
      >
        {displayCells.map((cell) => {
          if (!cell) return null;
          if (!cell.playable || cell.square === null) {
            return <span className="cell cell--light" aria-hidden="true" key={cellKey(cell)} />;
          }
          const square = cell.square;
          const piece = pieces.get(square);
          const legalOrigin = legalOrigins.has(square);
          const legalDestination = destinations.has(square);
          const selectedCell = square === selected;
          const last =
            game.lastStep?.origin === square || game.lastStep?.destination === square;
          const labels = [
            `Square ${square + 1}`,
            piece ? pieceName(piece) : "empty",
            selectedCell ? "selected" : "",
            legalOrigin ? "movable" : "",
            legalDestination ? "legal destination" : "",
          ].filter(Boolean);
          return (
            <button
              className={[
                "cell",
                "cell--dark",
                selectedCell ? "is-selected" : "",
                legalOrigin ? "is-origin" : "",
                legalDestination ? "is-destination" : "",
                last ? "is-last" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              type="button"
              aria-label={labels.join(", ")}
              aria-pressed={selectedCell}
              disabled={busy || !game.isHumanTurn || Boolean(game.outcome)}
              onClick={() => chooseSquare(square)}
              key={cellKey(cell)}
              data-square={square + 1}
            >
              <span className="square-number" aria-hidden="true">
                {square + 1}
              </span>
              {piece && (
                <span className={`piece piece--${piece.color} piece--${piece.kind}`}>
                  {piece.kind === "king" && (
                    <span className="piece__crown" aria-hidden="true">
                      ♛
                    </span>
                  )}
                </span>
              )}
              {legalDestination && <span className="destination-dot" aria-hidden="true" />}
            </button>
          );
        })}
      </div>
      <div className="board-caption" aria-hidden="true">
        <span>{game.humanColor === "red" ? "Red house" : "White house"}</span>
        <span>American checkers · ACF 1–32</span>
      </div>
    </div>
  );
}
