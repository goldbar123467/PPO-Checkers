import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";
import {
  beginPointerPress,
  boardOrder,
  cancelPointerPress,
  finishPointerPress,
  pointToBoardSquare,
  type PointerPress,
} from "../boardInteraction";
import type { BoardCell, GameSnapshot, Piece } from "../types";

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

export function Board({ game, busy, onMove }: BoardProps) {
  const [selected, setSelected] = useState<number | null>(null);
  const [focusedSquare, setFocusedSquare] = useState<number | null>(null);
  const [feedback, setFeedback] = useState("Use arrow keys to move between dark squares.");
  const buttonRefs = useRef(new Map<number, HTMLButtonElement>());
  const pressRef = useRef<PointerPress | null>(null);
  const suppressClickRef = useRef<number | null>(null);
  const moveDispatchedRef = useRef(false);
  const previousBusyRef = useRef(busy);

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
  const displayCells = order.rows.flatMap((row) =>
    order.columns.map((column) => cells.get(`${row}:${column}`)),
  );
  const playableSquares = displayCells.flatMap((cell) =>
    cell?.playable && cell.square !== null ? [cell.square] : [],
  );
  const defaultFocusedSquare =
    game.forcedSquare ?? legalOrigins.values().next().value ?? playableSquares[0] ?? null;
  const rovingSquare = focusedSquare ?? defaultFocusedSquare;
  const locked = busy || !game.isHumanTurn || Boolean(game.outcome) || moveDispatchedRef.current;

  useEffect(() => {
    if (game.forcedSquare !== null) {
      setSelected(game.forcedSquare);
      setFocusedSquare(game.forcedSquare);
    } else if (selected !== null && !legalOrigins.has(selected)) {
      setSelected(null);
    }
  }, [game.forcedSquare, legalOrigins, selected]);

  useEffect(() => {
    moveDispatchedRef.current = false;
    pressRef.current = null;
  }, [game.id, game.ply, game.forcedSquare, game.sideToMove]);

  useEffect(() => {
    if (previousBusyRef.current && !busy) moveDispatchedRef.current = false;
    previousBusyRef.current = busy;
  }, [busy]);

  function chooseSquare(square: number) {
    if (busy || moveDispatchedRef.current) {
      setFeedback("Please wait for the current move to finish.");
      return;
    }
    if (game.outcome) {
      setFeedback("The game is over. Start a new match to play again.");
      return;
    }
    if (!game.isHumanTurn) {
      setFeedback("Please wait for the neural policy to finish its turn.");
      return;
    }

    const destination = destinations.get(square);
    if (selected !== null && destination) {
      moveDispatchedRef.current = true;
      setFeedback(`Moving from square ${selected + 1} to square ${square + 1}.`);
      onMove(selected, destination.destination);
      return;
    }

    if (legalOrigins.has(square)) {
      if (game.forcedSquare === square) {
        setSelected(square);
        setFeedback(`Capture must continue from square ${square + 1}.`);
      } else if (square === selected) {
        setSelected(null);
        setFeedback("Selection cleared.");
      } else {
        setSelected(square);
        setFeedback(`Square ${square + 1} selected. Choose a marked destination.`);
      }
      return;
    }

    const piece = pieces.get(square);
    setFeedback(
      piece
        ? `Square ${square + 1} cannot move in this position.`
        : `Square ${square + 1} is not a legal destination.`,
    );
  }

  function clearSelection() {
    if (game.forcedSquare !== null) {
      setFeedback(`Capture must continue from square ${game.forcedSquare + 1}.`);
      return;
    }
    setSelected(null);
    setFeedback("Selection cleared.");
  }

  function squareAtPointer(event: PointerEvent<HTMLDivElement>): number | null {
    const outerBounds = event.currentTarget.getBoundingClientRect();
    const style = getComputedStyle(event.currentTarget);
    const cssPixels = (value: string) => {
      const parsed = Number.parseFloat(value);
      return Number.isFinite(parsed) ? parsed : 0;
    };
    const borderLeft = cssPixels(style.borderLeftWidth);
    const borderRight = cssPixels(style.borderRightWidth);
    const borderTop = cssPixels(style.borderTopWidth);
    const borderBottom = cssPixels(style.borderBottomWidth);
    const bounds = {
      left: outerBounds.left + borderLeft,
      top: outerBounds.top + borderTop,
      width: outerBounds.width - borderLeft - borderRight,
      height: outerBounds.height - borderTop - borderBottom,
    };
    return pointToBoardSquare(
      event.clientX,
      event.clientY,
      bounds,
      game.board,
      game.humanColor,
    );
  }

  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.isPrimary === false || event.button !== 0) return;
    suppressClickRef.current = null;
    const square = squareAtPointer(event);
    const next = beginPointerPress(pressRef.current, event.pointerId, square, locked);
    if (next !== pressRef.current) {
      pressRef.current = next;
      event.currentTarget.setPointerCapture?.(event.pointerId);
    }
  }

  function onPointerUp(event: PointerEvent<HTMLDivElement>) {
    if (event.isPrimary === false || event.button !== 0) return;
    const result = finishPointerPress(
      pressRef.current,
      event.pointerId,
      squareAtPointer(event),
      locked,
    );
    pressRef.current = result.next;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (result.activation !== null) {
      suppressClickRef.current = result.activation;
      chooseSquare(result.activation);
    }
  }

  function onPointerCancel(event: PointerEvent<HTMLDivElement>) {
    pressRef.current = cancelPointerPress(pressRef.current, event.pointerId);
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function focusArrow(square: number, key: "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight") {
    const index = displayCells.findIndex((cell) => cell?.square === square);
    if (index < 0) return;
    const row = Math.floor(index / 8);
    const column = index % 8;
    const candidates = displayCells.flatMap((cell, candidateIndex) => {
      if (!cell?.playable || cell.square === null) return [];
      return [{ square: cell.square, row: Math.floor(candidateIndex / 8), column: candidateIndex % 8 }];
    });
    const next =
      key === "ArrowLeft" || key === "ArrowRight"
        ? candidates
            .filter((candidate) => candidate.row === row)
            .filter((candidate) =>
              key === "ArrowLeft" ? candidate.column < column : candidate.column > column,
            )
            .sort((a, b) =>
              key === "ArrowLeft" ? b.column - a.column : a.column - b.column,
            )[0]
        : candidates
            .filter((candidate) => candidate.row === row + (key === "ArrowUp" ? -1 : 1))
            .sort(
              (a, b) =>
                Math.abs(a.column - column) - Math.abs(b.column - column) ||
                a.column - b.column,
            )[0];
    if (next) {
      setFocusedSquare(next.square);
      buttonRefs.current.get(next.square)?.focus();
    }
  }

  function onSquareKeyDown(event: KeyboardEvent<HTMLButtonElement>, square: number) {
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
      event.preventDefault();
      focusArrow(square, event.key as "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight");
      return;
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      const next = event.key === "Home" ? playableSquares[0] : playableSquares.at(-1);
      if (next !== undefined) {
        setFocusedSquare(next);
        buttonRefs.current.get(next)?.focus();
      }
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      clearSelection();
    }
  }

  function onSquareClick(square: number) {
    if (suppressClickRef.current === square) {
      suppressClickRef.current = null;
      return;
    }
    chooseSquare(square);
  }

  return (
    <div className="board-shell">
      <div className="board-tools">
        <p id="board-instructions">
          Tap a ringed piece, then a marked square. Keyboard: arrows, Enter or Space, Escape.
        </p>
        {selected !== null && game.forcedSquare === null && (
          <button type="button" className="text-action" onClick={clearSelection}>
            Clear selection
          </button>
        )}
      </div>
      <div
        className={`board board--${game.humanColor}`}
        role="group"
        aria-label={`Checkers board from ${game.humanColor}'s side`}
        aria-describedby="board-instructions board-feedback"
        aria-busy={busy}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
        onContextMenu={(event) => event.preventDefault()}
        onDragStart={(event) => event.preventDefault()}
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
              tabIndex={square === rovingSquare ? 0 : -1}
              onFocus={() => setFocusedSquare(square)}
              onClick={() => onSquareClick(square)}
              onKeyDown={(event) => onSquareKeyDown(event, square)}
              ref={(node) => {
                if (node) buttonRefs.current.set(square, node);
                else buttonRefs.current.delete(square);
              }}
              key={cellKey(cell)}
              data-square-index={square}
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
      <p className="sr-only" id="board-feedback" aria-live="polite" aria-atomic="true">
        {feedback}
      </p>
      <div className="board-caption" aria-hidden="true">
        <span>{game.humanColor === "red" ? "Red house" : "White house"}</span>
        <span>American checkers · ACF 1–32</span>
      </div>
    </div>
  );
}
