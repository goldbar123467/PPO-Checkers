/**
 * American checkers rules on 32-bit bitboards.
 *
 * A line-for-line port of `src/checkers/rules` (board, state, moves, terminal). Squares use the
 * zero-based ACF 1–32 numbering; row 0 is Red's home row. Player 0 is Red and moves first;
 * player 1 is called White in the Python engine and Black in the web interface. Parity with the
 * Python engine is enforced by `src/test/engine.parity.test.ts`.
 */

export const BOARD_SIZE = 8;
export const PLAYABLE_SQUARES = 32;
export const NO_PROGRESS_LIMIT = 40;
export const REPETITION_LIMIT = 3;
export const DEFAULT_MAX_PLIES = 512;

export type Player = 0 | 1;

export type TerminationReason =
  | "no_pieces"
  | "no_legal_move"
  | "no_progress"
  | "repetition"
  | "ply_cap";

export interface State {
  readonly men: readonly [number, number];
  readonly kings: readonly [number, number];
  readonly side: Player;
  readonly captureInProgress: boolean;
  readonly movingSquare: number | null;
  readonly sequenceOrigin: number | null;
  readonly capturedPending: number;
  readonly noProgress: readonly [number, number];
  readonly ply: number;
}

export interface Step {
  readonly origin: number;
  readonly destination: number;
  readonly captured: number | null;
}

export interface Transition {
  readonly after: State;
  readonly moveCompleted: boolean;
}

export interface Outcome {
  readonly winner: Player | null;
  readonly reason: TerminationReason;
}

export interface TerminalOptions {
  readonly maxPlies?: number;
  readonly repetitionDraws?: boolean;
  readonly repetitionCount?: number;
}

/** Row/column deltas for the four diagonal directions, matching `DIRECTION_DELTAS`. */
export const DIRECTION_DELTAS: readonly (readonly [number, number])[] = [
  [1, -1],
  [1, 1],
  [-1, -1],
  [-1, 1],
];
const RED_FORWARD: readonly number[] = [0, 1];
const WHITE_FORWARD: readonly number[] = [2, 3];
const KING_DIRECTIONS: readonly number[] = [0, 1, 2, 3];

export function opponent(player: Player): Player {
  return player === 0 ? 1 : 0;
}

export function bit(square: number): number {
  return (1 << square) >>> 0;
}

export function hasBit(mask: number, square: number): boolean {
  return ((mask >>> square) & 1) === 1;
}

export function coord(square: number): [row: number, column: number] {
  const row = Math.floor(square / 4);
  const offset = square % 4;
  return [row, 6 + (row % 2) - 2 * offset];
}

export function isPlayable(row: number, column: number): boolean {
  return row >= 0 && row < BOARD_SIZE && column >= 0 && column < BOARD_SIZE && (row + column) % 2 === 0;
}

/** Return the zero-based ACF square at a playable coordinate, or null. */
export function squareAt(row: number, column: number): number | null {
  if (!isPlayable(row, column)) return null;
  return row * 4 + (6 + (row % 2) - column) / 2;
}

export function rotateSquare(square: number): number {
  return PLAYABLE_SQUARES - 1 - square;
}

type Geometry = readonly [adjacent: number | null, landing: number | null];

/** Adjacent and jump-landing squares for every square and direction (the Python `GEOMETRY`). */
export const GEOMETRY: readonly (readonly Geometry[])[] = Array.from(
  { length: PLAYABLE_SQUARES },
  (_, square) => {
    const [row, column] = coord(square);
    return DIRECTION_DELTAS.map(([dr, dc]): Geometry => {
      const adjacent = squareAt(row + dr, column + dc);
      return [adjacent, adjacent === null ? null : squareAt(row + 2 * dr, column + 2 * dc)];
    });
  },
);

export function initialState(): State {
  const twelve = (1 << 12) - 1;
  return {
    men: [twelve, (twelve << 20) >>> 0],
    kings: [0, 0],
    side: 0,
    captureInProgress: false,
    movingSquare: null,
    sequenceOrigin: null,
    capturedPending: 0,
    noProgress: [0, 0],
    ply: 0,
  };
}

export function occupied(state: State): number {
  return (state.men[0] | state.men[1] | state.kings[0] | state.kings[1]) >>> 0;
}

function directionsFor(state: State, square: number): readonly number[] {
  if (hasBit(state.kings[state.side], square)) return KING_DIRECTIONS;
  return state.side === 0 ? RED_FORWARD : WHITE_FORWARD;
}

function captureStepsFrom(state: State, origin: number): Step[] {
  const other = opponent(state.side);
  const opponentPieces = state.men[other] | state.kings[other];
  const board = occupied(state);
  const captures: Step[] = [];
  for (const direction of directionsFor(state, origin)) {
    const [adjacent, landing] = GEOMETRY[origin][direction];
    if (adjacent === null || landing === null) continue;
    if (!hasBit(opponentPieces, adjacent)) continue;
    if (hasBit(state.capturedPending, adjacent)) continue;
    if (hasBit(board, landing)) continue;
    captures.push({ origin, destination: landing, captured: adjacent });
  }
  return captures;
}

function simpleStepsFrom(state: State, origin: number): Step[] {
  const board = occupied(state);
  const moves: Step[] = [];
  for (const direction of directionsFor(state, origin)) {
    const [adjacent] = GEOMETRY[origin][direction];
    if (adjacent !== null && !hasBit(board, adjacent)) {
      moves.push({ origin, destination: adjacent, captured: null });
    }
  }
  return moves;
}

function compareSteps(a: Step, b: Step): number {
  return (
    a.origin - b.origin ||
    a.destination - b.destination ||
    (a.captured ?? -1) - (b.captured ?? -1)
  );
}

function ownSquares(state: State): number[] {
  const own = state.men[state.side] | state.kings[state.side];
  const squares: number[] = [];
  for (let square = 0; square < PLAYABLE_SQUARES; square += 1) {
    if (hasBit(own, square)) squares.push(square);
  }
  return squares;
}

/** Every legal step (one simple move or one jump) in deterministic order, with forced capture. */
export function legalSteps(state: State): Step[] {
  if (state.captureInProgress) {
    return captureStepsFrom(state, state.movingSquare as number).sort(compareSteps);
  }
  const origins = ownSquares(state);
  const captures = origins.flatMap((origin) => captureStepsFrom(state, origin));
  if (captures.length > 0) return captures.sort(compareSteps);
  return origins.flatMap((origin) => simpleStepsFrom(state, origin)).sort(compareSteps);
}

function isKingRow(player: Player, square: number): boolean {
  return coord(square)[0] === (player === 0 ? 7 : 0);
}

function movedBoards(state: State, step: Step) {
  const actor = state.side;
  const originBit = bit(step.origin);
  const destinationBit = bit(step.destination);
  const men: [number, number] = [state.men[0], state.men[1]];
  const kings: [number, number] = [state.kings[0], state.kings[1]];
  const wasMan = hasBit(men[actor], step.origin);
  if (wasMan) men[actor] = ((men[actor] ^ originBit) | destinationBit) >>> 0;
  else kings[actor] = ((kings[actor] ^ originBit) | destinationBit) >>> 0;
  return { men, kings, wasMan };
}

function promote(men: [number, number], kings: [number, number], player: Player, square: number) {
  const destinationBit = bit(square);
  men[player] = (men[player] ^ destinationBit) >>> 0;
  kings[player] = (kings[player] | destinationBit) >>> 0;
}

function finishCapture(intermediate: State, wasMan: boolean): State {
  const actor = intermediate.side;
  const other = opponent(actor);
  const men: [number, number] = [intermediate.men[0], intermediate.men[1]];
  const kings: [number, number] = [intermediate.kings[0], intermediate.kings[1]];
  men[other] = (men[other] & ~intermediate.capturedPending) >>> 0;
  kings[other] = (kings[other] & ~intermediate.capturedPending) >>> 0;
  const destination = intermediate.movingSquare as number;
  if (wasMan && isKingRow(actor, destination)) promote(men, kings, actor, destination);
  const noProgress: [number, number] = [intermediate.noProgress[0], intermediate.noProgress[1]];
  noProgress[actor] = 0;
  return {
    men,
    kings,
    side: other,
    captureInProgress: false,
    movingSquare: null,
    sequenceOrigin: null,
    capturedPending: 0,
    noProgress,
    ply: intermediate.ply,
  };
}

function applyCapture(state: State, step: Step): Transition {
  const { men, kings, wasMan } = movedBoards(state, step);
  const intermediate: State = {
    men,
    kings,
    side: state.side,
    captureInProgress: true,
    movingSquare: step.destination,
    sequenceOrigin: state.captureInProgress ? state.sequenceOrigin : step.origin,
    capturedPending: (state.capturedPending | bit(step.captured as number)) >>> 0,
    noProgress: state.noProgress,
    ply: state.ply + 1,
  };
  const promotionEndsMove = wasMan && isKingRow(state.side, step.destination);
  if (!promotionEndsMove && captureStepsFrom(intermediate, step.destination).length > 0) {
    return { after: intermediate, moveCompleted: false };
  }
  return { after: finishCapture(intermediate, wasMan), moveCompleted: true };
}

function applySimple(state: State, step: Step): Transition {
  const { men, kings, wasMan } = movedBoards(state, step);
  const actor = state.side;
  if (wasMan && isKingRow(actor, step.destination)) promote(men, kings, actor, step.destination);
  const noProgress: [number, number] = [state.noProgress[0], state.noProgress[1]];
  noProgress[actor] = wasMan ? 0 : noProgress[actor] + 1;
  return {
    after: {
      men,
      kings,
      side: opponent(actor),
      captureInProgress: false,
      movingSquare: null,
      sequenceOrigin: null,
      capturedPending: 0,
      noProgress,
      ply: state.ply + 1,
    },
    moveCompleted: true,
  };
}

function sameStep(a: Step, b: Step): boolean {
  return a.origin === b.origin && a.destination === b.destination && a.captured === b.captured;
}

/** Apply one legal step. Throws if the step is not legal in `state`. */
export function applyStep(state: State, step: Step): Transition {
  if (!legalSteps(state).some((candidate) => sameStep(candidate, step))) {
    throw new Error("step is not legal in this state");
  }
  return step.captured === null ? applySimple(state, step) : applyCapture(state, step);
}

/** Placement-plus-side key used for threefold repetition at completed-move boundaries. */
export function positionKey(state: State): string {
  if (state.captureInProgress) {
    throw new Error("position keys exist only at completed-move boundaries");
  }
  return `${state.men[0]}:${state.men[1]}:${state.kings[0]}:${state.kings[1]}:${state.side}`;
}

/** Return the terminal outcome, or null while play can continue. Losses precede draws. */
export function terminalOutcome(state: State, options: TerminalOptions = {}): Outcome | null {
  const maxPlies = options.maxPlies ?? DEFAULT_MAX_PLIES;
  const winner = opponent(state.side);
  if (((state.men[state.side] | state.kings[state.side]) >>> 0) === 0) {
    return { winner, reason: "no_pieces" };
  }
  if (legalSteps(state).length === 0) return { winner, reason: "no_legal_move" };
  if (state.noProgress.every((counter) => counter >= NO_PROGRESS_LIMIT)) {
    return { winner: null, reason: "no_progress" };
  }
  if (
    options.repetitionDraws &&
    !state.captureInProgress &&
    (options.repetitionCount ?? 0) >= REPETITION_LIMIT
  ) {
    return { winner: null, reason: "repetition" };
  }
  if (state.ply >= maxPlies) return { winner: null, reason: "ply_cap" };
  return null;
}
