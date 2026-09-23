/**
 * Actor-canonical observations and the fixed 128-slot action space.
 *
 * Ports of `checkers.env.encoding.encode_observation` and `checkers.env.masking`. The observation
 * is eight `8 × 8` planes from the side to move's perspective: own men, own kings, opponent men,
 * opponent kings, pending captures, the forced moving piece, the actor's no-progress counter, and
 * the normalized ply.
 */
import {
  BOARD_SIZE,
  DEFAULT_MAX_PLIES,
  DIRECTION_DELTAS,
  NO_PROGRESS_LIMIT,
  PLAYABLE_SQUARES,
  coord,
  hasBit,
  legalSteps,
  opponent,
  rotateSquare,
  type State,
  type Step,
} from "./rules";

export const OBSERVATION_PLANES = 8;
export const OBSERVATION_SIZE = OBSERVATION_PLANES * BOARD_SIZE * BOARD_SIZE;
export const DIRECTIONS_PER_SQUARE = 4;
export const ACTION_COUNT = PLAYABLE_SQUARES * DIRECTIONS_PER_SQUARE;

export interface LegalAction {
  readonly action: number;
  readonly step: Step;
}

function canonicalSquare(state: State, square: number): number {
  return state.side === 0 ? square : rotateSquare(square);
}

function writeMask(observation: Float32Array, plane: number, state: State, mask: number) {
  for (let square = 0; square < PLAYABLE_SQUARES; square += 1) {
    if (!hasBit(mask, square)) continue;
    const [row, column] = coord(canonicalSquare(state, square));
    observation[plane * 64 + row * BOARD_SIZE + column] = 1;
  }
}

/** Encode `state` as a `(8, 8, 8)` float32 tensor in plane-major order. */
export function encodeObservation(state: State, maxPlies = DEFAULT_MAX_PLIES): Float32Array {
  const observation = new Float32Array(OBSERVATION_SIZE);
  const actor = state.side;
  const other = opponent(actor);
  writeMask(observation, 0, state, state.men[actor]);
  writeMask(observation, 1, state, state.kings[actor]);
  writeMask(observation, 2, state, state.men[other]);
  writeMask(observation, 3, state, state.kings[other]);
  writeMask(observation, 4, state, state.capturedPending);
  if (state.movingSquare !== null) writeMask(observation, 5, state, 1 << state.movingSquare);
  observation.fill(Math.fround(state.noProgress[actor] / NO_PROGRESS_LIMIT), 6 * 64, 7 * 64);
  observation.fill(Math.fround(state.ply / maxPlies), 7 * 64, 8 * 64);
  return observation;
}

function worldDirection(step: Step): number {
  const [originRow, originColumn] = coord(step.origin);
  const [destinationRow, destinationColumn] = coord(step.destination);
  const distance = step.captured === null ? 1 : 2;
  const rowDelta = (destinationRow - originRow) / distance;
  const columnDelta = (destinationColumn - originColumn) / distance;
  return DIRECTION_DELTAS.findIndex(([dr, dc]) => dr === rowDelta && dc === columnDelta);
}

/** Encode one legal step as its canonical action ID for the side to move. */
export function stepToAction(state: State, step: Step): number {
  const direction = worldDirection(step);
  if (state.side === 1) {
    return rotateSquare(step.origin) * DIRECTIONS_PER_SQUARE + (DIRECTIONS_PER_SQUARE - 1 - direction);
  }
  return step.origin * DIRECTIONS_PER_SQUARE + direction;
}

/** Legal actions in the engine's deterministic legal-step order. */
export function legalActions(state: State): LegalAction[] {
  return legalSteps(state).map((step) => ({ action: stepToAction(state, step), step }));
}
