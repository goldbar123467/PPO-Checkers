import { encodeObservation } from "@/engine/encoding";
import { CheckersGame } from "@/engine/game";
import { greedyAction, legalProbabilities, sampleAction, seededRandom } from "@/engine/policy";
import {
  BOARD_SIZE,
  PLAYABLE_SQUARES,
  coord,
  hasBit,
  isPlayable,
  squareAt,
  type Player,
} from "@/engine/rules";
import type {
  BoardCell,
  Color,
  GameSnapshot,
  MoveRecord,
  Piece,
  PolicyInsight,
  PolicyMode,
} from "@/types";

export interface Evaluation {
  logits: Float32Array;
  value: number;
  inferenceMs: number;
}

export interface SessionOptions {
  id: string;
  humanColor: Color;
  policyMode: PolicyMode;
  seed: number;
  maxPlies: number;
  repetitionDraws: boolean;
}

export function colorOf(player: Player): Color {
  return player === 0 ? "red" : "black";
}

function playerOf(color: Color): Player {
  return color === "red" ? 0 : 1;
}

const BOARD: readonly BoardCell[] = Array.from({ length: BOARD_SIZE * BOARD_SIZE }, (_, index) => {
  const row = Math.floor(index / BOARD_SIZE);
  const column = index % BOARD_SIZE;
  return { row, column, playable: isPlayable(row, column), square: squareAt(row, column) };
});

/** One human-versus-policy game held entirely in the browser. */
export class GameSession {
  readonly id: string;
  readonly human: Player;
  readonly policyMode: PolicyMode;
  readonly seed: number;
  private readonly game: CheckersGame;
  private readonly random: () => number;
  private readonly moves: MoveRecord[] = [];
  private lastStep: { origin: number; destination: number } | null = null;

  constructor(options: SessionOptions) {
    this.id = options.id;
    this.human = playerOf(options.humanColor);
    this.policyMode = options.policyMode;
    this.seed = options.seed;
    this.random = seededRandom(options.seed);
    this.game = new CheckersGame({
      maxPlies: options.maxPlies,
      repetitionDraws: options.repetitionDraws,
    });
  }

  get isModelTurn(): boolean {
    return !this.game.terminated && this.game.state.side !== this.human;
  }

  /** The policy observation for the current state. */
  observation(): Float32Array {
    return encodeObservation(this.game.state);
  }

  /** Apply one human step identified by its origin and destination squares. */
  applyHumanMove(origin: number, destination: number): void {
    if (this.game.terminated) throw new Error("The game has already ended.");
    if (this.game.state.side !== this.human) throw new Error("Wait for the AI to finish its turn.");
    const matching = this.game
      .legalActions()
      .filter(({ step }) => step.origin === origin && step.destination === destination);
    if (matching.length !== 1) throw new Error("That is not a legal move in this position.");
    this.apply(matching[0].action);
  }

  /** Choose and apply the model's next step from a network evaluation of `observation()`. */
  applyModelDecision(evaluation: Evaluation): PolicyInsight {
    if (!this.isModelTurn) throw new Error("It is not the AI's turn.");
    const legal = this.game.legalActions();
    const action =
      this.policyMode === "greedy"
        ? greedyAction(evaluation.logits, legal)
        : sampleAction(evaluation.logits, legal, this.random);
    const confidence = legalProbabilities(evaluation.logits, legal).get(action) ?? 0;
    this.apply(action);
    return {
      value: evaluation.value,
      confidence,
      legalMoveCount: legal.length,
      inferenceMs: evaluation.inferenceMs,
    };
  }

  private apply(action: number): void {
    const result = this.game.step(action);
    this.lastStep = { origin: result.step.origin, destination: result.step.destination };
    if (result.notation !== null) {
      this.moves.push({ ply: this.moves.length + 1, actor: colorOf(result.actor), notation: result.notation });
    }
  }

  private pieces(): Piece[] {
    const { men, kings } = this.game.state;
    const pieces: Piece[] = [];
    for (let square = 0; square < PLAYABLE_SQUARES; square += 1) {
      for (const player of [0, 1] as const) {
        const kind = hasBit(men[player], square) ? "man" : hasBit(kings[player], square) ? "king" : null;
        if (kind === null) continue;
        const [row, column] = coord(square);
        pieces.push({ square, row, column, color: colorOf(player), kind });
      }
    }
    return pieces;
  }

  snapshot(): GameSnapshot {
    const state = this.game.state;
    const humanTurn = !this.game.terminated && state.side === this.human;
    const outcome = this.game.outcome;
    return {
      id: this.id,
      humanColor: colorOf(this.human),
      modelColor: colorOf(this.human === 0 ? 1 : 0),
      policyMode: this.policyMode,
      seed: this.seed,
      sideToMove: colorOf(state.side),
      isHumanTurn: humanTurn,
      captureInProgress: state.captureInProgress,
      forcedSquare: state.movingSquare,
      ply: state.ply,
      board: BOARD.map((cell) => ({ ...cell })),
      pieces: this.pieces(),
      legalMoves: humanTurn
        ? this.game.legalActions().map(({ action, step }) => ({ action, ...step }))
        : [],
      lastStep: this.lastStep,
      moves: [...this.moves],
      outcome:
        outcome === null
          ? null
          : {
              winner: outcome.winner === null ? null : colorOf(outcome.winner),
              reason: outcome.reason,
              isDraw: outcome.winner === null,
            },
    };
  }
}
