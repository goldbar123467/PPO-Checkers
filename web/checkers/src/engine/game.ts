/**
 * Step-wise game environment, matching `checkers.env.checkers_env.CheckersEnv`.
 *
 * One `step` applies one simple move or one jump of a capture sequence. Completed moves are
 * recorded in ACF notation (`11-15`, `22x18x9`), official positions are counted for optional
 * threefold repetition, and the terminal outcome is re-evaluated after every step.
 */
import { legalActions, type LegalAction } from "./encoding";
import {
  DEFAULT_MAX_PLIES,
  applyStep,
  initialState,
  positionKey,
  terminalOutcome,
  type Outcome,
  type Player,
  type State,
  type Step,
} from "./rules";

export interface GameConfig {
  readonly maxPlies?: number;
  readonly repetitionDraws?: boolean;
  readonly initialState?: State;
}

export interface StepResult {
  readonly actor: Player;
  readonly step: Step;
  readonly moveCompleted: boolean;
  readonly notation: string | null;
}

export function formatMove(squares: readonly number[], isCapture: boolean): string {
  return squares.map((square) => square + 1).join(isCapture ? "x" : "-");
}

export class CheckersGame {
  readonly maxPlies: number;
  readonly repetitionDraws: boolean;
  private currentState: State;
  private currentOutcome: Outcome | null;
  private readonly positionCounts = new Map<string, number>();
  private activeMoveSquares: number[] = [];

  constructor(config: GameConfig = {}) {
    this.maxPlies = config.maxPlies ?? DEFAULT_MAX_PLIES;
    this.repetitionDraws = config.repetitionDraws ?? true;
    const start = config.initialState ?? initialState();
    if (start.captureInProgress) throw new Error("games must start at a completed-move boundary");
    this.currentState = start;
    this.positionCounts.set(positionKey(start), 1);
    this.currentOutcome = this.evaluate();
  }

  get state(): State {
    return this.currentState;
  }

  get outcome(): Outcome | null {
    return this.currentOutcome;
  }

  get terminated(): boolean {
    return this.currentOutcome !== null;
  }

  /** Legal actions for the side to move; empty once the game has ended. */
  legalActions(): LegalAction[] {
    return this.terminated ? [] : legalActions(this.currentState);
  }

  step(action: number): StepResult {
    if (this.terminated) throw new Error("cannot step a finished game");
    const legal = legalActions(this.currentState).find((candidate) => candidate.action === action);
    if (!legal) throw new Error(`illegal action ${action}`);
    const actor = this.currentState.side;
    const { step } = legal;
    if (this.activeMoveSquares.length > 0) this.activeMoveSquares.push(step.destination);
    else this.activeMoveSquares = [step.origin, step.destination];

    const transition = applyStep(this.currentState, step);
    this.currentState = transition.after;
    let notation: string | null = null;
    if (transition.moveCompleted) {
      notation = formatMove(this.activeMoveSquares, step.captured !== null);
      this.activeMoveSquares = [];
      const key = positionKey(this.currentState);
      this.positionCounts.set(key, (this.positionCounts.get(key) ?? 0) + 1);
    }
    this.currentOutcome = this.evaluate();
    return { actor, step, moveCompleted: transition.moveCompleted, notation };
  }

  private evaluate(): Outcome | null {
    const state = this.currentState;
    return terminalOutcome(state, {
      maxPlies: this.maxPlies,
      repetitionDraws: this.repetitionDraws,
      repetitionCount: state.captureInProgress ? 0 : (this.positionCounts.get(positionKey(state)) ?? 0),
    });
  }
}
