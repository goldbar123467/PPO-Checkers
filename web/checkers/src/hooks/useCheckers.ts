import { useCallback, useEffect, useRef, useState } from "react";

import { GAME_RULES } from "@/lib/model";
import { createPolicyRunner, type PolicyRunner } from "@/lib/policyRunner";
import { GameSession } from "@/lib/session";
import type { Color, GameSnapshot, PolicyInsight } from "@/types";

export type PolicyStatus =
  | { state: "loading" }
  | { state: "ready" }
  | { state: "error"; message: string };

export interface UseCheckersOptions {
  /** Minimum time each AI step stays "thinking", so its moves are visible. */
  moveDelayMs?: number;
  createRunner?: () => Promise<PolicyRunner>;
}

function randomSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

let gameCounter = 0;

/** Own the policy runner and one in-browser game session. */
export function useCheckers({
  moveDelayMs = 420,
  createRunner = createPolicyRunner,
}: UseCheckersOptions = {}) {
  const [status, setStatus] = useState<PolicyStatus>({ state: "loading" });
  const [game, setGame] = useState<GameSnapshot | null>(null);
  const [thinking, setThinking] = useState(false);
  const [insight, setInsight] = useState<PolicyInsight | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const runnerRef = useRef<PolicyRunner | null>(null);
  const sessionRef = useRef<GameSession | null>(null);

  useEffect(() => {
    let active = true;
    createRunner().then(
      (runner) => {
        if (!active) {
          runner.dispose();
          return;
        }
        runnerRef.current = runner;
        setStatus({ state: "ready" });
      },
      (failure: unknown) => {
        if (active) setStatus({ state: "error", message: message(failure) });
      },
    );
    return () => {
      active = false;
      runnerRef.current?.dispose();
      runnerRef.current = null;
      sessionRef.current = null;
    };
  }, [attempt, createRunner]);

  const playModelTurn = useCallback(
    async (session: GameSession) => {
      const runner = runnerRef.current;
      if (!runner || !session.isModelTurn) return;
      setThinking(true);
      try {
        while (session.isModelTurn && sessionRef.current === session) {
          const started = performance.now();
          const evaluation = await runner.evaluate(session.observation());
          const remaining = moveDelayMs - (performance.now() - started);
          if (remaining > 0) await wait(remaining);
          if (sessionRef.current !== session) return;
          setInsight(session.applyModelDecision(evaluation));
          setGame(session.snapshot());
        }
      } catch (failure) {
        if (sessionRef.current === session) setError(message(failure));
      } finally {
        if (sessionRef.current === session) setThinking(false);
      }
    },
    [moveDelayMs],
  );

  const startGame = useCallback(
    (humanColor: Color) => {
      gameCounter += 1;
      const seed = randomSeed();
      const session = new GameSession({
        id: `game-${gameCounter}-${seed}`,
        humanColor,
        policyMode: "greedy",
        seed,
        ...GAME_RULES,
      });
      sessionRef.current = session;
      setError(null);
      setInsight(null);
      setThinking(false);
      setGame(session.snapshot());
      void playModelTurn(session);
    },
    [playModelTurn],
  );

  const move = useCallback(
    (origin: number, destination: number) => {
      const session = sessionRef.current;
      if (!session) return;
      try {
        session.applyHumanMove(origin, destination);
      } catch (failure) {
        setError(message(failure));
        return;
      }
      setError(null);
      setGame(session.snapshot());
      void playModelTurn(session);
    },
    [playModelTurn],
  );

  const retry = useCallback(() => {
    if (status.state === "error") {
      setGame(null);
      setStatus({ state: "loading" });
      setAttempt((value) => value + 1);
      return;
    }
    setError(null);
    const session = sessionRef.current;
    if (session) void playModelTurn(session);
  }, [playModelTurn, status.state]);

  return { status, game, thinking, insight, error, startGame, move, retry };
}
