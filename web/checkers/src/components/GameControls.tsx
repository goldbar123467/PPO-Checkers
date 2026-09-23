import { MODEL_INFO } from "@/lib/model";
import type { Color } from "@/types";

import type { PolicyStatus } from "@/hooks/useCheckers";

interface GameControlsProps {
  status: PolicyStatus["state"];
  humanColor: Color;
  busy: boolean;
  hasGame: boolean;
  onHumanColor: (value: Color) => void;
  onStart: () => void;
}

function startLabel(status: PolicyStatus["state"], busy: boolean, hasGame: boolean): string {
  if (status === "loading") return "Loading model…";
  if (status === "error") return "Model unavailable";
  if (busy) return "AI is moving…";
  return hasGame ? "Start a new game" : "Start game";
}

export function GameControls({
  status,
  humanColor,
  busy,
  hasGame,
  onHumanColor,
  onStart,
}: GameControlsProps) {
  const ready = status === "ready";
  return (
    <section className="panel setup-panel" aria-labelledby="setup-heading">
      <p className="panel-label">New game</p>
      <h2 id="setup-heading">Choose your side</h2>
      <p className="setup-intro">Red moves first. Pick Black to let the AI open.</p>

      <fieldset className="side-picker">
        <legend className="sr-only">Choose your checker color</legend>
        <button
          type="button"
          className={humanColor === "red" ? "side-choice is-selected" : "side-choice"}
          aria-pressed={humanColor === "red"}
          disabled={busy}
          onClick={() => onHumanColor("red")}
        >
          <span className="choice-piece choice-piece--red" aria-hidden="true" />
          <span><strong>Red</strong><small>You move first</small></span>
        </button>
        <button
          type="button"
          className={humanColor === "black" ? "side-choice is-selected" : "side-choice"}
          aria-pressed={humanColor === "black"}
          disabled={busy}
          onClick={() => onHumanColor("black")}
        >
          <span className="choice-piece choice-piece--black" aria-hidden="true" />
          <span><strong>Black</strong><small>AI moves first</small></span>
        </button>
      </fieldset>

      <button className="start-button" type="button" disabled={!ready || busy} onClick={onStart}>
        <span>{startLabel(status, busy, hasGame)}</span>
        <span aria-hidden="true">→</span>
      </button>

      <div className={`model-status model-status--${status}`} role="status">
        <span aria-hidden="true" />
        <p>
          <strong>{ready ? "Model ready" : status === "loading" ? "Loading model" : "Model failed to load"}</strong>
          <small>
            PPO update {MODEL_INFO.update.toLocaleString()} · {MODEL_INFO.parameterCount.toLocaleString()} parameters
          </small>
        </p>
      </div>
    </section>
  );
}
