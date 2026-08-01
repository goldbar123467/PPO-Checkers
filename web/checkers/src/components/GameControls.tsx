import type { Color, ModelInfo } from "../types";

interface GameControlsProps {
  model: ModelInfo;
  humanColor: Color;
  busy: boolean;
  hasGame: boolean;
  onHumanColor: (value: Color) => void;
  onStart: () => void;
}

export function GameControls({
  model,
  humanColor,
  busy,
  hasGame,
  onHumanColor,
  onStart,
}: GameControlsProps) {
  return (
    <section className="simple-panel setup-panel" aria-labelledby="setup-heading">
      <p className="panel-label">Game setup</p>
      <h2 id="setup-heading">Choose your side</h2>
      <p className="setup-intro">Orange moves first. White lets the AI make the opening move.</p>

      <fieldset className="side-picker">
        <legend className="sr-only">Choose your checker color</legend>
        <button
          type="button"
          className={humanColor === "red" ? "side-choice is-selected" : "side-choice"}
          aria-pressed={humanColor === "red"}
          disabled={busy}
          onClick={() => onHumanColor("red")}
        >
          <span className="choice-piece choice-piece--orange" aria-hidden="true">O</span>
          <span><strong>Orange</strong><small>You move first</small></span>
        </button>
        <button
          type="button"
          className={humanColor === "white" ? "side-choice is-selected" : "side-choice"}
          aria-pressed={humanColor === "white"}
          disabled={busy}
          onClick={() => onHumanColor("white")}
        >
          <span className="choice-piece choice-piece--white" aria-hidden="true">W</span>
          <span><strong>White</strong><small>AI moves first</small></span>
        </button>
      </fieldset>

      <button className="start-button" type="button" disabled={busy} onClick={onStart}>
        <span>{busy ? "AI is moving…" : hasGame ? "Start a new game" : "Start game"}</span>
        <span aria-hidden="true">→</span>
      </button>

      <div className="model-ready">
        <span aria-hidden="true" />
        <p><strong>Real model ready</strong><small>PPO checkpoint {model.update.toLocaleString()} · {model.parameterCount.toLocaleString()} parameters</small></p>
      </div>
    </section>
  );
}
