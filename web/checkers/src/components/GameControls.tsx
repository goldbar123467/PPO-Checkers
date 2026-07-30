import type { Color, ModelInfo, PolicyMode } from "../types";

interface GameControlsProps {
  model: ModelInfo;
  humanColor: Color;
  policyMode: PolicyMode;
  currentSeed: number | null;
  busy: boolean;
  hasGame: boolean;
  onHumanColor: (value: Color) => void;
  onPolicyMode: (value: PolicyMode) => void;
  onStart: () => void;
}

export function GameControls({
  model,
  humanColor,
  policyMode,
  currentSeed,
  busy,
  hasGame,
  onHumanColor,
  onPolicyMode,
  onStart,
}: GameControlsProps) {
  return (
    <section className="panel setup-panel" aria-labelledby="setup-heading">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Match desk</p>
          <h2 id="setup-heading">Set the table</h2>
        </div>
        <span className="ready-pill">
          <span className="ready-dot" /> neural model ready
        </span>
      </div>

      <fieldset className="segmented-field">
        <legend>Your side</legend>
        <div className="segment-row">
          {(["red", "white"] as const).map((color) => (
            <button
              key={color}
              type="button"
              className={humanColor === color ? "segment is-active" : "segment"}
              aria-pressed={humanColor === color}
              disabled={busy}
              onClick={() => onHumanColor(color)}
            >
              <span className={`mini-piece mini-piece--${color}`} />
              {color}
            </button>
          ))}
        </div>
      </fieldset>

      <label className="field-label" htmlFor="policy-mode">
        Policy
      </label>
      <select
        id="policy-mode"
        value={policyMode}
        disabled={busy}
        onChange={(event) => onPolicyMode(event.target.value as PolicyMode)}
      >
        <option value="greedy">Neural policy · greedy deterministic</option>
        <option value="sampled">Neural policy · seeded sampling</option>
      </select>
      <p className="field-help">
        Both modes use trained update {model.update.toLocaleString()}. Minimax-2 was an evaluation
        opponent and is not used here.
      </p>

      <div className="automatic-seed" aria-live="polite">
        <span>Automatic match seed</span>
        <strong>{currentSeed === null ? "Generated at start" : currentSeed}</strong>
        <small>
          A fresh random seed is created for every game. Sampled mode uses it; greedy mode ignores
          it.
        </small>
      </div>

      <button className="primary-action" type="button" disabled={busy} onClick={onStart}>
        <span>{busy ? "Model is moving…" : hasGame ? "Start new match" : "Begin match"}</span>
        <span aria-hidden="true">→</span>
      </button>

      <dl className="model-facts">
        <div>
          <dt>Neural checkpoint</dt>
          <dd>update {model.update.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Runtime</dt>
          <dd>{model.device.toUpperCase()} · server</dd>
        </div>
        <div>
          <dt>Parameters</dt>
          <dd>{model.parameterCount.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Bundle</dt>
          <dd title={model.bundleSha256}>{model.bundleSha256.slice(0, 10)}…</dd>
        </div>
      </dl>
    </section>
  );
}
