import { useEffect, useState } from "react";
import { createGame, fetchModel, submitMove } from "./api";
import { Board } from "./components/Board";
import { GameControls } from "./components/GameControls";
import { MatchLedger } from "./components/MatchLedger";
import type { Color, GameSnapshot, ModelInfo, PolicyMode } from "./types";

function randomMatchSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

export default function App() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [game, setGame] = useState<GameSnapshot | null>(null);
  const [humanColor, setHumanColor] = useState<Color>("red");
  const [policyMode, setPolicyMode] = useState<PolicyMode>("greedy");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchModel()
      .then((metadata) => {
        if (active) setModel(metadata);
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : "Could not reach the policy service.");
        }
      });
    return () => {
      active = false;
    };
  }, []);

  async function startGame() {
    const seed = randomMatchSeed();
    setBusy(true);
    setError(null);
    try {
      setGame(await createGame(humanColor, policyMode, seed));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The game could not be created.");
    } finally {
      setBusy(false);
    }
  }

  async function move(origin: number, destination: number) {
    if (!game) return;
    setBusy(true);
    setError(null);
    try {
      setGame(await submitMove(game.id, origin, destination));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The move could not be applied.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="masthead">
        <a className="brand" href="#top" aria-label="Red House checkers home">
          <span className="brand-mark" aria-hidden="true">
            RH
          </span>
          <span>
            <strong>Red House</strong>
            <small>neural policy room</small>
          </span>
        </a>
        <div className="local-badge">
          <span aria-hidden="true">●</span> live CPU policy · games are ephemeral
        </div>
      </header>

      <section className="hero" id="top">
        <p className="eyebrow">ML Lab · policy table 01</p>
        <h1>Play the run,<br />not a simulation.</h1>
        <p className="hero-copy">
          The saved PPO policy is loaded on the game server. Every legal move comes from the same
          tested American checkers engine used during training.
        </p>
      </section>

      {error && (
        <div className="error-banner" role="alert">
          <strong>Policy service error</strong>
          <span>{error}</span>
          <button type="button" onClick={() => window.location.reload()}>
            Retry connection
          </button>
        </div>
      )}

      {!model ? (
        <section className="loading-card" aria-live="polite">
          <span className="loader" aria-hidden="true" />
          <div>
            <p className="eyebrow">Startup gate</p>
            <h2>Waiting for the neural policy</h2>
            <p>A match cannot begin until its checksummed weights are ready.</p>
          </div>
        </section>
      ) : (
        <div className="workspace">
          <GameControls
            model={model}
            humanColor={humanColor}
            policyMode={policyMode}
            currentSeed={game?.seed ?? null}
            busy={busy}
            hasGame={Boolean(game)}
            onHumanColor={setHumanColor}
            onPolicyMode={setPolicyMode}
            onStart={startGame}
          />

          <section className="table-stage" aria-label="Game table">
            {game ? (
              <Board game={game} busy={busy} onMove={move} />
            ) : (
              <div className="empty-board" aria-label="Board waiting for a match">
                <div className="empty-board__seal" aria-hidden="true">32</div>
                <p>Choose a side, then begin the match.</p>
              </div>
            )}
          </section>

          {game ? (
            <MatchLedger game={game} busy={busy} />
          ) : (
            <section className="panel rules-note">
              <p className="eyebrow">Table rules</p>
              <h2>Server-authoritative play</h2>
              <p>
                Captures are mandatory. Multi-jumps remain with the same piece. Promotion and draw
                rules use the run's exact environment variables.
              </p>
              <ul>
                <li>Ring = a piece that can move</li>
                <li>Gold dot = a legal destination</li>
                <li>Fine outline = the last step</li>
              </ul>
            </section>
          )}
        </div>
      )}

      <footer>
        <span>American checkers · ACF 1–32</span>
        <span>model update {model?.update.toLocaleString() ?? "—"}</span>
      </footer>
    </main>
  );
}
