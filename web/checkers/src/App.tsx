import { useLayoutEffect, useRef, useState } from "react";

import { Board } from "@/components/Board";
import { GameControls } from "@/components/GameControls";
import { LogoMark } from "@/components/LogoMark";
import { GameStatus, MatchLedger } from "@/components/MatchLedger";
import { ModelInsight } from "@/components/ModelInsight";
import { useCheckers, type UseCheckersOptions } from "@/hooks/useCheckers";
import { MODEL_INFO } from "@/lib/model";
import type { Color } from "@/types";

export const REPOSITORY_URL = "https://github.com/goldbar123467/PPO-Checkers";

const megabytes = (MODEL_INFO.weightsSizeBytes / (1024 * 1024)).toFixed(1);

function App(options: UseCheckersOptions) {
  const { status, game, thinking, insight, error, startGame, move, retry } = useCheckers(options);
  const [humanColor, setHumanColor] = useState<Color>("red");
  const [highContrast, setHighContrast] = useState(false);
  const tableRef = useRef<HTMLElement>(null);
  const gameId = game?.id;
  const ready = status.state === "ready";
  const alert = status.state === "error" ? status.message : error;

  useLayoutEffect(() => {
    const table = tableRef.current;
    if (!gameId || !table) return;
    table.focus({ preventScroll: true });
    const bounds = table.getBoundingClientRect();
    if (bounds.top < 0 || bounds.bottom > window.innerHeight) table.scrollIntoView?.({ block: "start" });
  }, [gameId]);

  return (
    <div className={highContrast ? "site-shell board-high-contrast" : "site-shell"}>
      <a className="skip-link" href="#play">
        Skip to the checkers game
      </a>

      <header className="site-header">
        <a className="brand" href="#top" aria-label="PPO Checkers home">
          <LogoMark />
          <span>PPO Checkers</span>
        </a>
        <nav aria-label="Page navigation">
          <a href="#play">Play</a>
          <a href="#how-it-works">How it works</a>
          <a href="#results">Results</a>
          <a className="nav-github" href={REPOSITORY_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </nav>
      </header>

      <main id="top">
        <section className="hero" aria-labelledby="page-title">
          <div className="hero__copy">
            <p className="kicker">Reinforcement learning · runs in your browser</p>
            <h1 id="page-title">
              Play checkers against a neural network <em>trained by self-play.</em>
            </h1>
            <p className="hero__lede">
              A {MODEL_INFO.parameterCount.toLocaleString()}-parameter policy/value network learned
              American checkers with Proximal Policy Optimization. The model and the rules engine run
              entirely on your device: no server, no account, no tracking.
            </p>
            <div className="hero__actions">
              <a className="button button--primary" href="#play">Play now</a>
              <a className="button button--ghost" href={REPOSITORY_URL} target="_blank" rel="noreferrer">
                View the source
              </a>
            </div>
          </div>
          <dl className="hero__stats" aria-label="Model at a glance">
            <div><dt>Parameters</dt><dd>470K</dd></div>
            <div><dt>Self-play transitions</dt><dd>37.7M</dd></div>
            <div><dt>Score vs Minimax-2</dt><dd>90%</dd></div>
          </dl>
        </section>

        {alert ? (
          <section className="alert" role="alert">
            <div>
              <strong>{status.state === "error" ? "The model could not be loaded." : "Something interrupted the game."}</strong>
              <p>{alert}</p>
            </div>
            <button type="button" onClick={retry}>Try again</button>
          </section>
        ) : null}

        <section className="workspace" id="play" aria-label="Play checkers">
          <GameControls
            status={status.state}
            humanColor={humanColor}
            busy={thinking}
            hasGame={Boolean(game)}
            onHumanColor={setHumanColor}
            onStart={() => startGame(humanColor)}
          />

          <section className="board-stage" aria-label="Checkers game table" ref={tableRef} tabIndex={-1}>
            {game ? (
              <Board game={game} busy={thinking} onMove={move} />
            ) : (
              <div className="board-placeholder" aria-live="polite">
                <div className="mini-board" aria-hidden="true">
                  {Array.from({ length: 16 }, (_, index) => <span key={index} />)}
                </div>
                <strong>{ready ? "Your board is ready." : "Loading the trained policy…"}</strong>
                <p>
                  {ready
                    ? "Choose Red or Black, then press Start game."
                    : `Downloading ${megabytes} MB of network weights. This happens once.`}
                </p>
              </div>
            )}
          </section>

          <aside className="sidebar" aria-label="Game status and move history">
            {game ? <GameStatus game={game} busy={thinking} /> : (
              <section className="panel quick-guide">
                <p className="panel-label">Three quick rules</p>
                <ol>
                  <li><span>1</span>Pieces move diagonally forward on dark squares.</li>
                  <li><span>2</span>If you can jump, you must jump.</li>
                  <li><span>3</span>Reach the far row to crown a king.</li>
                </ol>
              </section>
            )}
            {game ? <ModelInsight insight={insight} thinking={thinking} /> : null}
            {game ? <MatchLedger game={game} /> : null}
            <section className="panel display-option">
              <label>
                <input
                  type="checkbox"
                  checked={highContrast}
                  onChange={(event) => setHighContrast(event.target.checked)}
                />
                High-contrast board
              </label>
            </section>
          </aside>
        </section>

        <section className="section" id="how-it-works" aria-labelledby="how-heading">
          <div className="section-heading">
            <p className="kicker">How it works</p>
            <h2 id="how-heading">From random moves to a policy that plays.</h2>
            <p>
              The network was never shown expert games. It played millions of moves against copies of
              itself, and PPO nudged it toward the legal moves that led to wins.
            </p>
          </div>
          <ol className="steps">
            <li>
              <span className="steps__index">01</span>
              <strong>Encode the board</strong>
              <p>Eight 8×8 planes from the mover's perspective: men, kings, pending captures, the forced piece, and game counters.</p>
            </li>
            <li>
              <span className="steps__index">02</span>
              <strong>Play itself</strong>
              <p>Vectorized self-play against current and past versions of the policy produces complete games: captures, kings, draws, and losses.</p>
            </li>
            <li>
              <span className="steps__index">03</span>
              <strong>Update with PPO</strong>
              <p>Clipped policy updates with GAE advantages improve the policy while limiting how far each update can move it.</p>
            </li>
            <li>
              <span className="steps__index">04</span>
              <strong>Pick a checkpoint</strong>
              <p>Later was not always better. Saved checkpoints played a fixed match protocol, and update 4,608 beat the final one.</p>
            </li>
          </ol>
          <div className="architecture" aria-label="Network architecture">
            <div><small>Input</small><strong>8 × 8 × 8</strong></div>
            <span aria-hidden="true">→</span>
            <div><small>Stem</small><strong>3×3 conv · 64</strong></div>
            <span aria-hidden="true">→</span>
            <div><small>Trunk</small><strong>6 residual blocks</strong></div>
            <span aria-hidden="true">→</span>
            <div className="architecture__heads">
              <div><small>Policy head</small><strong>128 action logits</strong></div>
              <div><small>Value head</small><strong>tanh value</strong></div>
            </div>
          </div>
          <p className="section-note">
            Illegal moves never reach the network's choice: the rules engine builds a 128-slot legal-action
            mask, and the policy picks only among the moves it allows.
          </p>
        </section>

        <section className="section results" id="results" aria-labelledby="results-heading">
          <div className="results__copy">
            <p className="kicker">Results</p>
            <h2 id="results-heading">The model on this page is update 4,608.</h2>
            <p>
              It was selected from a 6,144-update practice run as the best fully evaluated checkpoint. It
              had trained on 37,748,736 self-play transitions. Each score below comes from 216 fixed
              openings, played from both colors.
            </p>
            <p className="caveat">
              These are project evaluation results, not a human skill rating. The same openings helped
              select the checkpoint, and Minimax-2 is a shallow internal baseline.
            </p>
          </div>
          <dl className="scoreboard" aria-label="Selected checkpoint evaluation results">
            <div>
              <dt>vs. random</dt>
              <dd><strong>432–0–0</strong><small>wins · draws · losses</small></dd>
            </div>
            <div>
              <dt>vs. Minimax-2</dt>
              <dd><strong>354–70–8</strong><small>wins · draws · losses</small></dd>
            </div>
            <div>
              <dt>Training to this checkpoint</dt>
              <dd><strong>37.7M</strong><small>self-play transitions</small></dd>
            </div>
          </dl>
        </section>

        <section className="section in-browser" aria-labelledby="browser-heading">
          <div className="section-heading">
            <p className="kicker">Under the hood</p>
            <h2 id="browser-heading">The same model, no server.</h2>
          </div>
          <div className="feature-grid">
            <article>
              <strong>Exported from PyTorch</strong>
              <p>The trained bundle is converted to {megabytes} MB of raw float32 weights with a checksummed manifest, then evaluated by a dependency-free TypeScript forward pass in a Web Worker.</p>
            </article>
            <article>
              <strong>Parity-tested</strong>
              <p>The TypeScript rules engine and network are checked against the Python engine and PyTorch: 43 recorded games, 250+ greedy decisions, and published perft counts.</p>
            </article>
            <article>
              <strong>Honest by design</strong>
              <p>The AI always plays its highest-scoring legal move. The evaluation shown during play is the raw value-head output, not a calibrated win probability.</p>
            </article>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <a className="brand" href="#top" aria-label="Back to top">
          <LogoMark />
          <span>PPO Checkers</span>
        </a>
        <p>PyTorch · PPO self-play · React · Vite. Open source under the MIT License.</p>
        <a href={REPOSITORY_URL} target="_blank" rel="noreferrer">GitHub</a>
      </footer>
    </div>
  );
}

export default App;
