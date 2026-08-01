import { useMutation, useQuery } from "@tanstack/react-query";
import { useLayoutEffect, useRef, useState } from "react";

import { Board } from "@/components/Board";
import { GameControls } from "@/components/GameControls";
import { GameStatus, MatchLedger } from "@/components/MatchLedger";
import { createGame, fetchModel, submitMove } from "@/lib/api/checkers";
import { ApiError } from "@/lib/api/client";
import type { Color, GameSnapshot } from "@/lib/api/schemas";

function randomMatchSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

function App() {
  const model = useQuery({
    queryKey: ["policy-model"],
    queryFn: fetchModel,
    staleTime: Infinity,
  });
  const [game, setGame] = useState<GameSnapshot | null>(null);
  const [humanColor, setHumanColor] = useState<Color>("red");
  const [highContrast, setHighContrast] = useState(false);
  const tableRef = useRef<HTMLElement>(null);
  const start = useMutation({
    mutationFn: () => createGame(humanColor, "greedy", randomMatchSeed()),
    onSuccess: setGame,
  });
  const move = useMutation({
    mutationFn: ({ origin, destination }: { origin: number; destination: number }) => {
      if (!game) throw new Error("Start a game before making a move.");
      return submitMove(game.id, origin, destination);
    },
    onSuccess: setGame,
  });
  const gameId = game?.id;
  const busy = start.isPending || move.isPending;
  const error = model.error ?? start.error ?? move.error;

  useLayoutEffect(() => {
    if (gameId) tableRef.current?.focus({ preventScroll: true });
  }, [gameId]);

  function startNewGame() {
    move.reset();
    start.reset();
    setGame(null);
    start.mutate();
  }

  function recoverFromError() {
    if (model.isError) {
      void model.refetch();
      return;
    }
    if (start.isError) {
      startNewGame();
      return;
    }
    move.reset();
  }

  return (
    <div className={highContrast ? "site-shell board-high-contrast" : "site-shell"}>
      <a className="skip-link" href="#play-table">
        Skip to the checkers game
      </a>

      <header className="site-header">
        <a className="school-brand" href="#top" aria-label="IMSA West Checkers AI home">
          <img
            src="/assets/imsa-west-logo.png"
            width="447"
            height="447"
            alt="Indiana Math and Science Academy West logo"
          />
          <span>
            <strong>IMSA West</strong>
            <small>Checkers AI</small>
          </span>
        </a>
        <nav aria-label="Page navigation">
          <a href="#play-table">Play</a>
          <a href="#how-it-learns">How it learns</a>
          <a href="#real-results">Results</a>
        </nav>
      </header>

      <main id="top">
        <section className="game-intro" aria-labelledby="page-title">
          <div>
            <p className="kicker">Built at IMSA West · powered by a real PPO policy</p>
            <h1 id="page-title">Can you beat our checkers AI?</h1>
            <p>
              Pick a side, start the game, then tap an outlined piece and a dotted square.
              The same Python rules engine used during training checks every move.
            </p>
          </div>
          <div className="policy-online" role="status">
            <span aria-hidden="true" />
            {model.data ? `Policy update ${model.data.update.toLocaleString()} ready` : "Loading the saved policy…"}
          </div>
        </section>

        {error ? (
          <section className="error-message" role="alert">
            <div>
              <strong>The game server needs attention.</strong>
              <p>{error instanceof Error ? error.message : "The saved policy could not be reached."}</p>
              {error instanceof ApiError ? <small>Error code: {error.code}</small> : null}
            </div>
            <button type="button" onClick={recoverFromError}>Try again</button>
          </section>
        ) : null}

        {model.isPending ? (
          <section className="loading-state" aria-live="polite">
            <span className="loading-piece" aria-hidden="true" />
            <div>
              <strong>Setting up the board…</strong>
              <p>The server is loading and checking the trained policy.</p>
            </div>
          </section>
        ) : null}

        {model.data ? (
          <section className="game-workspace" id="play-table" aria-label="Play checkers">
            <GameControls
              model={model.data}
              humanColor={humanColor}
              busy={busy}
              hasGame={Boolean(game)}
              onHumanColor={setHumanColor}
              onStart={startNewGame}
            />

            <section className="board-stage" aria-label="Checkers game table" ref={tableRef} tabIndex={-1}>
              {game ? (
                <Board
                  game={game}
                  busy={busy}
                  onMove={(origin, destination) => move.mutate({ origin, destination })}
                />
              ) : (
                <div className="board-placeholder">
                  <div className="mini-board" aria-hidden="true">
                    {Array.from({ length: 16 }, (_, index) => <span key={index} />)}
                  </div>
                  <strong>Your board is ready.</strong>
                  <p>Choose orange or white, then press Start game.</p>
                </div>
              )}
            </section>

            <aside className="game-sidebar" aria-label="Game help and move history">
              {game ? <GameStatus game={game} busy={busy} /> : (
                <section className="simple-panel quick-guide">
                  <p className="panel-label">Three quick rules</p>
                  <ol>
                    <li><span>1</span>Move diagonally on blue squares.</li>
                    <li><span>2</span>If you can jump, you must jump.</li>
                    <li><span>3</span>Reach the far side to become a king.</li>
                  </ol>
                </section>
              )}
              <section className="simple-panel display-option">
                <label>
                  <input
                    type="checkbox"
                    checked={highContrast}
                    onChange={(event) => setHighContrast(event.target.checked)}
                  />
                  Stronger board contrast
                </label>
              </section>
              {game ? <MatchLedger game={game} /> : null}
            </aside>
          </section>
        ) : null}

        <section className="learning-section" id="how-it-learns" aria-labelledby="learning-heading">
          <div className="section-heading">
            <p className="kicker">Behind the board</p>
            <h2 id="learning-heading">How did the AI learn to play?</h2>
            <p>
              It was not given a list of perfect moves. It practiced against copies of itself and
              used PPO to adjust which legal moves were more likely.
            </p>
          </div>
          <ol className="learning-path">
            <li>
              <span>01</span>
              <div><strong>Read the board</strong><p>Eight actor-centered number layers describe pieces, kings, capture continuation, and game counters. A separate 128-slot mask marks legal actions.</p></div>
            </li>
            <li>
              <span>02</span>
              <div><strong>Play itself</strong><p>Self-play produces complete games—wins, losses, draws, captures, and quiet moves.</p></div>
            </li>
            <li>
              <span>03</span>
              <div><strong>Update with PPO</strong><p>PPO compares the new policy with the policy that collected each move and limits oversized changes.</p></div>
            </li>
            <li>
              <span>04</span>
              <div><strong>Test saved versions</strong><p>Later training was not always better, so checkpoints were tested with the same fixed match protocol.</p></div>
            </li>
          </ol>
        </section>

        <section className="results-section" id="real-results" aria-labelledby="results-heading">
          <div className="results-copy">
            <p className="kicker">Real project evidence</p>
            <h2 id="results-heading">The model on this page is update 4,608.</h2>
            <p>
              That checkpoint was selected from the saved practice run. It uses a 470,410-parameter
              policy network trained through 37,748,736 self-play transitions.
            </p>
            <p className="results-caveat">
              These are project evaluation results, not a human skill rating. The same evaluation
              set helped select the checkpoint, and Minimax-2 is a shallow project baseline.
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
              <dt>To this checkpoint</dt>
              <dd><strong>37.7M</strong><small>self-play transitions</small></dd>
            </div>
          </dl>
        </section>

        <section className="teacher-note" aria-labelledby="teacher-note-heading">
          <div className="teacher-note__mark" aria-hidden="true">MK</div>
          <div>
            <p className="kicker">Mr. Kitchen’s note</p>
            <h2 id="teacher-note-heading">The AI chooses moves. The rules engine keeps the game honest.</h2>
            <p>
              The neural network scores possible actions, but illegal moves are masked out before
              selection. It does not think or understand the board—it maps numbers to action scores.
            </p>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <img src="/assets/imsa-west-logo.png" width="447" height="447" alt="" />
        <p><strong>IMSA West Checkers AI</strong><br />A real student-facing machine-learning project.</p>
        <a href="#top">Back to top ↑</a>
      </footer>
    </div>
  );
}

export default App;
