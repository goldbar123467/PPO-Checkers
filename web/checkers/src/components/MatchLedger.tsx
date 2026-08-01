import type { GameSnapshot } from "../types";

const REASONS: Record<string, string> = {
  no_pieces: "no pieces remain",
  no_legal_move: "no legal move",
  no_progress: "40-move no-progress rule",
  repetition: "threefold repetition",
  ply_cap: "512-ply safety cap",
};

interface MatchLedgerProps {
  game: GameSnapshot;
}

interface GameStatusProps extends MatchLedgerProps {
  busy: boolean;
}

function statusText(game: GameSnapshot, busy: boolean): string {
  if (busy) return "The policy server is selecting an action…";
  if (game.outcome) {
    if (game.outcome.isDraw) return `Draw · ${REASONS[game.outcome.reason] ?? game.outcome.reason}`;
    return `${game.outcome.winner === game.humanColor ? "You win!" : "The saved policy wins"} · ${
      REASONS[game.outcome.reason] ?? game.outcome.reason
    }`;
  }
  if (game.captureInProgress) return "Keep jumping with the glowing piece!";
  return game.isHumanTurn ? "Your turn—pick a glowing piece." : "Saved policy's turn.";
}

function teamName(color: "red" | "white"): string {
  return color === "red" ? "orange" : "white";
}

export function GameStatus({ game, busy }: GameStatusProps) {
  return (
    <section className="simple-panel game-status" aria-labelledby="game-status-heading">
      <div className="turn-card" role="status" aria-live="polite" aria-atomic="true">
        <p className="panel-label">{game.outcome ? `${game.ply} turns completed` : `Turn ${game.ply + 1}`}</p>
        <h2 id="game-status-heading">{statusText(game, busy)}</h2>
        <p>
          You are <strong>{teamName(game.humanColor)}</strong>. The trained policy plays
          {` ${teamName(game.modelColor)}`} and always chooses its highest-scoring legal move.
        </p>
      </div>
    </section>
  );
}

export function MatchLedger({ game }: MatchLedgerProps) {
  return (
    <details className="simple-panel ledger">
      <summary>
        <span id="ledger-heading">Move history</span>
        <small>{game.moves.length} {game.moves.length === 1 ? "move" : "moves"}</small>
      </summary>
      {game.moves.length === 0 ? <p className="empty-ledger">Moves will appear here.</p> : (
          <ol className="move-list" aria-labelledby="ledger-heading">
            {game.moves.map((record) => (
              <li key={`${record.ply}-${record.notation}`}>
                <span>{record.ply.toString().padStart(2, "0")}</span>
                <span className={`move-color move-color--${record.actor}`} aria-hidden="true" />
                <strong>{record.notation}</strong>
                <small>{teamName(record.actor)}</small>
              </li>
            ))}
          </ol>
        )}
    </details>
  );
}
