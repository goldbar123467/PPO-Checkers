import type { Color, GameSnapshot, TerminationReason } from "@/types";

const REASONS: Record<TerminationReason, string> = {
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
  if (game.outcome) {
    const reason = REASONS[game.outcome.reason];
    if (game.outcome.isDraw) return `Draw · ${reason}`;
    return `${game.outcome.winner === game.humanColor ? "You win!" : "The AI wins"} · ${reason}`;
  }
  if (busy || !game.isHumanTurn) return "The AI is choosing a move…";
  if (game.captureInProgress) return "Keep jumping with the highlighted piece!";
  return "Your turn. Pick a highlighted piece.";
}

export function teamName(color: Color): string {
  return color === "red" ? "Red" : "Black";
}

export function GameStatus({ game, busy }: GameStatusProps) {
  return (
    <section className="panel game-status" aria-labelledby="game-status-heading">
      <div className="turn-card" role="status" aria-live="polite" aria-atomic="true">
        <p className="panel-label">
          {game.outcome ? `Game over after ${game.moves.length} moves` : `Move ${game.moves.length + 1}`}
        </p>
        <h2 id="game-status-heading">{statusText(game, busy)}</h2>
        <p>
          You are <strong>{teamName(game.humanColor)}</strong>. The AI plays{" "}
          {teamName(game.modelColor)} and always picks its highest-scoring legal move.
        </p>
      </div>
    </section>
  );
}

export function MatchLedger({ game }: MatchLedgerProps) {
  return (
    <details className="panel ledger">
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
