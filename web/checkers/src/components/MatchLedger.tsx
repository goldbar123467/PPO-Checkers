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
  busy: boolean;
}

function statusText(game: GameSnapshot, busy: boolean): string {
  if (busy) return "The model is considering its reply…";
  if (game.outcome) {
    if (game.outcome.isDraw) return `Draw · ${REASONS[game.outcome.reason] ?? game.outcome.reason}`;
    return `${game.outcome.winner === game.humanColor ? "You win" : "Model wins"} · ${
      REASONS[game.outcome.reason] ?? game.outcome.reason
    }`;
  }
  if (game.captureInProgress) return "Continue the forced capture with the highlighted piece.";
  return game.isHumanTurn ? "Your move. Select a ringed piece." : "Model turn.";
}

export function MatchLedger({ game, busy }: MatchLedgerProps) {
  return (
    <section className="panel ledger" aria-labelledby="ledger-heading">
      <div className="turn-card" aria-live="polite">
        <p className="eyebrow">Position · ply {game.ply} · seed {game.seed}</p>
        <h2>{statusText(game, busy)}</h2>
        <p>
          You are <strong>{game.humanColor}</strong>. The trained neural policy plays{" "}
          {game.modelColor} in{" "}
          {game.policyMode === "greedy" ? "deterministic greedy" : "seeded sampled"} mode.
        </p>
      </div>

      <div className="ledger-heading">
        <h3 id="ledger-heading">Move ledger</h3>
        <span>{game.moves.length} moves</span>
      </div>
      {game.moves.length === 0 ? (
        <p className="empty-ledger">The opening line will appear here.</p>
      ) : (
        <ol className="move-list">
          {game.moves.map((move) => (
            <li key={`${move.ply}-${move.notation}`}>
              <span>{move.ply.toString().padStart(2, "0")}</span>
              <span className={`move-color move-color--${move.actor}`} aria-hidden="true" />
              <strong>{move.notation}</strong>
              <small>{move.actor}</small>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
