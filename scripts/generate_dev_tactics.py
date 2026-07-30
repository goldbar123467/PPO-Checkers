"""Generate the versioned 50-case reachable checkers development tactical suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from checkers.agents.minimax_agent import MinimaxAgent
from checkers.env.masking import legal_action_map
from checkers.eval.suites import (
    DEV_TACTICAL_CASES,
    TacticalCase,
    TacticalSuite,
    TacticalSuiteManifest,
    forced_win_actions,
    tactical_case_record,
    tactical_cases_sha256,
)
from checkers.rules.board import PLAYABLE_SQUARES, bit, rotate_square
from checkers.rules.moves import apply_step
from checkers.rules.state import State
from checkers.rules.terminal import terminal_outcome

DEFAULT_SEED = 20_260_728
DEFAULT_HORIZON = 3
DEFAULT_MAX_GAMES = 10_000
MAX_POSITION_PLY = 480
MAX_PIECES = 10
MAX_LEGAL_ACTIONS = 8
MIN_LEGAL_ACTIONS = 2
MINIMUM_DEPTH1_MISSES = 5
DEFAULT_OUTPUT = Path("src/checkers/eval/data/dev_tactics_v1.json")
RATIONALE = (
    "Exhaustive terminal-only AND/OR search proves the frozen root-action set forces a win "
    "within three completed moves against every defense; no material evaluator is used."
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Deterministic bounded scan configuration."""

    seed: int = DEFAULT_SEED
    horizon: int = DEFAULT_HORIZON
    max_games: int = DEFAULT_MAX_GAMES
    target_cases: int = DEV_TACTICAL_CASES
    minimum_depth1_misses: int = MINIMUM_DEPTH1_MISSES

    def __post_init__(self) -> None:
        for name, value in (
            ("seed", self.seed),
            ("horizon", self.horizon),
            ("max_games", self.max_games),
            ("target_cases", self.target_cases),
            ("minimum_depth1_misses", self.minimum_depth1_misses),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if min(self.horizon, self.max_games, self.target_cases) < 1:
            raise ValueError("horizon, max_games, and target_cases must be positive")
        if not 0 <= self.minimum_depth1_misses <= self.target_cases:
            raise ValueError("minimum_depth1_misses must be within target_cases")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Selected cases and exact scan diagnostics."""

    cases: tuple[TacticalCase, ...]
    games_scanned: int
    depth1_misses: int


def _rotate_mask(mask: int) -> int:
    rotated = 0
    for square in range(PLAYABLE_SQUARES):
        if mask & bit(square):
            rotated |= bit(rotate_square(square))
    return rotated


def _duplicate_group(state: State) -> str:
    original = {
        "men": state.men,
        "kings": state.kings,
        "side": int(state.side_to_move),
    }
    transformed = {
        "men": (_rotate_mask(state.men[1]), _rotate_mask(state.men[0])),
        "kings": (_rotate_mask(state.kings[1]), _rotate_mask(state.kings[0])),
        "side": int(state.side_to_move.opponent),
    }
    candidates = [
        json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in (original, transformed)
    ]
    return hashlib.sha256(min(candidates).encode("utf-8")).hexdigest()


def _candidate(state: State) -> bool:
    legal_count = len(legal_action_map(state))
    return (
        not state.capture_in_progress
        and MIN_LEGAL_ACTIONS <= legal_count <= MAX_LEGAL_ACTIONS
        and state.occupied.bit_count() <= MAX_PIECES
        and state.ply < MAX_POSITION_PLY
    )


def generate_cases(config: GenerationConfig) -> GenerationResult:
    """Scan seeded reachable playouts and select exact nontrivial tactics.

    Args:
        config: Fixed seed, horizon, scan cap, and target criteria.

    Returns:
        Selected cases plus games scanned and depth-one miss count.

    Raises:
        RuntimeError: If the bounded scan cannot satisfy the requested criteria.
    """

    rng = random.Random(config.seed)
    cases: list[TacticalCase] = []
    duplicate_groups: set[str] = set()
    depth1_misses = 0
    games_scanned = 0

    for game_index in range(config.max_games):
        games_scanned = game_index + 1
        state = State.initial()
        replay_actions: list[int] = []
        while terminal_outcome(state) is None:
            action_map = legal_action_map(state)
            if _candidate(state):
                duplicate_group = _duplicate_group(state)
                if duplicate_group not in duplicate_groups:
                    winning_actions = forced_win_actions(
                        state,
                        max_completed_moves=config.horizon,
                    )
                    if winning_actions and len(winning_actions) < len(action_map):
                        deep_action = MinimaxAgent(
                            depth=config.horizon, seed=config.seed
                        ).select_action(state)
                        if deep_action in winning_actions:
                            shallow_action = MinimaxAgent(depth=1, seed=config.seed).select_action(
                                state
                            )
                            minimum_horizon = next(
                                horizon
                                for horizon in range(1, config.horizon + 1)
                                if forced_win_actions(
                                    state,
                                    max_completed_moves=horizon,
                                )
                            )
                            case = TacticalCase(
                                case_id=f"dev-tactic-{len(cases) + 1:03d}",
                                state=state,
                                max_completed_moves=config.horizon,
                                winning_actions=winning_actions,
                                replay_actions=tuple(replay_actions),
                                rationale=RATIONALE,
                                duplicate_group=duplicate_group,
                                source_game=game_index,
                                source_step=len(replay_actions),
                                difficulty=(
                                    f"forced_win_first_detected_at_{minimum_horizon}_completed_moves"
                                ),
                            )
                            cases.append(case)
                            duplicate_groups.add(duplicate_group)
                            depth1_misses += int(shallow_action not in winning_actions)
                            if (
                                len(cases) == config.target_cases
                                and depth1_misses >= config.minimum_depth1_misses
                            ):
                                return GenerationResult(
                                    cases=tuple(cases),
                                    games_scanned=games_scanned,
                                    depth1_misses=depth1_misses,
                                )

            selected_action = rng.choice(tuple(action_map))
            state = apply_step(state, action_map[selected_action]).after
            replay_actions.append(selected_action)

    raise RuntimeError(
        f"scan exhausted {config.max_games} games with {len(cases)} cases and "
        f"{depth1_misses} depth-one misses"
    )


def _sha256_files(paths: tuple[Path, ...], *, repository: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative_path = path.resolve().relative_to(repository.resolve())
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_suite(result: GenerationResult, *, script_path: Path) -> TacticalSuite:
    """Attach immutable provenance and content hashes to generated cases.

    Args:
        result: Successful exact generation result.
        script_path: Generator source path for provenance hashing.

    Returns:
        Fully reverified tactical suite.
    """

    repository = script_path.resolve().parents[1]
    rules_paths = tuple((repository / "src/checkers/rules").glob("*.py")) + (
        repository / "src/checkers/env/masking.py",
    )
    manifest = TacticalSuiteManifest(
        schema_version=1,
        name="checkers_dev_tactics_v1",
        case_count=len(result.cases),
        horizon_completed_moves=DEFAULT_HORIZON,
        generator_seed=DEFAULT_SEED,
        games_scanned=result.games_scanned,
        depth1_misses=result.depth1_misses,
        depth3_solved=len(result.cases),
        provenance=(
            "Positions are exact completed-move states reached by seeded uniform-random legal "
            "play from State.initial(); replay_actions proves each prefix."
        ),
        license="CC0-1.0",
        author="PPO Checkers deterministic generator",
        creation_method=(
            "scripts/generate_dev_tactics.py; exact terminal-only AND/OR labels; depth-3 "
            "membership check; 180-degree colour-swap symmetry deduplication"
        ),
        review_status="programmatically_verified_pending_human_review",
        grade_band="not_applicable_game_state",
        safety_categories=("none",),
        subject_categories=("american_checkers_tactics",),
        difficulty="forced_win_within_three_completed_moves",
        split="dev",
        cases_sha256=tactical_cases_sha256(result.cases),
        generator_source_sha256=hashlib.sha256(script_path.read_bytes()).hexdigest(),
        rules_source_sha256=_sha256_files(rules_paths, repository=repository),
        goal_sha256=hashlib.sha256(
            (repository / "docs/experiment-contract.md").read_bytes()
        ).hexdigest(),
    )
    return TacticalSuite(manifest=manifest, cases=result.cases)


def _json_document(suite: TacticalSuite) -> dict[str, object]:
    manifest = asdict(suite.manifest)
    manifest["safety_categories"] = list(suite.manifest.safety_categories)
    manifest["subject_categories"] = list(suite.manifest.subject_categories)
    return {
        "manifest": manifest,
        "cases": [tactical_case_record(case) for case in suite.cases],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    """Generate, reverify, and atomically write the canonical suite."""

    arguments = _parser().parse_args()
    output = arguments.output
    if output.exists() and not arguments.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    result = generate_cases(GenerationConfig())
    suite = build_suite(result, script_path=Path(__file__))
    text = json.dumps(_json_document(suite), indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(suite.cases),
                "games_scanned": suite.manifest.games_scanned,
                "depth1_misses": suite.manifest.depth1_misses,
                "depth3_solved": suite.manifest.depth3_solved,
                "cases_sha256": suite.manifest.cases_sha256,
                "file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
