"""Versioned, resumable, power-validated baseline evaluation reporting."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import yaml

from checkers.eval import baseline_eval
from checkers.eval.arena import MatchResult, play_balanced_match
from checkers.eval.baseline_eval import (
    BASELINE_AGENTS,
    BaselineConfig,
    BaselineMatchSummary,
    agent_spec,
    build_evaluation_report,
    build_population_report,
    build_tactical_report,
    load_baseline_config,
    match_result_record,
    parse_match_result_record,
    summarize_match,
)
from checkers.eval.elo import EloEstimate
from checkers.eval.power import MatchScore, score_interval
from checkers.eval.suites import TacticalCase, load_dev_tactical_suite
from checkers.rules.state import PlayerId, State

CONFIG_PATH = Path("configs/checkers-baselines-v1.yaml")
EXPECTED_GAMES = 784
EXPECTED_COMPARISONS = 6
EXPECTED_RAW_GAMES = 783
SMALL_GAMES = 2
NEUTRAL_SCORE = 0.5
SHA256_LENGTH = 64
TACTICAL_CASES = 50
INTERMEDIATE_DEPTH = 2


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _forced_red_win_state() -> State:
    return State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _small_match(
    first: str = "greedy",
    second: str = "random",
    seed: int = 7,
) -> MatchResult:
    return play_balanced_match(
        first=agent_spec(first, max_plies=512),
        second=agent_spec(second, max_plies=512),
        games=2,
        seed=seed,
        initial_state=_forced_red_win_state(),
    )


def _small_round_robin() -> tuple[MatchResult, ...]:
    matches: list[MatchResult] = []
    seed = 100
    for index, (first, second) in enumerate(
        (
            ("greedy", "random"),
            ("minimax(1)", "random"),
            ("minimax(2)", "random"),
            ("minimax(1)", "greedy"),
            ("minimax(2)", "greedy"),
            ("minimax(2)", "minimax(1)"),
        )
    ):
        matches.append(_small_match(first, second, seed + index))
    return tuple(matches)


def _config_document(**updates: object) -> str:
    loaded: object = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document = cast(dict[str, object], loaded)
    document.update(updates)
    return yaml.safe_dump(document, sort_keys=True)


def test_reviewed_baseline_config_is_exactly_power_sized_and_complete() -> None:
    config = load_baseline_config(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config.experiment_id == "checkers-baselines-v1"
    assert config.games_per_match == EXPECTED_GAMES
    assert config.power_plan.raw_games == EXPECTED_RAW_GAMES
    assert config.power_plan.balanced_games == EXPECTED_GAMES
    assert len(config.comparisons) == EXPECTED_COMPARISONS
    assert {agent for pair in config.comparisons for agent in pair} == set(BASELINE_AGENTS)
    assert config.experiment_label == "baseline"


def test_each_supported_agent_spec_builds_exact_declared_agent() -> None:
    for name in BASELINE_AGENTS:
        spec = agent_spec(name, max_plies=512)
        assert spec.build(1).name == name


def test_match_result_record_round_trips_every_replay_field() -> None:
    match = _small_match()

    record = match_result_record(match)
    restored = parse_match_result_record(record)

    assert restored == match
    assert restored.records[0].actions
    assert restored.records[0].moves


def test_match_summary_contains_ci_elo_colour_and_terminal_counts() -> None:
    summary = summarize_match(_small_match(), elapsed_seconds=0.25)

    assert isinstance(summary, BaselineMatchSummary)
    assert summary.games == SMALL_GAMES
    assert summary.first_as_red_games == 1
    assert summary.score.score == NEUTRAL_SCORE
    assert summary.elo.difference == 0.0
    assert sum(count for _reason, count in summary.termination_counts) == SMALL_GAMES
    assert len(summary.records_sha256) == SHA256_LENGTH


def test_population_report_uses_all_six_edges_and_marks_proxy_not_evaluated() -> None:
    report = build_population_report(_small_round_robin())
    pairs = cast(list[object], report["pairs"])
    cycles = cast(dict[str, object], report["three_cycles"])
    proxy = cast(dict[str, object], report["exploitability_proxy"])
    external = cast(dict[str, object], report["external_anchor"])

    assert report["agents"] == list(BASELINE_AGENTS)
    assert len(pairs) == EXPECTED_COMPARISONS
    assert cycles["count"] == 0
    assert proxy["status"] == "NOT_EVALUATED"
    assert external["status"] == "NOT_AVAILABLE"


def test_tactical_report_preserves_exact_depth_superset_evidence() -> None:
    report = build_tactical_report(seed=20260728, depths=(1, 2, 3))
    evaluations = cast(dict[str, object], report["evaluations"])
    depth_three = cast(dict[str, object], evaluations["minimax(3)"])
    comparison = cast(dict[str, object], report["depth_1_vs_3"])

    assert report["suite_case_count"] == TACTICAL_CASES
    assert depth_three["solved"] == TACTICAL_CASES
    assert comparison["deep_is_superset"] is True
    assert comparison["passes_gate"] is True


def test_tactical_tie_breaks_are_isolated_per_case_and_order_independent() -> None:
    suite = load_dev_tactical_suite()

    forward = baseline_eval._evaluate_minimax_tactical(
        depth=1,
        seed=20260728,
        cases=suite.cases,
    )
    reverse = baseline_eval._evaluate_minimax_tactical(
        depth=1,
        seed=20260728,
        cases=tuple(reversed(suite.cases)),
    )

    assert forward.solved == suite.manifest.case_count - suite.manifest.depth1_misses
    assert set(forward.solved_case_ids) == set(reverse.solved_case_ids)


def test_tactical_report_matches_generator_isolation_contract() -> None:
    suite = load_dev_tactical_suite()
    report = build_tactical_report(seed=suite.manifest.generator_seed, depths=(1, 2, 3))
    evaluations = cast(dict[str, object], report["evaluations"])
    depth_one = cast(dict[str, object], evaluations["minimax(1)"])

    assert depth_one["solved"] == suite.manifest.case_count - suite.manifest.depth1_misses
    assert report["case_evaluation_seed_policy"] == "fresh policy with the declared seed per case"


def test_final_report_has_sources_hashes_assumptions_and_gate_verdicts() -> None:
    config = load_baseline_config(CONFIG_PATH.read_text(encoding="utf-8"))
    matches = _small_round_robin()
    summaries = tuple(summarize_match(match, elapsed_seconds=0.1) for match in matches)
    report = build_evaluation_report(
        config=config,
        matches=matches,
        summaries=summaries,
        tactical=build_tactical_report(seed=config.seed, depths=config.tactical_depths),
        git_commit="a" * 40,
        config_sha256="b" * 64,
        goal_sha256="c" * 64,
        raw_games_sha256="d" * 64,
        hardware={"execution_device": "cpu"},
        dependencies={"python": "3.12"},
    )
    gate = cast(dict[str, object], report["gate_5"])
    sources = cast(list[dict[str, object]], report["sources"])
    assumptions = cast(list[object], report["statistical_assumptions"])
    non_monotonicity = cast(dict[str, object], report["search_depth_non_monotonicity"])
    tactical_depths = cast(
        list[dict[str, object]],
        non_monotonicity["tactical_depth_observations"],
    )

    assert gate["power_justified"] is True
    assert gate["no_catastrophic_inversion"] is True
    assert gate["tactical_superset_or_substantially_more"] is True
    assert sources
    assert all(cast(str, source["url"]).startswith("https://") for source in sources)
    assert assumptions
    assert non_monotonicity["any_point_estimate_non_monotonicity"] is True
    assert tactical_depths[0]["shallower_depth"] == 1
    assert tactical_depths[0]["deeper_depth"] == INTERMEDIATE_DEPTH
    assert tactical_depths[0]["deeper_point_estimate_is_lower"] is True
    assert non_monotonicity["diagnosis"]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: load_baseline_config("[]"), "mapping"),
        (lambda: load_baseline_config("schema_version: 2"), "schema_version"),
        (lambda: agent_spec("unknown", max_plies=512), "unknown"),
        (lambda: agent_spec("random", max_plies=0), "max_plies"),
        (lambda: match_result_record(cast(MatchResult, "match")), "MatchResult"),
        (lambda: parse_match_result_record([]), "mapping"),
        (
            lambda: summarize_match(cast(MatchResult, "match"), elapsed_seconds=1.0),
            "MatchResult",
        ),
        (lambda: summarize_match(_small_match(), elapsed_seconds=-1.0), "elapsed"),
        (lambda: build_population_report((_small_match(),)), "complete"),
        (lambda: build_tactical_report(seed=0, depths=()), "depths"),
    ],
)
def test_baseline_evaluation_functions_reject_invalid_inputs(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_baseline_config_and_summary_records_reject_corruption() -> None:
    config = load_baseline_config(CONFIG_PATH.read_text(encoding="utf-8"))
    summary = summarize_match(_small_match(), elapsed_seconds=0.1)

    with pytest.raises(ValueError, match="games_per_match"):
        replace(config, games_per_match=2)
    with pytest.raises(ValueError, match="comparisons"):
        replace(config, comparisons=config.comparisons[:-1])
    with pytest.raises(ValueError, match="records_sha256"):
        replace(summary, records_sha256="bad")
    with pytest.raises(ValueError, match="colour"):
        replace(summary, first_as_red_games=0)


def test_baseline_config_public_type_is_explicit() -> None:
    config: BaselineConfig = load_baseline_config(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config.repetition_draws is True


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "mapping"),
        ("schema_version: 1", "missing required"),
        (_config_document(agents="random"), "list"),
        (_config_document(comparisons=[["random"]]), "exactly two"),
        (_config_document(seed=True), "integer"),
        (_config_document(seed=-1), "unsigned"),
        (_config_document(games_per_match=0), "positive"),
        (_config_document(confidence="high"), "numeric"),
        (_config_document(confidence=float("inf")), "finite"),
        (_config_document(confidence=1.0), "strictly between"),
        (_config_document(repetition_draws=1), "bool"),
    ],
)
def test_config_parser_rejects_each_primitive_schema_corruption(
    text: str,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        load_baseline_config(text)


def test_config_parser_requires_text() -> None:
    with pytest.raises(TypeError, match="text"):
        load_baseline_config(cast(str, b"yaml"))


def test_config_parser_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        load_baseline_config(_config_document(unexpected_field=True))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda config: replace(config, schema_version=2), "schema_version"),
        (lambda config: replace(config, experiment_label="trial"), "experiment_label"),
        (lambda config: replace(config, smallest_effect=0.6), "below one"),
        (lambda config: replace(config, agents=("random",)), "agents"),
        (
            lambda config: replace(
                config,
                comparisons=cast(tuple[tuple[str, str], ...], []),
            ),
            "tuple",
        ),
        (
            lambda config: replace(
                config,
                comparisons=cast(tuple[tuple[str, str], ...], (("random",),)),
            ),
            "two-name",
        ),
        (
            lambda config: replace(config, comparisons=(("random", "random"),)),
            "distinct",
        ),
        (lambda config: replace(config, tactical_depths=(1,)), "tactical_depths"),
        (lambda config: replace(config, tactical_depths=(2, 1)), "tactical_depths"),
        (lambda config: replace(config, tactical_depths=(1, 1)), "tactical_depths"),
        (
            lambda config: replace(config, seed=baseline_eval.UINT64_MAX),
            "insufficient uint64",
        ),
    ],
)
def test_config_record_rejects_semantic_corruption(
    factory: Callable[[BaselineConfig], BaselineConfig],
    message: str,
) -> None:
    config = load_baseline_config(CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises((TypeError, ValueError), match=message):
        factory(config)


def test_agent_and_record_parsers_reject_primitive_corruption() -> None:
    with pytest.raises(TypeError, match="string"):
        agent_spec(cast(str, 1), max_plies=512)
    with pytest.raises(ValueError, match="empty"):
        agent_spec("", max_plies=512)

    record = copy.deepcopy(match_result_record(_small_match()))
    state = cast(dict[str, object], record["initial_state"])
    state["men"] = [1]
    with pytest.raises(ValueError, match="two integers"):
        parse_match_result_record(record)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda summary: replace(summary, second_agent="greedy"), "distinct"),
        (lambda summary: replace(summary, games=3), "even"),
        (
            lambda summary: replace(summary, score=cast(MatchScore, "bad")),
            "MatchScore",
        ),
        (
            lambda summary: replace(
                summary,
                score=score_interval(wins=1, draws=0, losses=0),
            ),
            "game count",
        ),
        (
            lambda summary: replace(summary, elo=cast(EloEstimate, "bad")),
            "EloEstimate",
        ),
        (lambda summary: replace(summary, mean_steps=-1.0), "non-negative"),
        (
            lambda summary: replace(
                summary,
                termination_counts=cast(tuple[tuple[str, int], ...], []),
            ),
            "tuple",
        ),
        (
            lambda summary: replace(
                summary,
                termination_counts=cast(tuple[tuple[str, int], ...], (("one",),)),
            ),
            "name/count",
        ),
        (
            lambda summary: replace(
                summary,
                termination_counts=(("one", 1), ("one", 1)),
            ),
            "unique",
        ),
        (
            lambda summary: replace(summary, termination_counts=(("one", -1),)),
            "non-negative",
        ),
        (
            lambda summary: replace(summary, termination_counts=(("one", 1),)),
            "sum",
        ),
    ],
)
def test_match_summary_rejects_semantic_corruption(
    factory: Callable[[BaselineMatchSummary], BaselineMatchSummary],
    message: str,
) -> None:
    summary = summarize_match(_small_match(), elapsed_seconds=0.1)
    with pytest.raises((TypeError, ValueError), match=message):
        factory(summary)


def test_infinite_elo_values_receive_standard_json_labels() -> None:
    assert baseline_eval._finite_or_label(float("inf")) == "+Infinity"
    assert baseline_eval._finite_or_label(float("-inf")) == "-Infinity"
    assert baseline_eval._finite_or_label(1.0) == 1.0


def test_population_and_tactical_validate_container_shapes() -> None:
    with pytest.raises(TypeError, match="tuple"):
        build_population_report(cast(tuple[MatchResult, ...], []))
    with pytest.raises(TypeError, match="MatchResult"):
        build_population_report(cast(tuple[MatchResult, ...], ("bad",)))
    with pytest.raises(ValueError, match="sorted and unique"):
        build_tactical_report(seed=0, depths=(2, 1))
    with pytest.raises(TypeError, match="TacticalCase"):
        baseline_eval._evaluate_minimax_tactical(
            depth=1,
            seed=0,
            cases=cast(tuple[TacticalCase, ...], ("bad",)),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        baseline_eval._evaluate_minimax_tactical(depth=1, seed=0, cases=())


def _valid_final_report_inputs() -> tuple[
    BaselineConfig,
    tuple[MatchResult, ...],
    tuple[BaselineMatchSummary, ...],
    dict[str, object],
]:
    config = load_baseline_config(CONFIG_PATH.read_text(encoding="utf-8"))
    matches = _small_round_robin()
    summaries = tuple(summarize_match(match, elapsed_seconds=0.1) for match in matches)
    tactical = build_tactical_report(seed=config.seed, depths=config.tactical_depths)
    return config, matches, summaries, tactical


def _build_final_with(
    *,
    config: BaselineConfig,
    matches: tuple[MatchResult, ...],
    summaries: tuple[BaselineMatchSummary, ...],
    tactical: dict[str, object],
    git_commit: str = "a" * 40,
) -> dict[str, object]:
    return build_evaluation_report(
        config=config,
        matches=matches,
        summaries=summaries,
        tactical=tactical,
        git_commit=git_commit,
        config_sha256="b" * 64,
        goal_sha256="c" * 64,
        raw_games_sha256="d" * 64,
        hardware={},
        dependencies={},
    )


def test_final_report_rejects_mismatched_evidence() -> None:
    config, matches, summaries, tactical = _valid_final_report_inputs()

    with pytest.raises(TypeError, match="BaselineConfig"):
        _build_final_with(
            config=cast(BaselineConfig, "bad"),
            matches=matches,
            summaries=summaries,
            tactical=tactical,
        )
    with pytest.raises(TypeError, match="summaries"):
        _build_final_with(
            config=config,
            matches=matches,
            summaries=cast(tuple[BaselineMatchSummary, ...], []),
            tactical=tactical,
        )
    with pytest.raises(ValueError, match="one-to-one"):
        _build_final_with(
            config=config,
            matches=matches,
            summaries=summaries[:-1],
            tactical=tactical,
        )
    with pytest.raises(ValueError, match="correspond"):
        _build_final_with(
            config=config,
            matches=matches,
            summaries=(replace(summaries[0], records_sha256="0" * 64), *summaries[1:]),
            tactical=tactical,
        )
    with pytest.raises(TypeError, match="mapping"):
        _build_final_with(
            config=config,
            matches=matches,
            summaries=summaries,
            tactical=cast(dict[str, object], []),
        )
    with pytest.raises(ValueError, match="Git SHA"):
        _build_final_with(
            config=config,
            matches=matches,
            summaries=summaries,
            tactical=tactical,
            git_commit="bad",
        )
    with pytest.raises(TypeError, match="mappings"):
        build_evaluation_report(
            config=config,
            matches=matches,
            summaries=summaries,
            tactical=tactical,
            git_commit="a" * 40,
            config_sha256="b" * 64,
            goal_sha256="c" * 64,
            raw_games_sha256="d" * 64,
            hardware=cast(dict[str, object], []),
            dependencies={},
        )
    with pytest.raises(ValueError, match="complete configured"):
        _build_final_with(
            config=config,
            matches=matches[:-1],
            summaries=summaries[:-1],
            tactical=tactical,
        )
