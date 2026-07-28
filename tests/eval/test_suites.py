"""Exact terminal-only tactical suite and depth-comparison gates."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from checkers.agents.minimax_agent import MinimaxAgent
from checkers.env.masking import legal_action_map, step_to_action
from checkers.eval.suites import (
    DEV_TACTICAL_CASES,
    TacticalAgentError,
    TacticalCase,
    TacticalCaseResult,
    TacticalDepthComparison,
    TacticalEvaluation,
    TacticalSuite,
    TacticalSuiteManifest,
    compare_tactical,
    evaluate_tactical,
    forced_win_actions,
    load_dev_tactical_suite,
    parse_tactical_suite,
    replay_tactical_case,
    tactical_case_record,
    tactical_cases_sha256,
)
from checkers.rules.moves import Step, apply_step
from checkers.rules.state import PlayerId, State

TACTICAL_HORIZON = 3
MINIMUM_SHALLOW_MISSES = 5
EXPECTED_NET_GAIN = 2
TACTICAL_DATA_PATH = Path("src/checkers/eval/data/dev_tactics_v1.json")


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _forced_win_state() -> State:
    return State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _two_jump_win_state() -> State:
    return State(
        men=(_mask(9), _mask(14, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _terminal_state() -> State:
    return State(
        men=(_mask(9), 0),
        kings=(0, 0),
        side_to_move=PlayerId.WHITE,
    )


def _capture_state() -> State:
    state = _two_jump_win_state()
    return apply_step(state, Step(origin=8, destination=17, captured=13)).after


def _json_document() -> dict[str, object]:
    return cast(dict[str, object], json.loads(TACTICAL_DATA_PATH.read_text(encoding="utf-8")))


def _first_case_record(document: dict[str, object]) -> dict[str, object]:
    cases = cast(list[object], document["cases"])
    return cast(dict[str, object], cases[0])


def _state_record(document: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], _first_case_record(document)["state"])


def _parse_document(document: dict[str, object]) -> TacticalSuite:
    return parse_tactical_suite(json.dumps(document, sort_keys=True))


class TableAgent:
    """Select the frozen first exact solution for each suite state."""

    name = "exact-table"

    def __init__(self, cases: tuple[TacticalCase, ...]) -> None:
        self._actions = {case.state: case.winning_actions[0] for case in cases}

    def select_action(self, state: State) -> int:
        return self._actions[state]


class IllegalAgent:
    """Policy test double that violates the action contract."""

    name = "illegal"

    def select_action(self, state: State) -> int:
        del state
        return -1


def test_terminal_only_solver_finds_immediate_forced_capture() -> None:
    state = _forced_win_state()
    expected = step_to_action(state, Step(origin=8, destination=17, captured=13))

    assert forced_win_actions(state, max_completed_moves=1) == (expected,)


def test_terminal_only_solver_counts_multi_jump_as_one_completed_move() -> None:
    state = _two_jump_win_state()
    expected = step_to_action(state, Step(origin=8, destination=17, captured=13))

    assert forced_win_actions(state, max_completed_moves=1) == (expected,)


def test_terminal_only_solver_does_not_substitute_material_at_horizon() -> None:
    state = State.initial()

    assert forced_win_actions(state, max_completed_moves=1) == ()


def test_terminal_only_solver_returns_no_root_action_for_terminal_state() -> None:
    assert forced_win_actions(_terminal_state(), max_completed_moves=1) == ()


def test_dev_suite_manifest_and_case_count_are_exact() -> None:
    suite = load_dev_tactical_suite()

    assert isinstance(suite, TacticalSuite)
    assert suite.manifest.schema_version == 1
    assert suite.manifest.case_count == DEV_TACTICAL_CASES
    assert len(suite.cases) == DEV_TACTICAL_CASES
    assert suite.manifest.horizon_completed_moves == TACTICAL_HORIZON
    assert suite.manifest.split == "dev"
    assert suite.manifest.review_status == "programmatically_verified_pending_human_review"


def test_every_dev_case_replays_exactly_from_standard_opening() -> None:
    suite = load_dev_tactical_suite()

    assert all(replay_tactical_case(case) == case.state for case in suite.cases)


def test_every_frozen_solution_is_recomputed_by_independent_terminal_oracle() -> None:
    suite = load_dev_tactical_suite()

    for case in suite.cases:
        assert (
            forced_win_actions(
                case.state,
                max_completed_moves=case.max_completed_moves,
            )
            == case.winning_actions
        )


def test_dev_cases_are_nontrivial_unique_and_symmetry_deduplicated() -> None:
    suite = load_dev_tactical_suite()

    assert len({case.case_id for case in suite.cases}) == DEV_TACTICAL_CASES
    assert len({case.state for case in suite.cases}) == DEV_TACTICAL_CASES
    assert len({case.duplicate_group for case in suite.cases}) == DEV_TACTICAL_CASES
    assert all(
        0 < len(case.winning_actions) < len(legal_action_map(case.state)) for case in suite.cases
    )


def test_dev_case_metadata_tracks_required_data_discipline_fields() -> None:
    suite = load_dev_tactical_suite()
    manifest = suite.manifest

    assert manifest.provenance
    assert manifest.license
    assert manifest.author
    assert manifest.creation_method
    assert manifest.grade_band == "not_applicable_game_state"
    assert manifest.safety_categories == ("none",)
    assert manifest.subject_categories == ("american_checkers_tactics",)
    assert manifest.difficulty == "forced_win_within_three_completed_moves"
    assert all(case.rationale for case in suite.cases)
    assert all(case.difficulty for case in suite.cases)


def test_exact_table_agent_solves_all_frozen_cases() -> None:
    suite = load_dev_tactical_suite()
    result = evaluate_tactical(TableAgent(suite.cases), suite.cases)

    assert result.solved == DEV_TACTICAL_CASES
    assert result.accuracy == 1.0
    assert result.solved_case_ids == tuple(case.case_id for case in suite.cases)


def test_depth_three_solves_superset_of_depth_one_on_exact_suite() -> None:
    suite = load_dev_tactical_suite()
    shallow = evaluate_tactical(MinimaxAgent(depth=1, seed=20260728), suite.cases)
    deep = evaluate_tactical(MinimaxAgent(depth=3, seed=20260728), suite.cases)
    comparison = compare_tactical(shallow, deep, substantial_gain=MINIMUM_SHALLOW_MISSES)

    assert deep.solved == DEV_TACTICAL_CASES
    assert comparison.deep_is_superset is True
    assert comparison.lost_case_ids == ()
    assert comparison.gained >= MINIMUM_SHALLOW_MISSES
    assert comparison.passes_gate is True


def test_depth_comparison_reports_non_superset_but_substantial_net_gain() -> None:
    case_ids = ("a", "b", "c", "d")
    shallow = TacticalEvaluation.from_case_results(
        agent_name="shallow",
        results=tuple(
            TacticalCaseResult(case_id=case_id, selected_action=0, solved=case_id == "a")
            for case_id in case_ids
        ),
    )
    deep = TacticalEvaluation.from_case_results(
        agent_name="deep",
        results=tuple(
            TacticalCaseResult(case_id=case_id, selected_action=0, solved=case_id != "a")
            for case_id in case_ids
        ),
    )

    comparison = compare_tactical(shallow, deep, substantial_gain=2)

    assert comparison.deep_is_superset is False
    assert comparison.lost_case_ids == ("a",)
    assert comparison.gained_case_ids == ("b", "c", "d")
    assert comparison.net_gain == EXPECTED_NET_GAIN
    assert comparison.substantially_more is True
    assert comparison.passes_gate is True


def test_tactical_evaluator_attributes_illegal_action() -> None:
    suite = load_dev_tactical_suite()

    with pytest.raises(TacticalAgentError, match="illegal.*-1") as caught:
        evaluate_tactical(IllegalAgent(), suite.cases[:1])

    assert caught.value.agent_name == "illegal"
    assert caught.value.case_id == suite.cases[0].case_id
    assert caught.value.action == -1


def test_manifest_record_rejects_runtime_corruption() -> None:
    manifest = load_dev_tactical_suite().manifest

    with pytest.raises(ValueError, match="schema_version"):
        replace(manifest, schema_version=2)
    with pytest.raises(TypeError, match="name"):
        replace(manifest, name=cast(str, 1))
    with pytest.raises(TypeError, match="generator_seed"):
        replace(manifest, generator_seed=cast(int, True))
    with pytest.raises(ValueError, match="generator_seed"):
        replace(manifest, generator_seed=-1)
    with pytest.raises(ValueError, match="depth1_misses"):
        replace(manifest, depth1_misses=DEV_TACTICAL_CASES + 1)
    with pytest.raises(ValueError, match="depth3_solved"):
        replace(manifest, depth3_solved=DEV_TACTICAL_CASES - 1)
    with pytest.raises(TypeError, match="safety_categories"):
        replace(
            manifest,
            safety_categories=cast(tuple[str, ...], ["none"]),
        )
    with pytest.raises(ValueError, match="safety_categories"):
        replace(manifest, safety_categories=())
    with pytest.raises(ValueError, match="unique"):
        replace(manifest, safety_categories=("none", "none"))
    with pytest.raises(TypeError, match="cases_sha256"):
        replace(manifest, cases_sha256=cast(str, 1))


def test_tactical_case_record_rejects_runtime_corruption() -> None:
    case = load_dev_tactical_suite().cases[0]
    legal_actions = tuple(legal_action_map(case.state))

    with pytest.raises(TypeError, match="state"):
        replace(case, state=cast(State, "state"))
    with pytest.raises(ValueError, match="boundary"):
        replace(case, state=_capture_state())
    with pytest.raises(ValueError, match="nonterminal"):
        replace(case, state=_terminal_state())
    with pytest.raises(TypeError, match="winning_actions"):
        replace(case, winning_actions=cast(tuple[int, ...], [case.winning_actions[0]]))
    with pytest.raises(TypeError, match="winning_actions"):
        replace(case, winning_actions=(cast(int, True),))
    with pytest.raises(ValueError, match="winning_actions"):
        replace(case, winning_actions=(128,))
    with pytest.raises(ValueError, match="strict subset"):
        replace(case, winning_actions=legal_actions)
    with pytest.raises(TypeError, match="replay_actions"):
        replace(case, replay_actions=cast(tuple[int, ...], [0]))
    with pytest.raises(TypeError, match="source_game"):
        replace(case, source_game=cast(int, True))
    with pytest.raises(ValueError, match="source_game"):
        replace(case, source_game=-1)
    with pytest.raises(ValueError, match="source_step"):
        replace(case, source_step=0)
    with pytest.raises(TypeError, match="TacticalCase"):
        tactical_case_record(cast(TacticalCase, "case"))
    with pytest.raises(TypeError, match="cases"):
        tactical_cases_sha256(cast(tuple[TacticalCase, ...], [case]))
    with pytest.raises(TypeError, match="TacticalCase"):
        replay_tactical_case(cast(TacticalCase, "case"))


def test_tactical_json_parser_rejects_field_type_corruption() -> None:
    with pytest.raises(TypeError, match="text"):
        parse_tactical_suite(cast(str, 1))
    with pytest.raises(ValueError, match="missing"):
        parse_tactical_suite("{}")
    with pytest.raises(TypeError, match="manifest"):
        parse_tactical_suite('{"manifest": [], "cases": []}')

    document = _json_document()
    document["cases"] = {}
    with pytest.raises(TypeError, match="cases"):
        _parse_document(document)

    document = _json_document()
    del _first_case_record(document)["case_id"]
    with pytest.raises(ValueError, match="missing"):
        _parse_document(document)

    document = _json_document()
    _state_record(document)["men"] = [1]
    with pytest.raises(TypeError, match="men"):
        _parse_document(document)

    document = _json_document()
    _state_record(document)["side_to_move"] = "RED"
    with pytest.raises(TypeError, match="side_to_move"):
        _parse_document(document)

    document = _json_document()
    _state_record(document)["side_to_move"] = 2
    with pytest.raises(ValueError, match="0 or 1"):
        _parse_document(document)

    document = _json_document()
    _state_record(document)["capture_in_progress"] = 0
    with pytest.raises(TypeError, match="capture_in_progress"):
        _parse_document(document)

    document = _json_document()
    _state_record(document)["moving_square"] = "square"
    with pytest.raises(TypeError, match="moving_square"):
        _parse_document(document)

    document = _json_document()
    _state_record(document)["ply"] = "late"
    with pytest.raises(TypeError, match="ply"):
        _parse_document(document)


def test_tactical_suite_rejects_cross_record_corruption() -> None:
    suite = load_dev_tactical_suite()
    first = suite.cases[0]
    second = suite.cases[1]

    with pytest.raises(TypeError, match="manifest"):
        replace(suite, manifest=cast(TacticalSuiteManifest, "manifest"))
    with pytest.raises(TypeError, match="cases"):
        replace(suite, cases=cast(tuple[TacticalCase, ...], list(suite.cases)))
    with pytest.raises(ValueError, match="case_count"):
        replace(suite, cases=suite.cases[:-1])

    horizon_cases = (replace(first, max_completed_moves=2), *suite.cases[1:])
    with pytest.raises(ValueError, match="horizon"):
        replace(suite, cases=horizon_cases)

    duplicate_id_cases = (
        first,
        replace(second, case_id=first.case_id),
        *suite.cases[2:],
    )
    with pytest.raises(ValueError, match="IDs"):
        replace(suite, cases=duplicate_id_cases)

    duplicate_state_cases = (
        first,
        replace(
            first,
            case_id=second.case_id,
            duplicate_group=second.duplicate_group,
        ),
        *suite.cases[2:],
    )
    with pytest.raises(ValueError, match="states"):
        replace(suite, cases=duplicate_state_cases)

    duplicate_group_cases = (
        first,
        replace(second, duplicate_group=first.duplicate_group),
        *suite.cases[2:],
    )
    with pytest.raises(ValueError, match="duplicate groups"):
        replace(suite, cases=duplicate_group_cases)

    with pytest.raises(ValueError, match="digest"):
        replace(
            suite,
            manifest=replace(suite.manifest, cases_sha256="0" * 64),
        )

    replay_case = replace(first, replay_actions=first.replay_actions[:-1])
    replay_cases = (replay_case, *suite.cases[1:])
    replay_manifest = replace(
        suite.manifest,
        cases_sha256=tactical_cases_sha256(replay_cases),
    )
    with pytest.raises(ValueError, match="replay"):
        TacticalSuite(manifest=replay_manifest, cases=replay_cases)

    nonwinning = next(
        action for action in legal_action_map(first.state) if action not in first.winning_actions
    )
    oracle_case = replace(first, winning_actions=(nonwinning,))
    oracle_cases = (oracle_case, *suite.cases[1:])
    oracle_manifest = replace(
        suite.manifest,
        cases_sha256=tactical_cases_sha256(oracle_cases),
    )
    with pytest.raises(ValueError, match="oracle"):
        TacticalSuite(manifest=oracle_manifest, cases=oracle_cases)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: forced_win_actions(cast(State, "state"), max_completed_moves=1), "State"),
        (lambda: forced_win_actions(State.initial(), max_completed_moves=0), "max_completed_moves"),
        (
            lambda: forced_win_actions(
                State.initial(),
                max_completed_moves=cast(int, True),
            ),
            "max_completed_moves",
        ),
        (lambda: parse_tactical_suite("not-json"), "JSON"),
        (lambda: parse_tactical_suite("[]"), "root"),
        (
            lambda: evaluate_tactical(cast(object, "agent"), ()),  # type: ignore[arg-type]
            "Agent",
        ),
        (lambda: evaluate_tactical(TableAgent(()), cast(tuple[TacticalCase, ...], [])), "cases"),
        (
            lambda: compare_tactical(
                TacticalEvaluation.from_case_results(
                    agent_name="same",
                    results=(TacticalCaseResult("a", 0, True),),
                ),
                TacticalEvaluation.from_case_results(
                    agent_name="same",
                    results=(TacticalCaseResult("a", 0, True),),
                ),
                substantial_gain=1,
            ),
            "distinct",
        ),
        (
            lambda: compare_tactical(
                TacticalEvaluation.from_case_results(
                    agent_name="first",
                    results=(TacticalCaseResult("a", 0, True),),
                ),
                TacticalEvaluation.from_case_results(
                    agent_name="second",
                    results=(TacticalCaseResult("b", 0, True),),
                ),
                substantial_gain=1,
            ),
            "case IDs",
        ),
    ],
)
def test_tactical_functions_reject_invalid_inputs(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_tactical_public_records_reject_corruption() -> None:
    suite = load_dev_tactical_suite()
    case = suite.cases[0]
    evaluation = evaluate_tactical(TableAgent(suite.cases), suite.cases[:2])
    comparison = compare_tactical(
        evaluation,
        TacticalEvaluation.from_case_results(
            agent_name="other",
            results=evaluation.results,
        ),
        substantial_gain=1,
    )

    with pytest.raises(ValueError, match="case_id"):
        replace(case, case_id="")
    with pytest.raises(ValueError, match="winning_actions"):
        replace(case, winning_actions=())
    with pytest.raises(ValueError, match="replay"):
        replace(case, replay_actions=())
    with pytest.raises(ValueError, match="digest"):
        replace(suite.manifest, cases_sha256="bad")
    with pytest.raises(ValueError, match="case_count"):
        replace(suite.manifest, case_count=DEV_TACTICAL_CASES - 1)
    corrupt_manifest = replace(suite.manifest)
    object.__setattr__(corrupt_manifest, "case_count", 1)
    with pytest.raises(ValueError, match="manifest"):
        replace(suite, manifest=corrupt_manifest)
    with pytest.raises(ValueError, match="solved"):
        replace(evaluation, solved=0)
    with pytest.raises(ValueError, match="comparison"):
        replace(comparison, gained_case_ids=("not-a-case",))


def test_tactical_evaluation_records_reject_corruption() -> None:
    solved = TacticalCaseResult("a", 0, True)

    with pytest.raises(TypeError, match="solved"):
        replace(solved, solved=cast(bool, 1))
    with pytest.raises(TypeError, match="results"):
        TacticalEvaluation(
            agent_name="agent",
            results=cast(tuple[TacticalCaseResult, ...], [solved]),
            solved=1,
        )
    with pytest.raises(ValueError, match="empty"):
        TacticalEvaluation(agent_name="agent", results=(), solved=0)
    with pytest.raises(ValueError, match="unique"):
        TacticalEvaluation(agent_name="agent", results=(solved, solved), solved=2)
    with pytest.raises(TypeError, match="results"):
        TacticalEvaluation.from_case_results(
            agent_name="agent",
            results=cast(tuple[TacticalCaseResult, ...], [solved]),
        )
    with pytest.raises(ValueError, match="cases"):
        evaluate_tactical(TableAgent(()), ())


def test_tactical_comparison_record_rejects_bookkeeping_corruption() -> None:
    shallow = TacticalEvaluation.from_case_results(
        agent_name="shallow",
        results=(
            TacticalCaseResult("a", 0, True),
            TacticalCaseResult("b", 0, False),
        ),
    )
    deep = TacticalEvaluation.from_case_results(
        agent_name="deep",
        results=(
            TacticalCaseResult("a", 0, False),
            TacticalCaseResult("b", 0, True),
        ),
    )
    comparison = compare_tactical(shallow, deep, substantial_gain=1)

    with pytest.raises(ValueError, match="distinct"):
        replace(comparison, deep_agent="shallow")
    with pytest.raises(TypeError, match="case_ids"):
        replace(comparison, case_ids=cast(tuple[str, ...], ["a", "b"]))
    with pytest.raises(ValueError, match="case_ids"):
        replace(comparison, case_ids=())
    with pytest.raises(ValueError, match="unique"):
        replace(comparison, case_ids=("a", "a"))
    with pytest.raises(ValueError, match="disjoint"):
        replace(comparison, gained_case_ids=("a", "b"))
    with pytest.raises(ValueError, match="substantial_gain"):
        replace(comparison, substantial_gain=0)
    with pytest.raises(TypeError, match="deep_is_superset"):
        replace(comparison, deep_is_superset=cast(bool, 1))
    with pytest.raises(ValueError, match="deep_is_superset"):
        replace(comparison, deep_is_superset=True)
    with pytest.raises(TypeError, match="net_gain"):
        replace(comparison, net_gain=cast(int, True))
    with pytest.raises(ValueError, match="net_gain"):
        replace(comparison, net_gain=1)
    with pytest.raises(TypeError, match="substantially_more"):
        replace(comparison, substantially_more=cast(bool, 1))
    with pytest.raises(ValueError, match="substantially_more"):
        replace(comparison, substantially_more=True)
    with pytest.raises(TypeError, match="passes_gate"):
        replace(comparison, passes_gate=cast(bool, 1))
    with pytest.raises(ValueError, match="passes_gate"):
        replace(comparison, passes_gate=True)

    with pytest.raises(TypeError, match="shallow"):
        compare_tactical(cast(TacticalEvaluation, "shallow"), deep, substantial_gain=1)
    with pytest.raises(TypeError, match="deep"):
        compare_tactical(shallow, cast(TacticalEvaluation, "deep"), substantial_gain=1)


def test_tactical_comparison_public_type_is_immutable() -> None:
    evaluation = TacticalEvaluation.from_case_results(
        agent_name="first",
        results=(TacticalCaseResult("a", 0, True),),
    )
    other = TacticalEvaluation.from_case_results(
        agent_name="second",
        results=(TacticalCaseResult("a", 0, True),),
    )
    comparison: TacticalDepthComparison = compare_tactical(
        evaluation,
        other,
        substantial_gain=1,
    )

    with pytest.raises((AttributeError, TypeError)):
        comparison.net_gain = 1  # type: ignore[misc]


def test_manifest_public_type_is_explicit() -> None:
    manifest: TacticalSuiteManifest = load_dev_tactical_suite().manifest
    assert manifest.name == "checkers_dev_tactics_v1"
