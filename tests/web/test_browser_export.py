"""Browser weight export, strict reload, and committed parity-fixture tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from checkers.agents.policy_agent import PolicyAgent
from checkers.env.checkers_env import CheckersEnv
from checkers.env.masking import legal_action_map
from checkers.rl.networks import CheckersNetwork
from checkers.rules.state import PlayerId, State
from checkers.web.browser_export import (
    BROWSER_POLICY_SCHEMA,
    MANIFEST_NAME,
    PARITY_FIXTURE_SCHEMA,
    WEIGHTS_NAME,
    BrowserPolicyError,
    FixtureGameSpec,
    build_parity_fixture,
    default_fixture_games,
    encode_browser_weights,
    load_browser_policy,
    observation_record,
    state_from_record,
    write_browser_policy,
)
from checkers.web.policy_bundle import LoadedPolicy

WEB_ROOT = Path(__file__).resolve().parents[2] / "web" / "checkers"
MODEL_DIR = WEB_ROOT / "src" / "model"
FIXTURE_PATH = WEB_ROOT / "src" / "test" / "fixtures" / "parity.json"
RELEASE_BUNDLE_SHA256 = "5d6c5c8392f7fb6897a596f5eb204f7d958f6f828d1cf56cfce98b3fcfec34fe"
DEPLOYED_UPDATE = 4608
PARAMETER_COUNT = 470_410
# Float32 convolutions take different kernel paths on different CPUs (oneDNN on or off, SIMD
# width, batch shape), which moves logits of magnitude ~20 by up to ~1e-4. The exact guards are the
# greedy-action checks; these tolerances only bound numeric drift.
LOGIT_ATOL = 1e-4
LOGIT_RTOL = 1e-5
VALUE_ATOL = 1e-4
MAX_PLIES = 512
MIN_GREEDY_DECISIONS = 100
TERMINATION_REASONS = {"no_pieces", "no_legal_move", "no_progress", "repetition", "ply_cap"}


@cache
def _committed_policy() -> tuple[CheckersNetwork, Mapping[str, object]]:
    return load_browser_policy(MODEL_DIR)


@cache
def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _copy_policy(tmp_path: Path) -> Path:
    target = tmp_path / "model"
    target.mkdir(parents=True)
    for name in (MANIFEST_NAME, WEIGHTS_NAME):
        (target / name).write_bytes((MODEL_DIR / name).read_bytes())
    return target


def _rewrite_manifest(directory: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = directory / MANIFEST_NAME
    manifest = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_weights(directory: Path, values: np.ndarray[Any, np.dtype[np.float32]]) -> None:
    payload = values.astype("<f4").tobytes()
    (directory / WEIGHTS_NAME).write_bytes(payload)
    _rewrite_manifest(
        directory,
        lambda manifest: manifest.update(weightsSha256=hashlib.sha256(payload).hexdigest()),
    )


def test_committed_browser_weights_are_the_deployed_release_policy() -> None:
    """The website's weights must trace to the checksummed update-4608 release bundle."""

    network, manifest = _committed_policy()

    assert manifest["schema"] == BROWSER_POLICY_SCHEMA
    assert manifest["update"] == DEPLOYED_UPDATE
    assert manifest["sourceBundleSha256"] == RELEASE_BUNDLE_SHA256
    assert manifest["parameterCount"] == PARAMETER_COUNT
    assert manifest["weightsSizeBytes"] == PARAMETER_COUNT * 4
    assert manifest["maxPlies"] == MAX_PLIES
    assert manifest["repetitionDraws"] is True
    assert sum(parameter.numel() for parameter in network.parameters()) == PARAMETER_COUNT
    assert not network.training


def test_committed_fixture_replays_exactly_in_the_python_engine() -> None:
    """Every recorded legal set, notation, outcome, and final state must be reproducible."""

    fixture = _fixture()
    _network, manifest = _committed_policy()
    assert fixture["schema"] == PARITY_FIXTURE_SCHEMA
    assert fixture["weightsSha256"] == manifest["weightsSha256"]
    assert {game["outcome"]["reason"] for game in fixture["games"]} == TERMINATION_REASONS

    for game in fixture["games"]:
        environment = CheckersEnv(
            max_plies=game["maxPlies"],
            repetition_draws=game["repetitionDraws"],
            initial_state=state_from_record(game["initialState"]),
        )
        environment.reset(seed=game["seed"])
        notation: list[str] = []
        for legal, action in zip(game["legal"], game["actions"], strict=True):
            assert list(legal_action_map(environment.state)) == legal
            _observation, _reward, _terminated, _truncated, info = environment.step(action)
            if info["checkers_move_san"] is not None:
                notation.append(info["checkers_move_san"])
        assert environment.terminated
        assert environment.outcome is not None
        assert notation == game["notation"]
        assert environment.outcome.reason.value == game["outcome"]["reason"]
        winner = environment.outcome.winner
        assert (None if winner is None else winner.name.lower()) == game["outcome"]["winner"]
        assert environment.state == state_from_record(game["finalState"])


def test_committed_fixture_matches_browser_network_outputs() -> None:
    """Recorded observations, logits, values, and greedy moves must match the weights."""

    network, _manifest = _committed_policy()
    fixture = _fixture()
    assert any(case["state"]["captureInProgress"] for case in fixture["network"])
    assert any(case["state"]["side"] == PlayerId.WHITE for case in fixture["network"])

    for case in fixture["network"]:
        state = state_from_record(case["state"])
        assert observation_record(state) == case["observation"]
        observation = torch.as_tensor(observation_record_to_array(case["observation"]))
        with torch.inference_mode():
            output = network(observation.unsqueeze(0))
        torch.testing.assert_close(
            output.logits[0], torch.tensor(case["logits"]), atol=LOGIT_ATOL, rtol=LOGIT_RTOL
        )
        assert float(output.value[0]) == pytest.approx(case["value"], abs=VALUE_ATOL)
        assert list(legal_action_map(state)) == case["legal"]
        greedy = PolicyAgent(network=network, mode="greedy", seed=0).select_action(state)
        assert greedy == case["greedy"]


def observation_record_to_array(record: Mapping[str, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Expand a compact fixture observation into the network's ``(8, 8, 8)`` input."""

    observation = np.zeros((8, 8, 8), dtype=np.float32)
    flat = observation.reshape(-1)
    flat[record["ones"]] = 1.0
    observation[6].fill(record["noProgress"])
    observation[7].fill(record["ply"])
    return observation


def test_committed_greedy_games_are_reproduced_by_browser_weights() -> None:
    """Every greedy fixture decision must equal the Python greedy policy's choice."""

    network, _manifest = _committed_policy()
    agent = PolicyAgent(network=network, mode="greedy", seed=0)
    checked = 0
    for game in _fixture()["games"]:
        greedy_sides = {player for player in PlayerId if game[player.name.lower()] == "greedy"}
        if not greedy_sides:
            continue
        environment = CheckersEnv(
            max_plies=game["maxPlies"],
            repetition_draws=game["repetitionDraws"],
            initial_state=state_from_record(game["initialState"]),
        )
        environment.reset(seed=game["seed"])
        for action in game["actions"]:
            if environment.state.side_to_move in greedy_sides:
                assert agent.select_action(environment.state) == action
                checked += 1
            environment.step(action)
    assert checked > MIN_GREEDY_DECISIONS


def test_export_round_trip_is_exact(tmp_path: Path, loaded_policy: LoadedPolicy) -> None:
    """Written browser weights must reload into identical tensors with a truthful manifest."""

    manifest = write_browser_policy(loaded_policy, tmp_path / "out")
    network, reloaded_manifest = load_browser_policy(tmp_path / "out")

    assert reloaded_manifest == json.loads(json.dumps(manifest))
    assert manifest["sourceBundleSha256"] == loaded_policy.sha256
    assert manifest["update"] == loaded_policy.metadata.update_idx
    for expected, actual in zip(
        loaded_policy.network.state_dict().values(), network.state_dict().values(), strict=True
    ):
        assert torch.equal(expected, actual)


def test_encoded_layout_is_contiguous_state_dict_order() -> None:
    """Tensor records must tile the weight file in ``state_dict`` order without gaps."""

    network = CheckersNetwork()
    weights, records = encode_browser_weights(network)
    offset = 0
    for record, (name, tensor) in zip(records, network.state_dict().items(), strict=True):
        assert record == {
            "name": name,
            "shape": list(tensor.shape),
            "offset": offset,
            "length": tensor.numel(),
        }
        offset += tensor.numel()
    assert len(weights) == offset * 4


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda manifest: manifest.update(schema="OTHER"), "schema"),
        (lambda manifest: manifest.update(weightsSha256="0" * 64), "digest"),
        (lambda manifest: manifest.update(tensors=[]), "layout does not match"),
        (lambda manifest: manifest.update(tensors="bad"), "layout does not match"),
        (
            lambda manifest: manifest["tensors"][0].update(name="renamed"),
            "layout mismatch for stem.0.weight",
        ),
        (
            lambda manifest: manifest["tensors"][1].update(offset=0),
            "layout mismatch for stem.0.bias",
        ),
        (
            lambda manifest: manifest["tensors"].__setitem__(0, "not-an-object"),
            "tensor record must be a JSON object",
        ),
    ],
)
def test_load_rejects_invalid_manifests(
    tmp_path: Path, change: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Manifest schema, digest, and layout errors must fail closed."""

    directory = _copy_policy(tmp_path)
    _rewrite_manifest(directory, change)

    with pytest.raises(BrowserPolicyError, match=message):
        load_browser_policy(directory)


def test_load_rejects_non_finite_and_trailing_weights(tmp_path: Path) -> None:
    """Weight values must be finite and exactly fill the declared layout."""

    values = np.frombuffer((MODEL_DIR / WEIGHTS_NAME).read_bytes(), dtype="<f4").copy()
    non_finite = _copy_policy(tmp_path / "nan")
    poisoned = values.copy()
    poisoned[0] = np.nan
    _rewrite_weights(non_finite, poisoned)
    with pytest.raises(BrowserPolicyError, match="non-finite"):
        load_browser_policy(non_finite)

    trailing = _copy_policy(tmp_path / "trailing")
    _rewrite_weights(trailing, np.concatenate([values, np.zeros(1, dtype=np.float32)]))
    with pytest.raises(BrowserPolicyError, match="trailing"):
        load_browser_policy(trailing)


def test_load_rejects_missing_or_malformed_files(tmp_path: Path) -> None:
    """Unreadable files and non-object manifests are explicit policy errors."""

    with pytest.raises(BrowserPolicyError, match="unreadable"):
        load_browser_policy(tmp_path)
    directory = _copy_policy(tmp_path)
    (directory / MANIFEST_NAME).write_text("[]", encoding="utf-8")
    with pytest.raises(BrowserPolicyError, match="manifest must be a JSON object"):
        load_browser_policy(directory)
    (directory / MANIFEST_NAME).write_text("{", encoding="utf-8")
    with pytest.raises(BrowserPolicyError, match="unreadable"):
        load_browser_policy(directory)


def test_public_functions_validate_argument_types(loaded_policy: LoadedPolicy) -> None:
    """Public export helpers must reject values of the wrong runtime type."""

    with pytest.raises(TypeError, match="network"):
        encode_browser_weights(cast(CheckersNetwork, object()))
    with pytest.raises(TypeError, match="loaded"):
        write_browser_policy(cast(LoadedPolicy, object()), Path("unused"))
    with pytest.raises(TypeError, match="output_dir"):
        write_browser_policy(loaded_policy, cast(Path, "unused"))
    with pytest.raises(TypeError, match="directory"):
        load_browser_policy(cast(Path, "unused"))
    with pytest.raises(TypeError, match="network"):
        build_parity_fixture(cast(CheckersNetwork, object()), {})


def test_parity_fixture_builder_records_replayable_games() -> None:
    """The builder must record each requested game and network cases from visited states."""

    network = CheckersNetwork()
    lone_kings = State(men=(0, 0), kings=(1 << 0, 1 << 31), side_to_move=PlayerId.RED)
    specs = (
        FixtureGameSpec(seed=1, red="greedy", white="sampled", max_plies=12),
        FixtureGameSpec(seed=2, red="random", white="random", initial_state=lone_kings),
    )
    fixture = build_parity_fixture(network, {"weightsSha256": "f" * 64}, specs)

    assert fixture["schema"] == PARITY_FIXTURE_SCHEMA
    assert fixture["weightsSha256"] == "f" * 64
    games = cast(list[dict[str, Any]], fixture["games"])
    assert [game["seed"] for game in games] == [1, 2]
    assert games[0]["outcome"] == {"winner": None, "reason": "ply_cap"}
    assert games[1]["initialState"]["kings"] == [1, 1 << 31]
    assert cast(list[object], fixture["network"])


def test_default_fixture_games_are_frozen() -> None:
    """The recorded game set must keep covering policy agents and rule variants."""

    specs = default_fixture_games()
    assert len(specs) == len(set(specs))
    assert {spec.red for spec in specs} | {spec.white for spec in specs} == {
        "random",
        "greedy",
        "sampled",
    }
    assert any(not spec.repetition_draws for spec in specs)
    assert any(spec.initial_state is not None for spec in specs)
