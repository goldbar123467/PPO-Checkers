"""Browser-native policy export: raw float32 weights, a JSON manifest, and parity fixtures.

The website runs the selected policy entirely in the browser. This module converts a verified
PyTorch inference bundle into a dependency-free little-endian float32 weight file plus a JSON
manifest, and records parity fixtures that pin the TypeScript rules engine, observation encoder,
and network forward pass to this Python implementation.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from checkers.agents.policy_agent import PolicyAgent
from checkers.env.checkers_env import CheckersEnv
from checkers.env.encoding import DEFAULT_MAX_PLIES, encode_observation
from checkers.env.masking import legal_action_map
from checkers.rl.networks import CheckersNetwork
from checkers.rules.state import PlayerId, State
from checkers.web.policy_bundle import LoadedPolicy

BROWSER_POLICY_SCHEMA = "CHECKERS_BROWSER_POLICY_1"
PARITY_FIXTURE_SCHEMA = "CHECKERS_BROWSER_PARITY_1"
MANIFEST_NAME = "policy.json"
WEIGHTS_NAME = "policy.bin"
FLOAT32_BYTES = 4
LOGIT_SIGNIFICANT_DIGITS = 9
NETWORK_CASE_LIMIT = 24
NETWORK_CASE_STRIDE = 37
SPECIAL_CASE_LIMIT = 4

AgentKind = Literal["random", "greedy", "sampled"]
JsonObject = dict[str, object]


class BrowserPolicyError(RuntimeError):
    """Raised when browser weights or their manifest fail validation."""


@dataclass(frozen=True, slots=True)
class FixtureGameSpec:
    """One deterministic game recorded in the parity fixture.

    Args:
        seed: Seed for the random agent and any sampled policy agent.
        red: Agent that plays Red, the first player.
        white: Agent that plays White.
        max_plies: Environment ply cap.
        repetition_draws: Whether threefold repetition ends the game.
        initial_state: Completed-move start position; defaults to the standard opening.
    """

    seed: int
    red: AgentKind
    white: AgentKind
    max_plies: int = DEFAULT_MAX_PLIES
    repetition_draws: bool = True
    initial_state: State | None = None


def default_fixture_games() -> tuple[FixtureGameSpec, ...]:
    """Return the frozen set of games recorded for browser parity."""

    random_games = tuple(
        FixtureGameSpec(seed=seed, red="random", white="random") for seed in range(24)
    )
    kings_endgame = State(
        men=(0, 0),
        kings=((1 << 0) | (1 << 1), (1 << 30) | (1 << 31)),
        side_to_move=PlayerId.RED,
    )
    lone_kings = State(men=(0, 0), kings=(1 << 0, 1 << 31), side_to_move=PlayerId.RED)
    variants = (
        FixtureGameSpec(seed=100, red="random", white="random", max_plies=24),
        FixtureGameSpec(seed=101, red="random", white="random", max_plies=33),
        FixtureGameSpec(seed=102, red="random", white="random", repetition_draws=False),
        FixtureGameSpec(seed=103, red="random", white="random", repetition_draws=False),
        *(
            FixtureGameSpec(
                seed=seed,
                red="random",
                white="random",
                repetition_draws=repetition,
                initial_state=kings_endgame,
            )
            for seed, repetition in ((110, False), (111, False), (112, True), (113, True))
        ),
        FixtureGameSpec(
            seed=125,
            red="random",
            white="random",
            repetition_draws=False,
            initial_state=lone_kings,
        ),
    )
    policy_games = (
        FixtureGameSpec(seed=0, red="greedy", white="greedy"),
        *(FixtureGameSpec(seed=seed, red="greedy", white="random") for seed in range(200, 203)),
        *(FixtureGameSpec(seed=seed, red="random", white="greedy") for seed in range(300, 303)),
        *(FixtureGameSpec(seed=seed, red="sampled", white="sampled") for seed in range(400, 402)),
        FixtureGameSpec(seed=500, red="greedy", white="sampled"),
    )
    return random_games + variants + policy_games


def _ordered_tensors(network: CheckersNetwork) -> list[tuple[str, torch.Tensor]]:
    return [
        (name, tensor.detach().cpu().to(torch.float32).contiguous())
        for name, tensor in network.state_dict().items()
    ]


def encode_browser_weights(network: CheckersNetwork) -> tuple[bytes, list[JsonObject]]:
    """Serialize every network tensor as contiguous little-endian float32 values.

    Args:
        network: Network whose ``state_dict`` order defines the file layout.

    Returns:
        Raw weight bytes and one manifest record per tensor with element offsets.
    """

    if not isinstance(network, CheckersNetwork):
        raise TypeError("network must be a CheckersNetwork")
    chunks: list[bytes] = []
    records: list[JsonObject] = []
    offset = 0
    for name, tensor in _ordered_tensors(network):
        values = tensor.numpy().astype("<f4", copy=False).ravel()
        records.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "offset": offset,
                "length": int(values.size),
            }
        )
        chunks.append(values.tobytes())
        offset += int(values.size)
    return b"".join(chunks), records


def browser_manifest(loaded: LoadedPolicy, weights: bytes, tensors: list[JsonObject]) -> JsonObject:
    """Build the manifest that describes browser weights and their provenance.

    Args:
        loaded: Verified source inference bundle.
        weights: Encoded float32 weight bytes.
        tensors: Tensor layout records from ``encode_browser_weights``.

    Returns:
        A JSON-serializable manifest.
    """

    metadata = loaded.metadata
    return {
        "schema": BROWSER_POLICY_SCHEMA,
        "bundleId": metadata.bundle_id,
        "experimentId": metadata.experiment_id,
        "update": metadata.update_idx,
        "globalStep": metadata.global_step,
        "sourceBundleSha256": loaded.sha256,
        "sourceBundleSizeBytes": loaded.size_bytes,
        "sourceCheckpointSha256": metadata.source_checkpoint_sha256,
        "sourceGitSha": metadata.source_git_sha,
        "configSha256": metadata.config_sha256,
        "maxPlies": metadata.max_plies,
        "repetitionDraws": metadata.repetition_draws,
        "actionCount": metadata.action_count,
        "observationShape": list(metadata.observation_shape),
        "parameterCount": len(weights) // FLOAT32_BYTES,
        "dtype": "float32",
        "byteOrder": "little",
        "weightsFile": WEIGHTS_NAME,
        "weightsSha256": hashlib.sha256(weights).hexdigest(),
        "weightsSizeBytes": len(weights),
        "tensors": tensors,
    }


def write_browser_policy(loaded: LoadedPolicy, output_dir: Path) -> JsonObject:
    """Write ``policy.bin`` and ``policy.json`` for a verified inference bundle.

    Args:
        loaded: Verified source inference bundle.
        output_dir: Directory that receives both files.

    Returns:
        The written manifest.
    """

    if not isinstance(loaded, LoadedPolicy):
        raise TypeError("loaded must be a LoadedPolicy")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    weights, tensors = encode_browser_weights(loaded.network)
    manifest = browser_manifest(loaded, weights, tensors)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / WEIGHTS_NAME).write_bytes(weights)
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise BrowserPolicyError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], value)


def load_browser_policy(directory: Path) -> tuple[CheckersNetwork, Mapping[str, object]]:
    """Strictly load browser weights back into a CPU ``CheckersNetwork``.

    Args:
        directory: Directory containing ``policy.json`` and its weight file.

    Returns:
        The evaluated network and its parsed manifest.

    Raises:
        BrowserPolicyError: If the manifest, digest, layout, or values are invalid.
    """

    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    try:
        manifest = _require_mapping(
            json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8")), "manifest"
        )
        weights = (directory / WEIGHTS_NAME).read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise BrowserPolicyError("browser policy files are unreadable") from error
    if manifest.get("schema") != BROWSER_POLICY_SCHEMA:
        raise BrowserPolicyError("browser policy schema is unsupported")
    if manifest.get("weightsSha256") != hashlib.sha256(weights).hexdigest():
        raise BrowserPolicyError("browser weights digest mismatch")
    values = np.frombuffer(weights, dtype="<f4")
    if not np.isfinite(values).all():
        raise BrowserPolicyError("browser weights contain non-finite values")

    network = CheckersNetwork().cpu()
    expected = network.state_dict()
    records = manifest.get("tensors")
    if not isinstance(records, list) or len(records) != len(expected):
        raise BrowserPolicyError("browser tensor layout does not match CheckersNetwork")
    state: dict[str, torch.Tensor] = {}
    offset = 0
    for raw_record, (name, reference) in zip(records, expected.items(), strict=True):
        record = _require_mapping(raw_record, "tensor record")
        if (
            record.get("name") != name
            or record.get("shape") != list(reference.shape)
            or record.get("offset") != offset
            or record.get("length") != reference.numel()
        ):
            raise BrowserPolicyError(f"browser tensor layout mismatch for {name}")
        chunk = values[offset : offset + reference.numel()]
        state[name] = torch.from_numpy(chunk.copy()).reshape(reference.shape)
        offset += reference.numel()
    if offset != values.size:
        raise BrowserPolicyError("browser weights contain trailing values")
    network.load_state_dict(state, strict=True)
    network.eval()
    return network, manifest


def _state_record(state: State) -> JsonObject:
    return {
        "men": list(state.men),
        "kings": list(state.kings),
        "side": int(state.side_to_move),
        "captureInProgress": state.capture_in_progress,
        "movingSquare": state.moving_square,
        "sequenceOrigin": state.sequence_origin,
        "capturedPending": state.captured_pending,
        "noProgress": list(state.no_progress),
        "ply": state.ply,
    }


def state_from_record(record: Mapping[str, object]) -> State:
    """Rebuild a validated ``State`` from its parity-fixture record."""

    men = cast(list[int], record["men"])
    kings = cast(list[int], record["kings"])
    no_progress = cast(list[int], record["noProgress"])
    return State(
        men=(men[0], men[1]),
        kings=(kings[0], kings[1]),
        side_to_move=PlayerId(cast(int, record["side"])),
        capture_in_progress=cast(bool, record["captureInProgress"]),
        moving_square=cast(int | None, record["movingSquare"]),
        sequence_origin=cast(int | None, record["sequenceOrigin"]),
        captured_pending=cast(int, record["capturedPending"]),
        no_progress=(no_progress[0], no_progress[1]),
        ply=cast(int, record["ply"]),
    )


def observation_record(state: State) -> JsonObject:
    """Return a compact exact form of the default-horizon observation for ``state``.

    Planes 0-5 are binary and stored as flat indices of ones; planes 6 and 7 are constant fills.
    """

    observation = encode_observation(state)
    binary = observation[:6].reshape(-1)
    return {
        "ones": [int(index) for index in np.flatnonzero(binary)],
        "noProgress": float(observation[6, 0, 0]),
        "ply": float(observation[7, 0, 0]),
    }


def _rounded(value: float) -> float:
    return float(f"{value:.{LOGIT_SIGNIFICANT_DIGITS}g}")


def network_case(network: CheckersNetwork, state: State) -> JsonObject:
    """Record a state's observation, logits, value, and masked greedy action."""

    observation = torch.as_tensor(encode_observation(state)).unsqueeze(0)
    with torch.inference_mode():
        output = network(observation)
    logits = output.logits[0]
    legal = list(legal_action_map(state))
    greedy = max(legal, key=lambda action: (float(logits[action]), -action))
    return {
        "state": _state_record(state),
        "observation": observation_record(state),
        "logits": [_rounded(float(value)) for value in logits],
        "value": _rounded(float(output.value[0])),
        "legal": legal,
        "greedy": greedy,
    }


def _agent(
    kind: AgentKind, network: CheckersNetwork, seed: int, rng: random.Random
) -> PolicyAgent | random.Random:
    if kind == "random":
        return rng
    return PolicyAgent(network=network, mode=kind, seed=seed)


def play_fixture_game(
    spec: FixtureGameSpec, network: CheckersNetwork
) -> tuple[JsonObject, list[State]]:
    """Play one fixture game and record every legal-action set and chosen action.

    Args:
        spec: Frozen game description.
        network: Policy network used by greedy or sampled agents.

    Returns:
        The JSON game record and every non-terminal state visited, in order.
    """

    environment = CheckersEnv(
        max_plies=spec.max_plies,
        repetition_draws=spec.repetition_draws,
        initial_state=spec.initial_state,
    )
    environment.reset(seed=spec.seed)
    initial_state = environment.state
    rng = random.Random(spec.seed)
    agents = {
        PlayerId.RED: _agent(spec.red, network, spec.seed, rng),
        PlayerId.WHITE: _agent(spec.white, network, spec.seed + 1, rng),
    }
    actions: list[int] = []
    legal: list[list[int]] = []
    notation: list[str] = []
    states: list[State] = []
    while not environment.terminated:
        state = environment.state
        states.append(state)
        legal_actions = list(legal_action_map(state))
        legal.append(legal_actions)
        agent = agents[state.side_to_move]
        action = (
            agent.choice(legal_actions)
            if isinstance(agent, random.Random)
            else agent.select_action(state)
        )
        _observation, _reward, _terminated, _truncated, info = environment.step(action)
        actions.append(action)
        if info["checkers_move_san"] is not None:
            notation.append(cast(str, info["checkers_move_san"]))
    outcome = environment.outcome
    if outcome is None:  # pragma: no cover - the loop exits only on termination
        raise RuntimeError("fixture game ended without an outcome")
    record: JsonObject = {
        "seed": spec.seed,
        "red": spec.red,
        "white": spec.white,
        "maxPlies": spec.max_plies,
        "repetitionDraws": spec.repetition_draws,
        "initialState": _state_record(initial_state),
        "actions": actions,
        "legal": legal,
        "notation": notation,
        "outcome": {
            "winner": None if outcome.winner is None else outcome.winner.name.lower(),
            "reason": outcome.reason.value,
        },
        "finalState": _state_record(environment.state),
    }
    return record, states


def _network_states(states: Iterable[State]) -> list[State]:
    ordered = list(states)
    selected: dict[str, State] = {}

    def add(state: State) -> None:
        selected.setdefault(json.dumps(_state_record(state), sort_keys=True), state)

    for state in ordered[::NETWORK_CASE_STRIDE][:NETWORK_CASE_LIMIT]:
        add(state)
    for predicate in (
        lambda state: state.capture_in_progress,
        lambda state: bool(state.kings[0] | state.kings[1]),
        lambda state: state.side_to_move is PlayerId.WHITE and state.no_progress[1] > 0,
    ):
        for state in [state for state in ordered if predicate(state)][:SPECIAL_CASE_LIMIT]:
            add(state)
    return list(selected.values())


def build_parity_fixture(
    network: CheckersNetwork,
    manifest: Mapping[str, object],
    specs: Iterable[FixtureGameSpec] | None = None,
) -> JsonObject:
    """Build the complete browser parity fixture for one policy network.

    Args:
        network: Network loaded from browser weights.
        manifest: Browser policy manifest the fixture belongs to.
        specs: Games to record; defaults to ``default_fixture_games()``.

    Returns:
        A JSON-serializable fixture with game replays and network cases.
    """

    if not isinstance(network, CheckersNetwork):
        raise TypeError("network must be a CheckersNetwork")
    network.eval()
    games: list[JsonObject] = []
    visited: list[State] = []
    for spec in default_fixture_games() if specs is None else specs:
        record, states = play_fixture_game(spec, network)
        games.append(record)
        visited.extend(states)
    return {
        "schema": PARITY_FIXTURE_SCHEMA,
        "weightsSha256": manifest["weightsSha256"],
        "games": games,
        "network": [network_case(network, state) for state in _network_states(visited)],
    }
