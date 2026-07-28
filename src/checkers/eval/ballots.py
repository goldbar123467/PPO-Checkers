"""Oracle-enumerated opening evidence and position-deduplicated evaluation ballots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from checkers.env.checkers_env import CheckersEnv
from checkers.env.masking import ACTION_COUNT, step_to_action
from checkers.rules.moves import apply_step, legal_steps
from checkers.rules.notation import serialize_state
from checkers.rules.oracle import oracle_legal_steps
from checkers.rules.state import State
from checkers.rules.zobrist import position_key

SEQUENCE_SCHEMA = "CHECKERS_BALLOT_SEQUENCES_1"
BALLOT_SCHEMA = "CHECKERS_EVAL_BALLOTS_1"
BALLOT_MOVES = 3
SEQUENCE_COUNT = 302
BALLOT_COUNT = 216
EXPECTED_FIRST_MOVES = 7
TRANSPOSITION_EXAMPLE_COUNT = 2
SHA256_HEX_LENGTH = 64


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _records_sha256(records: list[dict[str, object]]) -> str:
    return hashlib.sha256(_canonical_bytes(records)).hexdigest()


def _checked_actions(actions: object) -> tuple[int, ...]:
    if not isinstance(actions, tuple):
        raise TypeError("actions must be a tuple")
    checked: list[int] = []
    for action in actions:
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("each action must be an integer")
        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"action must be in [0, {ACTION_COUNT - 1}]")
        checked.append(action)
    return tuple(checked)


@dataclass(frozen=True, slots=True)
class OpeningReplay:
    """Exact result of replaying one opening action sequence."""

    state: State
    completed_moves: int
    moves: tuple[str, ...]


def replay_opening(actions: tuple[int, ...]) -> OpeningReplay:
    """Replay from the initial position, checking production moves against the oracle."""

    checked_actions = _checked_actions(actions)
    environment = CheckersEnv(initial_state=State.initial())
    completed_moves: list[str] = []
    try:
        environment.reset(seed=0)
        for action in checked_actions:
            if environment.terminated:
                raise ValueError("opening contains actions after a terminal position")
            production = legal_steps(environment.state)
            oracle = oracle_legal_steps(environment.state)
            if production != oracle:
                raise RuntimeError("production/oracle disagreement while replaying an opening")
            _observation, _reward, _terminated, truncated, info = environment.step(action)
            if truncated:
                raise RuntimeError("opening replay unexpectedly truncated")
            notation = info["checkers_move_san"]
            if notation is not None:
                if not isinstance(notation, str):
                    raise RuntimeError("opening replay produced invalid move notation")
                completed_moves.append(notation)
        return OpeningReplay(
            state=environment.state,
            completed_moves=len(completed_moves),
            moves=tuple(completed_moves),
        )
    finally:
        environment.close()


def _sequence_hash(actions: tuple[int, ...], state: State) -> str:
    payload = {"actions": actions, "state": serialize_state(state)}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class OpeningSequence:
    """One distinct legal history ending after three completed checkers moves."""

    sequence_id: str
    actions: tuple[int, ...]
    moves: tuple[str, ...]
    state: State
    position_key: int

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("sequence_id must be non-empty text")
        checked_actions = _checked_actions(self.actions)
        if not isinstance(self.moves, tuple) or any(
            not isinstance(move, str) or not move for move in self.moves
        ):
            raise ValueError("moves must be a tuple of non-empty notation strings")
        if not isinstance(self.state, State):
            raise TypeError("state must be a State")
        replay = replay_opening(checked_actions)
        if replay.completed_moves != BALLOT_MOVES:
            raise ValueError(f"sequence must contain exactly {BALLOT_MOVES} completed moves")
        if replay.moves != self.moves or replay.state != self.state:
            raise ValueError("sequence record disagrees with its legal replay")
        if self.state.capture_in_progress:
            raise ValueError("sequence must end at a completed-move boundary")
        if position_key(self.state) != self.position_key:
            raise ValueError("sequence position_key disagrees with its boundary state")

    @property
    def sequence_hash(self) -> str:
        """Return the identity hash of the ordered action history and resulting state."""

        return _sequence_hash(self.actions, self.state)

    def to_record(self) -> dict[str, object]:
        """Return this sequence as a canonical JSON-compatible record."""

        return {
            "sequence_id": self.sequence_id,
            "sequence_hash": self.sequence_hash,
            "actions": list(self.actions),
            "moves": list(self.moves),
            "state": serialize_state(self.state),
            "position_key": f"{self.position_key:016x}",
        }


@dataclass(frozen=True, slots=True)
class OpeningBallot:
    """One representative sequence for a unique official boundary position."""

    ballot_id: str
    representative_sequence_id: str
    actions: tuple[int, ...]
    moves: tuple[str, ...]
    state: State
    position_key: int

    def __post_init__(self) -> None:
        if not isinstance(self.ballot_id, str) or not self.ballot_id:
            raise ValueError("ballot_id must be non-empty text")
        if (
            not isinstance(self.representative_sequence_id, str)
            or not self.representative_sequence_id
        ):
            raise ValueError("representative_sequence_id must be non-empty text")
        replay = replay_opening(_checked_actions(self.actions))
        if replay.completed_moves != BALLOT_MOVES:
            raise ValueError(f"ballot must contain exactly {BALLOT_MOVES} completed moves")
        if replay.moves != self.moves or replay.state != self.state:
            raise ValueError("ballot record disagrees with its legal replay")
        if self.state.capture_in_progress:
            raise ValueError("ballot must end at a completed-move boundary")
        if position_key(self.state) != self.position_key:
            raise ValueError("ballot position_key disagrees with its boundary state")

    def to_record(self) -> dict[str, object]:
        """Return this ballot as a canonical JSON-compatible record."""

        return {
            "ballot_id": self.ballot_id,
            "representative_sequence_id": self.representative_sequence_id,
            "actions": list(self.actions),
            "moves": list(self.moves),
            "state": serialize_state(self.state),
            "position_key": f"{self.position_key:016x}",
        }


@dataclass(frozen=True, slots=True)
class TranspositionExample:
    """Two different legal histories that collapse to one official position."""

    position_key: int
    first: OpeningSequence
    second: OpeningSequence

    def to_record(self) -> dict[str, object]:
        """Return a human-readable worked transposition example."""

        return {
            "position_key": f"{self.position_key:016x}",
            "sequences": [
                {
                    "sequence_id": sequence.sequence_id,
                    "actions": list(sequence.actions),
                    "moves": list(sequence.moves),
                }
                for sequence in (self.first, self.second)
            ],
        }


def _enumerate_actions() -> tuple[tuple[int, ...], ...]:
    sequences: list[tuple[int, ...]] = []

    def visit(state: State, actions: tuple[int, ...], completed_moves: int) -> None:
        if completed_moves == BALLOT_MOVES:
            if state.capture_in_progress:
                raise RuntimeError("three-move enumeration ended during a capture sequence")
            position_key(state)
            sequences.append(actions)
            return
        oracle = oracle_legal_steps(state)
        production = legal_steps(state)
        if oracle != production:
            raise RuntimeError("production/oracle disagreement during ballot enumeration")
        for step in oracle:
            action = step_to_action(state, step)
            transition = apply_step(state, step)
            visit(
                transition.after,
                (*actions, action),
                completed_moves + int(transition.move_completed),
            )

    visit(State.initial(), (), 0)
    return tuple(sequences)


def enumerate_sequences() -> tuple[OpeningSequence, ...]:
    """Enumerate all legal histories of exactly three completed checkers moves."""

    sequences: list[OpeningSequence] = []
    for index, actions in enumerate(_enumerate_actions()):
        replay = replay_opening(actions)
        sequences.append(
            OpeningSequence(
                sequence_id=f"sequence-{index:03d}",
                actions=actions,
                moves=replay.moves,
                state=replay.state,
                position_key=position_key(replay.state),
            )
        )
    result = tuple(sequences)
    if len(result) != SEQUENCE_COUNT:
        raise RuntimeError(
            f"oracle enumerated {len(result)} sequences; expected versioned count {SEQUENCE_COUNT}"
        )
    if len({sequence.sequence_hash for sequence in result}) != len(result):
        raise RuntimeError("oracle enumeration produced duplicate sequence hashes")
    distinct_first_moves = len({sequence.actions[0] for sequence in result})
    if distinct_first_moves != EXPECTED_FIRST_MOVES:
        raise RuntimeError(
            "first-move gate failed: expected 7; investigate the move generator before ballots"
        )
    return result


def _position_groups(
    sequences: tuple[OpeningSequence, ...],
) -> dict[int, list[OpeningSequence]]:
    groups: dict[int, list[OpeningSequence]] = {}
    signatures: dict[int, tuple[tuple[int, int], tuple[int, int], int]] = {}
    for sequence in sequences:
        key = sequence.position_key
        signature = (sequence.state.men, sequence.state.kings, int(sequence.state.side_to_move))
        if key in signatures and signatures[key] != signature:
            raise RuntimeError("position_key collision joined two different official positions")
        signatures[key] = signature
        groups.setdefault(key, []).append(sequence)
    return groups


def _transposition_examples(
    groups: dict[int, list[OpeningSequence]],
) -> tuple[TranspositionExample, ...]:
    examples = tuple(
        TranspositionExample(position_key=key, first=group[0], second=group[1])
        for key, group in groups.items()
        if len(group) > 1
    )
    if len(examples) < TRANSPOSITION_EXAMPLE_COUNT:
        raise RuntimeError("enumeration did not produce two transposition examples")
    return examples[:TRANSPOSITION_EXAMPLE_COUNT]


@dataclass(frozen=True, slots=True)
class SequenceManifest:
    """All sequence-level enumeration evidence and its canonical digest."""

    sequences: tuple[OpeningSequence, ...]
    sha256: str
    distinct_first_moves: int
    distinct_positions: int
    transposition_examples: tuple[TranspositionExample, ...]

    @property
    def count(self) -> int:
        """Return the number of distinct ordered histories."""

        return len(self.sequences)

    @property
    def completed_moves(self) -> int:
        """Return the completed-checkers-move ballot depth."""

        return BALLOT_MOVES

    def to_record(self) -> dict[str, object]:
        """Return the complete sequence evidence manifest."""

        return {
            "schema": SEQUENCE_SCHEMA,
            "completed_checkers_moves": self.completed_moves,
            "sequence_count": self.count,
            "sha256": self.sha256,
            "sha256_scope": "canonical compact JSON of the sequences array",
            "first_move_gate": {
                "distinct_first_moves": self.distinct_first_moves,
                "expected": EXPECTED_FIRST_MOVES,
                "passed": self.distinct_first_moves == EXPECTED_FIRST_MOVES,
            },
            "transposition_collapse": {
                "sequence_count": self.count,
                "position_count": self.distinct_positions,
                "collapsed_sequences": self.count - self.distinct_positions,
                "deduplicate_on": "position_key",
            },
            "worked_transposition_examples": [
                example.to_record() for example in self.transposition_examples
            ],
            "sequences": [sequence.to_record() for sequence in self.sequences],
        }

    def to_json(self) -> str:
        """Serialize as canonical, reviewable JSON."""

        return json.dumps(self.to_record(), sort_keys=True, indent=2) + "\n"


@dataclass(frozen=True, slots=True)
class BallotSet:
    """The position-key-deduplicated evaluation set and provenance."""

    ballots: tuple[OpeningBallot, ...]
    sha256: str
    source_sequence_count: int
    source_sequences_sha256: str
    distinct_first_moves: int
    transposition_examples: tuple[TranspositionExample, ...]
    deduplicate_on: str = "position_key"

    @property
    def count(self) -> int:
        """Return the number of unique official positions."""

        return len(self.ballots)

    @property
    def completed_moves(self) -> int:
        """Return the completed-checkers-move ballot depth."""

        return BALLOT_MOVES

    def to_record(self) -> dict[str, object]:
        """Return the complete evaluation ballot manifest."""

        return {
            "schema": BALLOT_SCHEMA,
            "completed_checkers_moves": self.completed_moves,
            "ballot_count": self.count,
            "sha256": self.sha256,
            "sha256_scope": "canonical compact JSON of the ballots array",
            "source_sequence_count": self.source_sequence_count,
            "source_sequences_sha256": self.source_sequences_sha256,
            "first_move_gate": {
                "distinct_first_moves": self.distinct_first_moves,
                "expected": EXPECTED_FIRST_MOVES,
                "passed": self.distinct_first_moves == EXPECTED_FIRST_MOVES,
            },
            "transposition_collapse": {
                "sequence_count": self.source_sequence_count,
                "position_count": self.count,
                "collapsed_sequences": self.source_sequence_count - self.count,
                "deduplicate_on": self.deduplicate_on,
            },
            "worked_transposition_examples": [
                example.to_record() for example in self.transposition_examples
            ],
            "ballots": [ballot.to_record() for ballot in self.ballots],
        }

    def to_json(self) -> str:
        """Serialize as canonical, reviewable JSON."""

        return json.dumps(self.to_record(), sort_keys=True, indent=2) + "\n"


def generate_ballot_artifacts() -> tuple[SequenceManifest, BallotSet]:
    """Generate exhaustive sequence evidence and its unique-position evaluation subset."""

    sequences = enumerate_sequences()
    groups = _position_groups(sequences)
    if len(groups) != BALLOT_COUNT:
        raise RuntimeError(
            f"position_key deduplication produced {len(groups)} ballots; expected {BALLOT_COUNT}"
        )
    examples = _transposition_examples(groups)
    sequence_records = [sequence.to_record() for sequence in sequences]
    sequence_sha256 = _records_sha256(sequence_records)
    sequence_manifest = SequenceManifest(
        sequences=sequences,
        sha256=sequence_sha256,
        distinct_first_moves=len({sequence.actions[0] for sequence in sequences}),
        distinct_positions=len(groups),
        transposition_examples=examples,
    )
    ballots = tuple(
        OpeningBallot(
            ballot_id=f"position-{key:016x}",
            representative_sequence_id=group[0].sequence_id,
            actions=group[0].actions,
            moves=group[0].moves,
            state=group[0].state,
            position_key=key,
        )
        for key, group in groups.items()
    )
    if len({ballot.position_key for ballot in ballots}) != len(ballots):
        raise RuntimeError("evaluation ballots contain duplicate position keys")
    ballot_sha256 = _records_sha256([ballot.to_record() for ballot in ballots])
    ballot_set = BallotSet(
        ballots=ballots,
        sha256=ballot_sha256,
        source_sequence_count=len(sequences),
        source_sequences_sha256=sequence_sha256,
        distinct_first_moves=sequence_manifest.distinct_first_moves,
        transposition_examples=examples,
    )
    return sequence_manifest, ballot_set


def _load_canonical(path: Path, expected_record: dict[str, object], expected_json: str) -> None:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        text = path.read_text(encoding="utf-8")
        actual = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid ballot JSON: {path}") from error
    if actual != expected_record:
        raise ValueError(f"ballot artifact disagrees with exhaustive oracle generation: {path}")
    if text != expected_json:
        raise ValueError(f"ballot artifact is not canonical JSON: {path}")


def load_sequence_manifest(path: Path) -> SequenceManifest:
    """Load and independently reproduce the committed exhaustive sequence evidence."""

    sequences, _ballots = generate_ballot_artifacts()
    _load_canonical(path, sequences.to_record(), sequences.to_json())
    return sequences


def load_ballot_set(path: Path) -> BallotSet:
    """Load and independently reproduce the committed unique-position evaluation set."""

    _sequences, ballots = generate_ballot_artifacts()
    _load_canonical(path, ballots.to_record(), ballots.to_json())
    return ballots
