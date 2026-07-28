"""Exact depth-limited minimax over completed checkers moves."""

from __future__ import annotations

import random
from dataclasses import dataclass

from checkers.agents.base import NoLegalActionError, validate_seed, validate_state
from checkers.agents.greedy_agent import material_score
from checkers.env.encoding import DEFAULT_MAX_PLIES
from checkers.env.masking import legal_action_map
from checkers.rules.moves import apply_step
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import terminal_outcome

TERMINAL_SCORE = 10_000


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Immutable diagnostics from one top-level minimax decision."""

    nodes: int
    cache_hits: int
    completed_move_edges: int
    continuation_edges: int
    max_environment_step_depth: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.nodes, "nodes")
        _nonnegative_integer(self.cache_hits, "cache_hits")
        _nonnegative_integer(self.completed_move_edges, "completed_move_edges")
        _nonnegative_integer(self.continuation_edges, "continuation_edges")
        _nonnegative_integer(
            self.max_environment_step_depth,
            "max_environment_step_depth",
        )
        if self.cache_hits > self.nodes:
            raise ValueError("cache_hits cannot exceed nodes")


@dataclass(slots=True)
class _MutableStats:
    nodes: int = 0
    cache_hits: int = 0
    completed_move_edges: int = 0
    continuation_edges: int = 0
    max_environment_step_depth: int = 0

    def freeze(self) -> SearchStats:
        return SearchStats(
            nodes=self.nodes,
            cache_hits=self.cache_hits,
            completed_move_edges=self.completed_move_edges,
            continuation_edges=self.continuation_edges,
            max_environment_step_depth=self.max_environment_step_depth,
        )


@dataclass(slots=True)
class _SearchContext:
    root: PlayerId
    cache: dict[tuple[State, int], int]
    stats: _MutableStats


def _positive_integer(value: object, name: str) -> int:
    checked = _nonnegative_integer(value, name)
    if checked < 1:
        raise ValueError(f"{name} must be positive")
    return checked


class MinimaxAgent:
    """Depth-limited minimax with material leaves and exact sequence handling."""

    def __init__(
        self,
        *,
        depth: int,
        seed: int | None = None,
        max_plies: int = DEFAULT_MAX_PLIES,
    ) -> None:
        """Configure search depth in completed moves, not environment steps.

        Args:
            depth: Positive number of completed checkers moves to search.
            seed: Optional deterministic tie-break seed.
            max_plies: R6.5 terminal boundary used inside the search tree.

        Raises:
            TypeError: If numeric values have invalid runtime types.
            ValueError: If depth or max plies is not positive.
        """

        self.depth = _positive_integer(depth, "depth")
        self.max_plies = _positive_integer(max_plies, "max_plies")
        self.name = f"minimax({self.depth})"
        self._rng = random.Random(validate_seed(seed))
        self.last_stats = SearchStats(0, 0, 0, 0, 0)

    def _terminal_value(self, state: State, root: PlayerId) -> int | None:
        outcome = terminal_outcome(state, max_plies=self.max_plies)
        return None if outcome is None else TERMINAL_SCORE * outcome.score_for(root)

    def _search(
        self,
        state: State,
        remaining_moves: int,
        environment_step_depth: int,
        context: _SearchContext,
    ) -> int:
        stats = context.stats
        stats.nodes += 1
        stats.max_environment_step_depth = max(
            stats.max_environment_step_depth,
            environment_step_depth,
        )
        cache_key = (state, remaining_moves)
        if cache_key in context.cache:
            stats.cache_hits += 1
            return context.cache[cache_key]
        terminal_value = self._terminal_value(state, context.root)
        if terminal_value is not None:
            context.cache[cache_key] = terminal_value
            return terminal_value
        if remaining_moves == 0:
            value = material_score(state, context.root)
            context.cache[cache_key] = value
            return value

        child_values: list[int] = []
        for step in legal_action_map(state).values():
            transition = apply_step(state, step)
            if transition.move_completed:
                stats.completed_move_edges += 1
            else:
                stats.continuation_edges += 1
            child_values.append(
                self._search(
                    transition.after,
                    remaining_moves - int(transition.move_completed),
                    environment_step_depth + 1,
                    context,
                )
            )
        value = max(child_values) if state.side_to_move is context.root else min(child_values)
        context.cache[cache_key] = value
        return value

    def select_action(self, state: State) -> int:
        """Choose a minimax-optimal legal action with seeded tie-breaking.

        Args:
            state: Nonterminal complete rules state.

        Returns:
            One legal canonical action ID attaining the depth-limited minimax value.

        Raises:
            TypeError: If ``state`` is not a ``State``.
            NoLegalActionError: If no legal action exists.
        """

        checked_state = validate_state(state)
        action_map = legal_action_map(checked_state)
        if not action_map:
            raise NoLegalActionError("state has no legal action")
        root = checked_state.side_to_move
        stats = _MutableStats()
        context = _SearchContext(root=root, cache={}, stats=stats)
        action_values: list[tuple[int, int]] = []
        for action, step in action_map.items():
            transition = apply_step(checked_state, step)
            if transition.move_completed:
                stats.completed_move_edges += 1
            else:
                stats.continuation_edges += 1
            value = self._search(
                transition.after,
                self.depth - int(transition.move_completed),
                1,
                context,
            )
            action_values.append((value, action))
        best_value = max(value for value, _action in action_values)
        best_actions = tuple(action for value, action in action_values if value == best_value)
        self.last_stats = stats.freeze()
        return self._rng.choice(best_actions)
