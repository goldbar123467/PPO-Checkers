"""Greedy or privately sampled arena adapter for a learned checkers network."""

from __future__ import annotations

from typing import Literal

import torch

from checkers.agents.base import NoLegalActionError, validate_state
from checkers.env.encoding import encode_observation
from checkers.env.masking import legal_action_mask
from checkers.rl.masked_categorical import MaskedCategorical
from checkers.rl.networks import CheckersNetwork
from checkers.rules.state import State

PolicyMode = Literal["greedy", "sampled"]
UINT64_MAX = (1 << 64) - 1


class PolicyAgent:
    """Select legal actions from one frozen-view shared policy/value network."""

    def __init__(
        self,
        *,
        network: CheckersNetwork,
        mode: PolicyMode,
        seed: int,
        name: str | None = None,
    ) -> None:
        """Create a deterministic greedy or privately seeded categorical policy."""

        if not isinstance(network, CheckersNetwork):
            raise TypeError("network must be a CheckersNetwork")
        if mode not in ("greedy", "sampled"):
            raise ValueError("mode must be greedy or sampled")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= seed <= UINT64_MAX:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("name must be non-empty text or None")
        self._network = network
        self._mode = mode
        self._device = next(network.parameters()).device
        self._generator = torch.Generator(device=self._device).manual_seed(seed)
        self.name = f"policy-{mode}" if name is None else name.strip()

    def select_action(self, state: State) -> int:
        """Return one greedy or temperature-one sampled action under the exact legal mask."""

        checked_state = validate_state(state)
        mask = torch.as_tensor(
            legal_action_mask(checked_state),
            dtype=torch.bool,
            device=self._device,
        ).unsqueeze(0)
        if not bool(mask.any().item()):
            raise NoLegalActionError("state has no legal action")
        observation = torch.as_tensor(
            encode_observation(checked_state),
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)
        was_training = self._network.training
        try:
            self._network.eval()
            with torch.no_grad():
                output = self._network(observation)
                distribution = MaskedCategorical(logits=output.logits, legal_mask=mask)
                if self._mode == "greedy":
                    action = distribution.probs.argmax(dim=-1)
                else:
                    action = torch.multinomial(
                        distribution.probs,
                        num_samples=1,
                        replacement=True,
                        generator=self._generator,
                    ).squeeze(1)
        finally:
            self._network.train(was_training)
        return int(action.item())
