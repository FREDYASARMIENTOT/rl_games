"""Agent implementations.

Classes are resolved lazily (PEP 562 ``__getattr__``) so importing one agent
does not pull in the others' heavy dependencies: Q-Learning needs only NumPy,
while DQN needs torch. ``from rl_games.agents import QLearningAgent`` therefore
works without torch installed.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_games.agents.dqn import DQNAgent
    from rl_games.agents.qlearning import QLearningAgent

__all__ = ["DQNAgent", "QLearningAgent"]

# public name -> module that defines it (imported the first time it is used)
_EXPORTS = {
    "DQNAgent": "rl_games.agents.dqn",
    "QLearningAgent": "rl_games.agents.qlearning",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = getattr(import_module(module_name), name)  # may raise (e.g. no torch)
    globals()[name] = value  # cache it so later lookups are cheap
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
