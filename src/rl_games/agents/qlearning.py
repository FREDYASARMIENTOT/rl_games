"""Tabular Q-learning over a discretised observation space.

Contents:
  - QLearningAgent : observation discretisation, epsilon-greedy policy,
                     temporal-difference update, training loop, and save/load

The discretisation, policy and update are exercises; see CHEATSHEET.md.
"""
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Self

import numpy as np

from rl_games import envs
from rl_games.agents.base import BaseAgent


class QLearningAgent(BaseAgent):
    """Tabular Q-learning over a discretised observation space."""

    label = "Q-Learning"
    def __init__(
        self,
        env_id: str,
        *,
        n_bins: int = 10,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
    ) -> None:
        super().__init__(
            env_id,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
        )
        self.n_bins = n_bins

        # Ask the env itself for the action count, and for the bounds unless
        # it reports an unbounded observation space.
        env = envs.make(env_id)
        try:
            self.n_actions = int(env.action_space.n)  # type: ignore[attr-defined]
            self._bounds, self._n_binary_dims = envs.bounds_for(env_id, env)
        finally:
            env.close()

        self._bins = [
            np.linspace(lo, hi, n_bins + 1)[1:-1] for lo, hi in self._bounds
        ]
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def discretize(self, obs: np.ndarray) -> tuple:
        """Map a continuous observation to a hashable tuple of bin indices.

        The result is used directly as a q_table key, so it must be a tuple
        of ints, one per observation dimension.
        """
        obs = np.asarray(obs, dtype=float)
        n_cont = len(self._bounds)

        indices: list[int] = []
        # Leading continuous dims: clip into range, then bin.
        for i in range(n_cont):
            lo, hi = self._bounds[i]
            value = float(np.clip(obs[i], lo, hi))
            indices.append(int(np.digitize(value, self._bins[i])))
        # Trailing 0/1 dims are already discrete: use as bin indices verbatim.
        for i in range(n_cont, n_cont + self._n_binary_dims):
            indices.append(int(round(float(obs[i]))))
        return tuple(indices)

    def select_action(self, state: tuple, *, deterministic: bool = False) -> int:
        """Epsilon-greedy action for an already-discretised `state`."""
        if not deterministic and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def _to_state(self, obs: np.ndarray) -> tuple:
        return self.discretize(obs)

    # ------------------------------------------------------------------
    # core RL
    # ------------------------------------------------------------------

    def _update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        """Apply one temporal-difference update to Q(state, action)."""
        current = self.q_table[state][action]
        # A terminal state has no future reward, so the bootstrap term is 0.
        future = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.gamma * future
        self.q_table[state][action] = current + self.lr * (target - current)

    def train(self, total_episodes: int = 10_000, log_interval: int = 100) -> list[float]:
        env = envs.make(self.env_id)
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            state = self.discretize(obs)
            total_reward = 0.0
            done = False

            # Environment loop
            while not done:
                # Select action
                action = self.select_action(state)
                # Take action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                # Update done flag
                done = terminated or truncated
                # Update state
                next_state = self.discretize(next_obs)
                # Update Q-table
                self._update(state, action, reward, next_state, done)
                # Update state
                state = next_state
                # Update total reward
                total_reward += reward

            self._decay_epsilon()
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % log_interval == 0:
                self._log_episode(
                    episode,
                    total_episodes,
                    rewards_history,
                    log_interval,
                    f"States visited: {len(self.q_table)}",
                )

        env.close()
        return rewards_history

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "env_id": self.env_id,
            "n_bins": self.n_bins,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {self.label} agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        agent = cls(
            env_id=data["env_id"],
            n_bins=data["n_bins"],
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
        )
        agent.q_table = defaultdict(
            lambda: np.zeros(agent.n_actions), data["q_table"]
        )
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        return (
            f"{self.label} agent for {self.env_id}\n"
            f"  Episodes trained : {self.training_episodes}\n"
            f"  States visited   : {len(self.q_table)}\n"
            f"  Epsilon          : {self.epsilon:.4f}\n"
            f"  LR / Gamma       : {self.lr} / {self.gamma}"
        )
