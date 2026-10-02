"""Local web UI ("live lab") for training and inspecting the RL agents.

A tiny stdlib-only HTTP server (no extra dependencies) that serves a
landing page and exposes a small JSON API:

    GET  /                -> landing page (index.html)
    GET  /api/status      -> info about the agent currently held in memory
    POST /api/train       -> train an agent from JSON hyperparameters
    POST /api/run         -> play episodes greedily with the current agent
    POST /api/delete      -> forget the current agent

Run it with:

    uv run python webapp/server.py

then open http://127.0.0.1:8000 in a browser.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from rl_games import envs

WEB_DIR = Path(__file__).resolve().parent
INDEX = WEB_DIR / "index.html"

HOST = "127.0.0.1"
PORT = 8000

# Hyperparameters each agent accepts (everything else is ignored).
_QLEARNING_KEYS = (
    "n_bins",
    "lr",
    "gamma",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay",
)
_DQN_KEYS = (
    "lr",
    "gamma",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay",
    "batch_size",
    "hidden",
    "target_update_freq",
)
_INT_KEYS = {"n_bins", "batch_size", "hidden", "target_update_freq", "episodes"}

# ── in-memory state (one agent per server process) ────────────────────
_api_lock = threading.RLock()
_state: dict[str, Any] = {"agent": None, "agent_type": None, "env_id": None}


# ── helpers ───────────────────────────────────────────────────────────


def _agent_class(agent_type: str):
    """Import the agent class on demand so torch is only loaded when needed."""
    if agent_type == "qlearning":
        from rl_games.agents.qlearning import QLearningAgent

        return QLearningAgent
    if agent_type == "dqn":
        from rl_games.agents.dqn import DQNAgent

        return DQNAgent
    raise ValueError(f"Unknown agent type: {agent_type!r}")


def _coerce(key: str, value: Any) -> int | float:
    return int(value) if key in _INT_KEYS else float(value)


def _build_agent(agent_type: str, env_id: str, params: dict[str, Any]):
    keys = _QLEARNING_KEYS if agent_type == "qlearning" else _DQN_KEYS
    kwargs = {
        key: _coerce(key, params[key])
        for key in keys
        if params.get(key) not in (None, "")
    }
    return _agent_class(agent_type)(env_id, **kwargs)


def _downsample(values: list[float], n: int) -> list[float]:
    """Reduce a long history to at most `n` points so the chart stays light."""
    if len(values) <= n:
        return [float(v) for v in values]
    step = len(values) / n
    return [float(values[int(i * step)]) for i in range(n)]


# ── API actions ───────────────────────────────────────────────────────


def _train(payload: dict[str, Any]) -> dict[str, Any]:
    with _api_lock:
        agent_type = payload.get("agent_type", "qlearning")
        env_id = payload.get("env_id") or envs.DEFAULT_ENV_ID
        episodes = max(1, int(payload.get("episodes", 2000)))

        agent = _build_agent(agent_type, env_id, payload)
        rewards = agent.train(total_episodes=episodes)

        _state.update(agent=agent, agent_type=agent_type, env_id=env_id)
        return {
            "agent_type": agent_type,
            "env_id": env_id,
            "episodes": len(rewards),
            "rewards": _downsample(rewards, 400),
            "epsilon": float(agent.epsilon),
            "mean_last_100": float(np.mean(rewards[-100:])) if rewards else 0.0,
            "info": agent.info(),
        }


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    with _api_lock:
        agent = _state["agent"]
        env_id = _state["env_id"]
        if agent is None:
            raise ValueError("No hay un agente entrenado todavía. Entrena uno primero.")

        n_episodes = max(1, int(payload.get("n_episodes", 1)))
        env = envs.make(env_id)
        episodes: list[dict[str, Any]] = []
        try:
            for _ in range(n_episodes):
                obs, _ = env.reset()
                states: list[list[float]] = []
                actions: list[int] = []
                rewards: list[float] = []
                done = False
                total = 0.0
                while not done:
                    action, _ = agent.predict(obs, deterministic=True)
                    action = int(action)
                    states.append([float(x) for x in np.asarray(obs).ravel()])
                    actions.append(action)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    rewards.append(float(reward))
                    total += float(reward)
                    done = terminated or truncated
                states.append([float(x) for x in np.asarray(obs).ravel()])
                episodes.append(
                    {
                        "states": states,
                        "actions": actions,
                        "rewards": rewards,
                        "total": total,
                        "steps": len(actions),
                    }
                )
        finally:
            env.close()

        return {"env_id": env_id, "episodes": episodes}


def _delete() -> dict[str, Any]:
    with _api_lock:
        _state.update(agent=None, agent_type=None, env_id=None)
    return {"ok": True}


def _status() -> dict[str, Any]:
    with _api_lock:
        agent = _state["agent"]
        return {
            "env_id": _state["env_id"],
            "agent_type": _state["agent_type"],
            "info": agent.info() if agent is not None else None,
        }


# ── HTTP handler ──────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send_json(200, _status())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        try:
            payload = self._read_json()
            if self.path == "/api/train":
                self._send_json(200, _train(payload))
            elif self.path == "/api/run":
                self._send_json(200, _run(payload))
            elif self.path == "/api/delete":
                self._send_json(200, _delete())
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as exc:  # surface the message to the browser
            self._send_json(500, {"error": str(exc)})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"RL Games live lab escuchando en {url}")
    print("Abre esa URL en el navegador. Pulsa Ctrl+C para detener.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo el servidor...")
        server.shutdown()


if __name__ == "__main__":
    main()