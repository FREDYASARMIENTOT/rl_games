![CI](https://github.com/emiliomunozai/rl_games/actions/workflows/ci.yml/badge.svg?branch=main)

A hands-on repo for understanding how Reinforcement Learning works.
Train, inspect, and visualise RL agents on [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) (or any other Gymnasium environment).

**The DQN learning algorithm is an exercise.** The scaffolding around it -- CLI,
persistence, environment handling, evaluation, a working **tabular Q-Learning**
agent and a local **web UI** -- is complete and working. The network and the
gradient step of DQN raise `NotImplementedError` until you write them.
See [Exercises](#exercises) below.

## LunarLander-v3 environment

The default environment is [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
The lander starts at the top of the screen and the goal is to land it softly on the landing pad (between the two flags) using thrust and rotation.

### State (observation) — 8 continuous values

| Index | Variable | Description | Range |
|:---:|---|---|---|
| 0 | x | Horizontal position | -2.5 to 2.5 |
| 1 | y | Vertical position | -2.5 to 2.5 |
| 2 | vx | Horizontal velocity | -10 to 10 |
| 3 | vy | Vertical velocity | -10 to 10 |
| 4 | angle | Angle of the lander (radians) | -6.28 to 6.28 |
| 5 | angular velocity | Rotation speed | -10 to 10 |
| 6 | left leg contact | 1 if left leg touches ground, 0 otherwise | 0 or 1 |
| 7 | right leg contact | 1 if right leg touches ground, 0 otherwise | 0 or 1 |

### Actions — 4 discrete

| Value | Action |
|:---:|---|
| 0 | Do nothing |
| 1 | Fire left orientation engine (rotate right) |
| 2 | Fire main engine (thrust up) |
| 3 | Fire right orientation engine (rotate left) |

### Rewards

| Event | Reward |
|---|---|
| Moving towards the landing pad | positive, proportional to distance reduction |
| Moving away from the landing pad | negative |
| Crash | **-100** |
| Successful landing (come to rest) | **+100** |
| Each leg ground contact | **+10** |
| Firing main engine (per frame) | **-0.3** |
| Firing side engine (per frame) | **-0.03** |

An episode is considered **solved** at **+200** points average over 100 episodes.
The episode ends when the lander crashes, lands, or after **1000 time steps** (truncation).

> Run `rlgames inspect` to see live state/action/reward values from the environment.

## Quick concepts — Q-Learning methods

### The core idea

An RL agent interacts with an **environment** in discrete time steps.
At each step it observes a **state** $s$, picks an **action** $a$, receives a **reward** $r$, and transitions to a new state $s'$.
The goal is to learn a **policy** $\pi(s) \to a$ that maximises the total (discounted) reward over time.

### Q-values and the Bellman equation

A **Q-value** $Q(s, a)$ estimates the expected cumulative reward of taking action $a$ in state $s$ and then following the optimal policy.
The optimal Q-values satisfy the **Bellman optimality equation**:

$$
Q^*(s, a) = r + \gamma \max_{a'} Q^*(s', a')
$$

where $\gamma \in [0, 1]$ is the **discount factor** (how much the agent cares about future vs. immediate rewards).

### Tabular Q-Learning

When the state and action spaces are small (or can be discretized), we store Q-values in a table and update them after every transition:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \bigr]
$$

- $\alpha$ (learning rate) — how fast we update.
- **$\varepsilon$-greedy** exploration — with probability $\varepsilon$ pick a random action, otherwise pick $\arg\max_a Q(s, a)$. $\varepsilon$ decays over time so the agent gradually shifts from exploring to exploiting.

> `src/rl_games/agents/qlearning.py` holds the tabular agent. The discretisation, policy and TD update are exercises.

### Deep Q-Network (DQN)

When the state space is continuous (like the 8-dimensional LunarLander observation), a table no longer works.
**DQN** replaces the table with a neural network $Q_\theta(s, a)$ and introduces two key tricks:

| Trick | Why |
|---|---|
| **Experience replay** | Store transitions in a buffer, sample random mini-batches — breaks correlation between consecutive samples and reuses data. |
| **Target network** | Keep a frozen copy of the Q-network and update it periodically — stabilises the moving Bellman target. |

Training step (one gradient update):

1. Sample a mini-batch $\{(s, a, r, s', \text{done})\}$ from the replay buffer.
2. Compute targets: $y = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - \text{done})$.
3. Minimise MSE between $Q_\theta(s, a)$ and $y$.

> `src/rl_games/agents/dqn.py` holds the from-scratch PyTorch agent. The replay buffer and training loop are complete; the network and the gradient step are exercises.

### Exploration vs. Exploitation

This is the fundamental trade-off in RL.
**Explore** (random actions) to discover new, potentially better strategies.
**Exploit** (greedy actions) to collect the highest reward based on current knowledge.
The $\varepsilon$-greedy schedule balances both: start with high $\varepsilon$ (mostly exploring) and anneal towards low $\varepsilon$ (mostly exploiting).

## Agents

| Agent | Algorithm | State representation | File |
|---|---|---|---|
| `qlearning` | Tabular Q-Learning | Discretized (`n_bins=10` per continuous dim) | `agents/qlearning.py` |
| `dqn` | DQN from scratch (PyTorch) | Raw continuous | `agents/dqn.py` |

Both share `agents/base.py`, which holds the hyperparameters, the
epsilon-greedy schedule and `predict()`. Each implements its own `train()`.

## Setup

```bash
uv sync
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

## CLI usage

```bash
rlgames <command> [agent] [options]
```

### Version

```bash
rlgames version
```

### List agents and their save status

```bash
rlgames list
```

### Inspect an environment

Show state/action spaces and sample a few random transitions to see what the agent observes.

```bash
rlgames inspect                          # LunarLander-v3 (default)
rlgames inspect --steps 10               # more sample transitions
rlgames inspect --env CartPole-v1        # any Gymnasium env
```

### Initialize a new untrained agent

```bash
rlgames init qlearning
rlgames init dqn
```

### Train an agent

Creates a save if none exists, resumes from an existing save otherwise.

```bash
rlgames train qlearning --episodes 20000
rlgames train dqn       --episodes 500
```

### Load a save and display info

```bash
rlgames load qlearning
rlgames load dqn --eval
```

### Simulate episodes (text output)

Run a trained agent and see every action, reward, and outcome in the terminal.

```bash
rlgames sim qlearning --episodes 3              # full episodes
rlgames sim dqn       --episodes 2 --verbose    # full episodes with state vectors
rlgames sim dqn       --episodes 5 --steps 10   # only first 10 steps per episode
```

### Render episodes (graphical window)

```bash
rlgames render qlearning --episodes 3
rlgames render dqn       --episodes 3
```

### Delete a saved agent

```bash
rlgames delete qlearning
rlgames delete dqn
```

## Local web app (live lab)

A small, **dependency-free** web UI to train and watch the agents in a browser.
It is handy for demos and for eyeballing how hyperparameters change behaviour
without touching the command line.

```bash
uv run python webapp/server.py
# then open http://127.0.0.1:8000
```

### What you can do

- **Set the hyperparameters** in a form -- agent (`qlearning` / `dqn`),
  environment, number of episodes, `n_bins`, `lr`, `gamma`,
  `epsilon_start` / `epsilon_end` / `epsilon_decay` -- and press **Entrenar**.
- **See the learning curve** drawn live on a canvas: the per-episode reward
  plus the 100-episode moving average, together with the final mean and ε.
- **Watch the agent behave**: **Simular** runs greedy episodes on the server
  and returns the full trajectory, which the page then animates step by step.
  Cart-pole is drawn from the real states (cart position and pole angle); any
  other environment falls back to per-dimension bars. Use **▶ Reproducir** and
  the speed slider to control playback.
- **Read the log panel** for the training summary and per-episode returns.
- **Limpiar** forgets the current agent and lets you start over.

### How it works

It is a single-file, **standard-library-only** HTTP server (`http.server`) plus
one HTML page -- no Flask/FastAPI, no build step, no new dependencies.

| Method | Path | Purpose |
|:---:|---|---|
| `GET` | `/` | serve the landing page (`index.html`) |
| `GET` | `/api/status` | info about the agent currently held in memory |
| `POST` | `/api/train` | build an agent from JSON hyperparameters and train it |
| `POST` | `/api/run` | play N greedy episodes and return the full trajectories |
| `POST` | `/api/delete` | forget the current agent |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"agent_type":"qlearning","env_id":"CartPole-v1","episodes":2000,"n_bins":12,"epsilon_decay":0.99}'
```

Notes:

- The trained agent lives **in memory** (one per server process): stopping the
  server loses it, and it never touches `saves/`.
- `qlearning` needs only NumPy + Gymnasium; `dqn` additionally needs torch
  (installed by `uv sync`). The agent classes are imported lazily, so merely
  opening the page never imports torch.
- A good first demo is `CartPole-v1` with `n_bins=12`, `epsilon_decay=0.99` and
  a few thousand episodes. The default `epsilon_decay=0.9995` keeps ε high for
  a long time, so the agent appears to learn slowly in a short run.

### Running without torch (Q-Learning only)

`uv sync` downloads torch (~108 MiB), which can be slow on a poor connection.
If you only care about the tabular agent you can skip it:

```bash
uv venv
uv pip install gymnasium numpy
# Linux / macOS
PYTHONPATH=src .venv/bin/python webapp/server.py
# Windows
$env:PYTHONPATH="src"; .venv\Scripts\python.exe webapp\server.py
```

The whole UI and the `qlearning` agent work; only the `dqn` option in the agent
dropdown will fail (it needs torch).

## Project structure

```
src/rl_games/
├── cli.py                  # argument parsing and output formatting only
├── registry.py             # which agents exist, where they are saved
├── evaluate.py             # run an agent greedily, without learning
├── envs.py                 # env construction and observation bounds
└── agents/
    ├── base.py             # shared hyperparameters, epsilon schedule, predict()
    ├── qlearning.py        # Tabular Q-Learning agent (implemented)
    └── dqn.py              # DQN agent from scratch (PyTorch, exercises)
webapp/
├── server.py               # stdlib HTTP server + JSON API (train/run/status)
└── index.html              # single-page UI: form, learning curve, animation
```

Saves are written to `saves/` in the working directory, one file per
(agent, environment) pair — e.g. `saves/qlearning_LunarLander-v3.pkl`.

## Using it as a library

The CLI is a thin wrapper, so everything is reachable from Python:

```python
from rl_games import registry, evaluate, envs

agent = registry.load_or_create("qlearning", "CartPole-v1")
agent.train(total_episodes=2000)

env = envs.make("CartPole-v1")
print(evaluate.run_episodes(agent, env, n_episodes=10))
env.close()
```

## Choosing an environment

Every command takes `--env`, defaulting to `LunarLander-v3`:

```bash
rlgames train qlearning --env CartPole-v1 --episodes 5000
rlgames train dqn       --env Acrobot-v1  --episodes 500
```

Most Gymnasium environments with a **discrete action space** work with no
extra code: the action count and observation shape are read from the
environment itself.

The one exception is tabular Q-learning on an environment that reports an
**unbounded** observation space. Bin edges cannot be placed between `-inf`
and `+inf`, so those environments need practical ranges added to `OBS_BOUNDS`
in `src/rl_games/envs.py`:

```python
OBS_BOUNDS: dict[str, tuple[np.ndarray, int]] = {
    "LunarLander-v3": (np.array([[-1.5, 1.5], ...]), 2),
    #                  ^ [low, high] per continuous dim       ^ trailing
    #                                                           binary dims
}
```

Run `rlgames list` to see which environments have entries. DQN never needs
them.

## Exercises

The **tabular Q-Learning** agent (`agents/qlearning.py`) is implemented in this
version and works end to end -- use it as a complete reference. The **DQN**
slots still raise `NotImplementedError` until you fill them in:

| # | Where | What | Status |
|:---:|---|---|:---:|
| 1 | `agents/dqn.py` → `QNetwork.__init__` | Build the fully-connected layers | todo |
| 2 | `agents/dqn.py` → `QNetwork.forward` | Run a batch of states through them | todo |
| 3 | `agents/dqn.py` → `DQNAgent.select_action` | Epsilon-greedy over the network | todo |
| 4 | `agents/dqn.py` → `DQNAgent._learn` | One Bellman gradient step | todo |
| 5 | `agents/qlearning.py` → `discretize` | Continuous observation → table key | **done** |
| 6 | `agents/qlearning.py` → `select_action` | Epsilon-greedy over the Q-table | **done** |
| 7 | `agents/qlearning.py` → `_update` | The temporal-difference update | **done** |

Each DQN stub carries a comment describing what it needs to do and which
attributes are already available. Suggested order: study 5 → 6 → 7 (the working
tabular agent), then implement 1 → 2 → 3 → 4 (DQN).

Check your progress with:

```bash
rlgames train qlearning --env CartPole-v1 --episodes 2000
rlgames load qlearning --env CartPole-v1 --eval
```

CartPole is the better signal than LunarLander — discretisation costs a
tabular agent most of the LunarLander state, so a mediocre score there does
not mean your code is wrong.

Reference solutions live in `CHEATSHEET.md`, which is gitignored: it is in
your working copy but never committed.
