# Dreamer-v1-revisited

A re-implementation of Dreamer v1 written in 2026 for learning purposes, built on top of the world model from [PlaNet-revisited](https://github.com/Farag-Y/PlaNet-revisited), using a modern Python project structure (uv, Hydra config, modular model layout).

Inspired by [@Kaixhin's PyTorch implementation](https://github.com/Kaixhin/PlaNet).

---

## What is Dreamer v1?

Dreamer ([Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603), Hafner et al. 2020) is a model-based reinforcement learning agent that learns a world model from pixel observations, then learns a policy and value function purely by "imagining" rollouts inside that world model's latent space.

It shares its world model with PlaNet — the **Recurrent State Space Model (RSSM)** — but replaces PlaNet's online CEM planning with actor-critic learning:

- **Recurrent State Space Model (RSSM):** the same latent dynamics model as PlaNet, with a deterministic (GRU) path and a stochastic (Gaussian) path, giving the agent a compact, predictive representation of the environment.
- **Learning from images only:** observations are encoded into latent states through a CNN encoder; a decoder and reward model are trained to reconstruct them, supervised only by raw pixels and scalar rewards.
- **Behavior learning via latent imagination:** instead of re-planning with CEM at every step, Dreamer trains an actor and a value model entirely on imagined trajectories rolled out inside the RSSM, backpropagating analytic gradients through the dynamics to improve the policy.
- **Fast inference:** at test time, acting is a single forward pass through the actor — no online search — since all the planning work has been distilled into the policy during training.

The result is an agent that solves continuous control tasks (Pendulum, BipedalWalker, MuJoCo locomotion, dm_control suite) directly from pixel input, and does so more efficiently than PlaNet's CEM-based planning.

---

## This Re-implementation

This repo starts from the [PlaNet-revisited](https://github.com/Farag-Y/PlaNet-revisited) codebase — same RSSM, encoder, observation model, and reward model — and extends it toward Dreamer v1 by replacing CEM planning with actor-critic learning in latent imagination. Goals:

- Clean, readable code that maps closely to the paper.
- Modern Python tooling: [uv](https://docs.astral.sh/uv/) for environment management, [Hydra](https://hydra.cc) for configuration.
- Modular layout: each model component (RSSM, observation model, reward model, actor/value) lives in its own file under `models/`.
- Uses `gymnasium` (the maintained fork of OpenAI Gym) instead of the original `gym`.
- Supports `dm_control` environments (the suite used in the original paper) alongside gymnasium.

It does **not** aim to reproduce the exact benchmark numbers from the paper.

---

## Project Structure

```
Dreamer-v1-revisited/
├── conf/
│   └── config.yaml        # All hyperparameters via Hydra
├── models/
│   ├── rssm.py            # Recurrent State Space Model (world model, from PlaNet-revisited)
│   ├── encoder.py
│   ├── observation_model.py
│   ├── reward_model.py
│   └── planner.py          # CEM planner (being replaced by actor-critic latent imagination)
├── env_wrapper.py          # Gymnasium + dm_control wrappers (image preprocessing)
├── experience_replay.py    # Replay buffer
├── main.py                 # Training entry point
└── utils.py
```

---

## Installation

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone <repo-url>
cd Dreamer-v1-revisited
uv sync
```

Run:

```bash
uv run python main.py
```

Override any config value inline:

```bash
uv run python main.py env=HalfCheetah-v5 seed=42
```

---

## Environments

Two environment families are supported. Set `env` in `conf/config.yaml` or via the command line.

**Gymnasium** — install with `uv sync` (included by default):

| Category | Examples |
|---|---|
| Classic Control | `Pendulum-v1`, `MountainCarContinuous-v0` |
| Box2D | `BipedalWalker-v3`, `CarRacing-v3` |
| MuJoCo | `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, ... |

**dm_control** — also included by default. These are the environments used in the original PlaNet paper:

| Environment | dm_control task |
|---|---|
| `cartpole-swingup` | Cartpole swingup from hanging position |
| `finger-spin` | Robotic finger spinning a body |
| `cheetah-run` | Half-cheetah running |
| `reacher-easy` | 2-link arm reaching |
| `cup-catch` | Ball-in-cup |
| `walker-walk` | Bipedal walker |

```bash
uv run python main.py env=cartpole-swingup
uv run python main.py env=finger-spin
```

> **macOS note:** dm_control rendering uses mujoco's native CGL renderer and does not require a system OpenGL installation.

---

## Playing Environments Manually

`play.py` lets you control any supported environment yourself using the keyboard. Useful for getting a feel for an environment before training.

```bash
uv run python play.py env=Pendulum-v1
uv run python play.py env=cartpole-swingup
```

A pygame window opens showing the environment at full resolution. Key bindings are displayed as an overlay at the bottom of the window and printed to the terminal on startup. Up to 4 action dimensions are mapped:

| Keys | Action dim |
|---|---|
| `←` / `→` | action\[0\] |
| `↑` / `↓` | action\[1\] |
| `A` / `D` | action\[2\] |
| `W` / `S` | action\[3\] |

Hold a key to push the action to its maximum; release to return to zero. `R` resets the episode, `Q` quits.

---

## Training on Vast.ai

`scripts/train_vastai.py` (run via `make train-vast`) is an interactive helper that rents a GPU on [Vast.ai](https://vast.ai), uploads the project, runs training, streams logs, and **automatically destroys the instance** when training finishes.

### Prerequisites

1. Install the Vast.ai CLI (requires the `vastai` dependency group):

   ```bash
   uv sync --group vastai
   ```

2. Authenticate:

   ```bash
   vastai set api-key <YOUR_KEY>
   ```

   Get your key at <https://cloud.vast.ai/account/>.

3. Add your API key to `.env` in the project root (see [`.env.example`](.env.example)):

   ```
   VAST_API_KEY=<your_key>
   ```

   The key is baked into the remote runner so the instance can self-destruct via the API when training ends.

### Usage

```bash
make train-vast
```

The script walks you through four interactive prompts:

| Prompt | Options |
|---|---|
| GPU type | RTX 4090, RTX 3090, RTX 3060, A100, H100, A6000 |
| CUDA version | 12.1 or 12.4 (both use PyTorch 2.4 images) |
| Entrypoint | full training run or evaluation only |
| Extra Hydra overrides | e.g. `env=HalfCheetah-v5 seed=42` |
| Max price per hour | e.g. `0.50` |

It then lists up to 10 matching offers (sorted by price) and lets you pick one. A confirmation prompt is shown before any money is spent.

Pass `--auto` to skip the offer picker and confirmation and use the cheapest match:

```bash
make train-vast ARGS="--auto"
```

### What it does (step by step)

1. **Preflight** — checks that `vastai` CLI is installed and authenticated, and that `VAST_API_KEY` is set in `.env`.
2. **Search offers** — queries Vast.ai for rentable instances matching your GPU, CUDA, and price constraints.
3. **Create instance** — provisions the selected offer with 50 GB disk and SSH access using the chosen PyTorch Docker image.
4. **Wait for boot** — polls until the instance status is `running` (5-minute timeout).
5. **Upload code** — rsyncs the project to `/workspace/` on the instance, excluding `.git/`, `outputs/`, `results/`, and caches.
6. **Install deps** — installs system OpenGL/EGL libraries needed by MuJoCo/dm-control, then runs `uv sync` on the instance.
7. **Upload runner** — generates `remote_run.sh` with your entrypoint and Hydra overrides baked in; it calls the Vast.ai API to destroy the instance when the run exits.
8. **Launch detached** — starts training via `nohup` so it survives SSH disconnects.
9. **Stream logs** — tails `/workspace/training.log` live. Press `Ctrl-C` to detach — training continues server-side and the instance self-destructs when done.

To reconnect to a running instance after detaching:

```bash
ssh -p <PORT> -o StrictHostKeyChecking=no root@<HOST>
tail -f /workspace/training.log
```

The reconnect command is printed when you detach.

---

## References

- [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603) — Hafner et al., 2020
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) — Hafner et al., 2019 (PlaNet, the world model this repo builds on)
- [google-research/dreamer](https://github.com/google-research/dreamer) — original TensorFlow implementation
- [Farag-Y/PlaNet-revisited](https://github.com/Farag-Y/PlaNet-revisited) — this repo's base implementation (RSSM, encoder, observation/reward models)
- [Kaixhin/PlaNet](https://github.com/Kaixhin/PlaNet) — PyTorch implementation that inspired the base PlaNet code
