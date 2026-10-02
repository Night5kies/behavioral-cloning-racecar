# Behavioral Cloning Race Car

A neural network policy that learns to drive a top-down race car by imitating an expert controller (behavioral cloning), built with PyTorch on a Gymnasium/Box2D `CarRacing`-style environment. It also includes an experimental PPO actor-critic implementation as a reinforcement-learning extension.

The racing simulator (`env/car_racing.py`, adapted from Gymnasium's CarRacing), the Pure Pursuit + Stanley expert controller (`expert.py`), the dataset utilities, and the CLI were provided as a starter framework. The data-collection noise model, the policy network, the training loop, the PPO extension, and the trained checkpoint are my own work.

## Approach

**Expert demonstrations.** A Pure Pursuit + Stanley controller drives the car while (observation, action) pairs are recorded. To make the learned policy more robust to its own mistakes, independent zero-mean Gaussian noise is added to the expert's actions before they are executed (std 0.05 on steering, 0.02 on gas and brake), then clipped to valid ranges. The noisy action is both executed and recorded, so the data covers states slightly off the expert's ideal trajectory. Collection runs either sequentially with rendering or in parallel across CPU cores (one seeded worker per episode), 10 episodes by default.

**Features.** Each observation is reduced to a 10-dimensional vector: car position (x, y), heading, speed, and path-relative quantities (lateral error, heading error, arclength, curvature, longitudinal and lateral velocity). Features are standardized with the dataset mean and standard deviation.

**Policy network.** An MLP mapping 10 features to 3 actions `[steer, gas, brake]`:

- 3 hidden layers of 128 units with ReLU, Xavier-uniform initialization
- `tanh` on steering (range [-1, 1]) and `sigmoid` on gas and brake (range [0, 1])

**Training.** Supervised regression on the expert actions with MSE loss and Adam (lr 1e-3), batch size 64, 100 epochs, and an 80/20 train/validation split. The model from the final epoch is saved; train and validation loss are logged every 10 epochs.

**PPO extension (experimental).** `policy.py` also contains a minimal PPO implementation: an actor-critic network with a shared 2-layer trunk, a Gaussian policy head with a learned log-std, and a value head; a rollout buffer with GAE; and a clipped-surrogate update with value loss, entropy bonus, and gradient clipping. It is not wired into the CLI and has no trained checkpoint. One simplification to note: the log-probabilities are computed on the squashed actions without a tanh/sigmoid correction.

## Installation

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Collect expert demonstrations (saved to `expert_data/obs.npy` and `expert_data/action.npy`):

```bash
# Sequential, with visualization
uv run grand_prix --collect-expert-data

# Parallel across CPU cores, no display (faster)
uv run grand_prix --collect-expert-data --fast
```

Train the behavioral cloning policy:

```bash
uv run grand_prix --train-neural-net-policy policies/my_policy.pt
```

Evaluate a trained policy in the rendered environment:

```bash
uv run grand_prix --evaluate-neural-net-policy policies/my_policy.pt
```

The feature normalization statistics are recomputed from `expert_data/obs.npy` when a policy is loaded, so collect expert data before you evaluate. Without that file the loader falls back to zero mean and unit variance, and the included checkpoint is unlikely to drive well.

## Project Structure

```
.
├── policies/
│   └── my_policy.pt          # Trained behavioral cloning checkpoint (state_dict)
├── src/grand_prix/
│   ├── __main__.py           # CLI entry point (collect / train / evaluate)
│   ├── expert.py             # Pure Pursuit + Stanley expert controller
│   ├── env/
│   │   ├── car_racing.py     # Box2D race car environment
│   │   └── collect_data.py   # Expert data collection with noise injection
│   └── neural_net/
│       ├── dataset.py        # Feature extraction, normalization, data loaders
│       └── policy.py         # Policy MLP, training loop, inference wrapper, PPO
├── pyproject.toml
└── uv.lock
```
