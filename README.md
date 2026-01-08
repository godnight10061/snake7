# snake7

Minimal Snake Gymnasium environment + PPO(LSTM) agent (via `sb3_contrib.RecurrentPPO`).

## Install

```bash
pip install -e .
```

## Quickstart

- Run tests: `pytest -q`
- Train (GPU + saves `best_model.zip` via eval): `python -m snake7.train --total-timesteps 600000 --device cuda --model-out runs/snake_ppo_lstm.zip --best-model-out runs/best_model.zip`
- Train (NEAT): `python -m snake7.train_neat --generations 200 --genome-out runs/best_genome.pkl --config-out runs/neat_config.txt`
- Eval: `python -m snake7.eval --model runs/best_model.zip --episodes 50 --deterministic`
- Watch: `python -m snake7.watch --model runs/best_model.zip --fps 12 --deterministic`

## Environment

- Actions: `0=straight, 1=left, 2=right` (relative to current heading)
- Observation: 9 floats `[dx, dy, dir_onehot(4), danger_front, danger_left, danger_right]`
- Training defaults add anti-stall shaping (hunger truncation + small distance shaping); see `python -m snake7.train --help`
