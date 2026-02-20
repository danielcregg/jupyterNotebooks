# Dynamic Economic Emissions Dispatch via Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-00A36C?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

A Jupyter Notebook and Gymnasium environment for solving the **Dynamic Economic Emissions Dispatch** (DEED) problem using modern reinforcement learning (PPO).

## Overview

The DEED problem is a multi-objective optimization challenge in power systems engineering: schedule 10 thermal generating units over a 24-hour horizon to minimize both **fuel cost** and **pollutant emissions** while satisfying power demand and operational constraints.

This repository provides:

- **`deed_env.py`** -- A Gymnasium-compatible RL environment implementing the DEED problem with a standard IEEE 10-generator benchmark system
- **`DEED.ipynb`** -- A Jupyter notebook demonstrating PPO training and **Pareto front analysis** for the multi-objective cost vs. emissions trade-off
- **`tests/test_deed_env.py`** -- Comprehensive test suite (37 tests) for the environment

### Research Contribution

The notebook includes a **Pareto front analysis** that trains RL agents across multiple cost/emissions weight combinations to approximate the Pareto-optimal trade-off curve. This demonstrates how RL can be used for multi-objective dispatch optimization without requiring traditional mathematical programming.

## Problem Formulation

**Cost function** (with valve-point loading effects):

$$F_C = \sum_{m=1}^{24} \sum_{i=1}^{10} \left[ a_i + b_i P_i^m + c_i (P_i^m)^2 + |d_i \sin(e_i (P_i^{\min} - P_i^m))| \right]$$

**Emissions function**:

$$F_E = \sum_{m=1}^{24} \sum_{i=1}^{10} E \left[ \alpha_i + \beta_i P_i^m + \gamma_i (P_i^m)^2 + \eta_i \exp(\delta_i P_i^m) \right]$$

**Constraints**: Power balance (Kron's loss formula), generation limits, ramp rate limits.

## Dependencies

- `gymnasium>=0.29.0`
- `stable-baselines3>=2.0.0`
- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `matplotlib>=3.5.0`
- `scipy>=1.7.0`
- `jupyter>=1.0.0`
- `pytest>=7.0.0`

## Getting Started

### Installation

```bash
git clone https://github.com/danielcregg/jupyterNotebooks.git
cd jupyterNotebooks
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook DEED.ipynb
```

### Run Tests

```bash
pytest tests/ -v
```

### Quick Start (Python)

```python
from deed_env import DEEDEnv

env = DEEDEnv(Wc=0.5, We=0.5)
obs, info = env.reset(seed=42)

# Run a random episode
terminated = False
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

print(f"Total cost: {info['total_cost']:,.2f}")
print(f"Total emissions: {info['total_emissions']:,.2f}")
```

## Notebook Sections

1. **Introduction** -- DEED problem definition and mathematical formulation
2. **Environment Setup** -- Create, explore, and visualize the Gymnasium environment
3. **PPO Training** -- Train a PPO agent using Stable Baselines3
4. **Results Analysis** -- Evaluate dispatch schedules, plot costs/emissions, compare with baseline
5. **Pareto Front Analysis** -- Multi-objective optimization across weight combinations

## Environment Details

| Property | Value |
|---|---|
| Generators | 10 (1 slack + 9 agent-controlled) |
| Horizon | 24 hours |
| Steps per episode | 216 (24 hours x 9 generators) |
| Action space | Discrete(101) per generator |
| Observation space | Box(14) -- normalized state features |
| Reward | Negative weighted sum of cost + emissions + penalties |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
