# Jupyter Notebooks - Dynamic Economic Emissions Dispatch

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

A Jupyter Notebook exploring Dynamic Economic Emissions Dispatch (DEED) using reinforcement learning with OpenAI Gym.

## Overview

This repository contains a comprehensive Jupyter Notebook that models the Dynamic Economic Emissions Dispatch problem -- an optimization challenge in power systems engineering. The notebook implements a custom OpenAI Gym environment to simulate generator scheduling, balancing power generation costs against emissions output over time. It demonstrates how reinforcement learning techniques can be applied to solve complex energy dispatch problems.

## Features

- Custom OpenAI Gym environment for generator dispatch simulation
- Mathematical formulation of cost and emissions functions
- Power generation constraints and load demand modeling
- Integration with NumPy and Pandas for numerical computation
- Interactive notebook format with step-by-step explanations
- Compatible with Google Colab, Binder, and Azure Notebooks

## Prerequisites

- [Python](https://www.python.org/downloads/) 3.6 or higher
- [Jupyter Notebook](https://jupyter.org/install) or JupyterLab
- Required Python packages:
  - `numpy`
  - `pandas`
  - `gym` (OpenAI Gym)
  - `math`

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/danielcregg/jupyterNotebooks.git
cd jupyterNotebooks
```

Install dependencies:

```bash
pip install jupyter numpy pandas gym
```

### Usage

Launch the notebook locally:

```bash
jupyter notebook DEED.ipynb
```

Or open in an online environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/danielcregg/jupyterNotebooks/master)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielcregg/jupyterNotebooks)

## Tech Stack

- **Language:** Python 3
- **Notebook:** Jupyter
- **Frameworks:** OpenAI Gym, NumPy, Pandas
- **Domain:** Power Systems / Reinforcement Learning

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
