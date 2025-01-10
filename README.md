# Probabilistic PDDL Solver for Block Stacking


This is the repository for term project of PKU 2024 Fall course **[Cognitive Reasoning](https://yzhu.io/courses/core/)**

## Project Requirements

- **Background**
  - Explore probabilistic PDDL solvers in the context of block stacking.
- **Objective**
  - Implement the solver and reproduce block stacking experiments.
- **Task Details**
  - Follow the experiments outlined in the paper by [Huang, De-An, et al.](https://arxiv.org/pdf/1908.06769).
- **Evaluation**
  - Validate the solver’s performance in block stacking tasks.

## Overview

```
Project/
│
├── data_utils/
│   ├── data_generator.py # implementation of seperated scene generation
│   ├── dataset.py        # implementation of dataset
│   └── task_generator.py # implementation of task generation
│
├── model/
│   ├── encoder.py        # implementation of object and predicate encoder
│   ├── mlp.py            # implementation of MLP
│   └── sgn.py            # implementation of sgn model
|
├── engine_sgn.py         # helper functions for sgn training
├── environment.py        # implementation of block stacking environment
├── main.py               # implementation of the entire evaluation process
├── planner.py            # implementation of CP
├── rendering.py          # implementation of rendering functions
├── train_sgn.py          # implementation of sgn training function
└── README.md             # this file
```

## Installation of dependency

```
pip install torch matplotlib tqdm numpy
```

## Usage

Please generate the dataset with `data_utils/task_generator.py` and refer to `main.py` to run inference. To run experiments with pretrained sgn, please run `train_sgn.py` accordingly.
