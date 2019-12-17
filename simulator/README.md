# Simulator

A Python framework for simulating the scheduling of user-defined task DAGs on user-defined CPU-GPU target platforms.

## Prerequisites
To install all relevant python packages:
```
pip install -r requirements.txt
```

## Map

`graphs`: Task DAGs used for results referenced in paper, stored in the interest of reproducibility.

`scripts`: Scripts used to generate all results presented in the paper (and some additional ones that we didn't use).

`Environment.py`: Module containing `PU` and `Node` classes which allow us to represent CPU-GPU computing environments.

`Graph.py`: Module containing `Task` and `DAG` classes which allow us to represent task graphs DAGs.

`Static_heuristics.py`: Implementations of existing static scheduling heuristics, as well as some new ones (most pertinently, HOFT). 
