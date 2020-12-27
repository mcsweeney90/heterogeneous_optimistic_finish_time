# Simulator

Python framework for simulating the scheduling of user-defined task DAGs on user-defined CPU-GPU target platforms.

## Prerequisites

This has only been tested for Python **>= 3.6** so performance for older versions cannot be guaranteed. In particular, I imagine there may be issues since it is often implicitly assumed that `dict` data types (and therefore also `defaultdicts` from the `collections` package) are ordered by insertion time. 

To install all relevant Python packages:
```
pip install -r requirements.txt
```

## Navigation

`graphs`: Task DAGs used for results referenced in paper. Stored using `gpickle` as it is faster to load an existing DAG than create a new one (and in the interest of reproducibility).

`scripts`: Scripts used to generate all results presented in the paper (and some additional ones that we didn't use).

`Environment.py`: Module containing `PU` and `Node` classes which allow us to represent CPU-GPU computing environments.

`Graph.py`: Module containing `Task` and `DAG` classes which allow us to represent task DAGs.

`Heuristics.py`: Implementations of existing static scheduling heuristics, as well as some new ones (most pertinently, HOFT). 
