# Random

Randomly-generated task DAGs.

## Standard Task Graph (STG) set

Directory `STG` contains 180 randomly-generated DAGs with 1002 tasks, in their original format, from the Standard Task Graph Set (http://www.kasahara.cs.waseda.ac.jp/schedule/). 
This is a set of task graphs, some from real applications and some randomly generated, proposed as a standard test suite for scheduling algorithms
in Tobita, T., Kasahara, H.: A standard task graph set for fair evaluation of multi-processor scheduling algorithms. J. Sched. 5(5), 379–394 (2002).

## DAGs without costs

`convert_and_save.py` converts the STG files to DAG objects and saves them (without setting any computation or communication costs) using `gpickle` in the directory `topologies`.

## DAGs with costs

`set_costs_and_save.py` loads the DAG objects from `topologies`, then makes six copies according to different cost regimes for both the Single GPU (1 GPU, 1 octacore CPU) and Multiple GPU (4 GPU, 4 octacore CPU) platforms, and saves them in the directory `Single_GPU` and `Multiple_GPU`, respectively. The different cost regimes are:

* `low_acc`: task acceleration ratios are sampled randomly from a Gamma distribution with mean and standard deviation 5.0. Communication costs are randomly-generated such that the DAG computation-to-communication ratio (CCR) falls into each of the intervals (0, 10), (10, 20) and (20, 50). The DAGs are then saved using `gpickle` in the subdirectories `CCR_0_10`, `CCR_10_20`, and `CCR_20_50`. 

* `high_acc`: task acceleration ratios are sampled randomly from a Gamma distribution with mean and standard deviation 50.0. Communication costs are randomly-generated such that the DAG computation-to-communication ratio (CCR) falls into each of the intervals (0, 10), (10, 20) and (20, 50). The DAGs are then saved using `gpickle` in the subdirectories `CCR_0_10`, `CCR_10_20`, and `CCR_20_50`. 





