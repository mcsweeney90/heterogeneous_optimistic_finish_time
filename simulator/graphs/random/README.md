Randomly-generated task DAGs.

Multiple_GPU:
DAG objects with computation and communication costs for the multiple GPU (4 GPU, 4 octacore CPU) testing platform.

Single_GPU:
DAG objects with computation and communication costs for the single GPU (1 GPU, 1 octacore CPU) testing platform.

STG:
Randomly-generated DAGs from the Standard Task Graph Set, http://www.kasahara.cs.waseda.ac.jp/schedule/. 
This is a set of task graphs, some from real applications and some randomly generated, proposed as a standard test suite for scheduling algorithms
in Tobita, T., Kasahara, H.: A standard task graph set for fair evaluation of multi-processor scheduling algorithms. J. Sched. 5(5), 379–394 (2002).
(In the original storage format.)

topologies:
DAGs with 1002 tasks from the STG converted to DAG objects, without computation and communication costs, and stored using gpickle.

convert_and_save_stg.py : 
Helper script for converting stg files to DAG objects and saving. 

set_costs_and_save.py : 
Helper script for reading unweighted DAG objects, setting costs and saving.


