DAGs for testing.  

STG_original :
Randomly-generated DAGs from the Standard Task Graph Set, http://www.kasahara.cs.waseda.ac.jp/schedule/. 
This is a set of task graphs, some from real applications and some randomly generated, proposed as a standard test suite for scheduling algorithms
in Tobita, T., Kasahara, H.: A standard task graph set for fair evaluation of multi-processor scheduling algorithms. J. Sched. 5(5), 379–394 (2002).
(In the original storage format.)


STG_unweighted :
Graphs from the STG converted into DAG objects (with no costs set) and stored as pickle files. 


Simple :
DAG objects with costs set using the "Simple node" (1 Opteron CPU worker, 1 Kepler K20 GPU worker, Gemini interconnect). 


Summit :
DAG objects with costs set using the "Summit node" (44 Power9 CPU workers, 6 Nvidia V100 GPU workers, Infiniband interconnect). 


Simple :
DAG objects with costs set using the "Titan node" (16 Opteron CPU workers, 1 Kepler K20 GPU worker, Gemini interconnect). 


Images:
Plots, images of DAGs, etc.


NLA_DAGs.py : 
Functions for constructing DAGs representing NLA applications.


convert_and_save_stg.py : 
Helper script for converting stg files to DAG objects and saving for future use. 


set_costs_and_save.py : 
Helper script for reading unweighted DAG objects, setting costs and saving for future use.


