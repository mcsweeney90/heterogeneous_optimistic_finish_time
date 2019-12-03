Cholesky task DAGs based on real timing data from node of the UoM CSF3.  

data:
Timing data used to set the computation and communication costs of the DAGs. 

images:
Any images representing Cholesky DAGs. 

nb32 -- nb1024:
Pickled DAG objects representing Cholesky DAGs with tile size nb.

summaries:
Basic DAG info (e.g., edge density, CCR) for all of the DAGs in this section.

create_and_save_cholesky_dags.py:
Create Cholesky DAGs using cholesky function from NLA_DAGs.py. Sample from real timing data to set computation and communication costs. 

draw_cholesky.py:
Very short script for drawing and saving Cholesky DAGs.

NLA_DAGs.py:
Contains function for creating Cholesky DAG object topologies. Will add functions for other NLA applications in the future. 

