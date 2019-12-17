# Cholesky

Task DAGs representing a tiled Cholesky factorization of a matrix.

## Real data

The folder `data` contains real BLAS and LAPACK kernel timing data from a single heterogeneous node of the University of Manchester CSF3 cluster. This is used to set the computation and communication costs of the DAGs.  

## Stored DAGs

`nb32 -- nb1024`: DAG objects representing Cholesky DAGs with tile size nb, stored using `gpickle`.

`summaries`: Basic DAG info (e.g., edge density and CCR) for all of the DAGs. 

`create_and_save_cholesky_dags.py`: Script for creating and saving the DAGs. 



