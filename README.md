# Heterogeneous Optimistic Finish Time (HOFT)

This repository provides the source code used to generate all results presented in the paper "An efficient static scheduling heuristic for accelerated architectures," which introduces a new static task scheduling heuristic for accelerated architectures called Heterogeneous Optimistic Finish Time (HOFT). 

## Simulator

To evaluate HOFT, we implemented a simple software framework which allows the simulated scheduling of user-defined task graphs (specifically, Directed Acyclic Graphs or DAGs) on user-defined CPU-GPU target platforms. The Python source code for this can be found in the folder `simulator`. Details on which Python packages are required and other installation information are provided in the folder's main [README](simulator/README.md) file.

## Paper

The LaTeX source for the paper and all other relevant files are contained in the folder `An_efficient_new_static_scheduling_heuristic_for_accelerated_architectures`.

## License

This project is licensed under the GPL-3.0 License - see [LICENSE](LICENSE) for details.

## References

[1] T. McSweeney, N. Walton and M. Zounon. An efficient static scheduling heuristic for accelerated architectures. Submitted to the International Conference on Computational Science 2020 (<https://www.iccs-meeting.org/iccs2020/>).
