# Scripts

Short scripts for generating results presented or described in the paper.

## Presented in paper

`benchmarking_HEFT`: Section 4.1 of paper, benchmarking the performance of Heterogeneous Earliest Finish Time (HEFT).

`HOFT`: Section 6 of paper, comparison of HOFT with HEFT.

## Referenced in paper

`low_CCR`: Sampling-based lookahead heuristics for low CCR/high-data DAGs for which HEFT (and HOFT) are likely to fail. Referred to in the conclusion but was not promising so we didn't pursue it very far.

## For interest

`static_scheduling_in_dynamic_environments`: Investigating how useful static scheduling actually is in practice by simulating static scheduling in dynamic environments. Never referenced in paper but provided some of the motivation for our decision to look at static scheduling for CPU and GPU. 

`min_min_for_HEFT`: A method we investigated for improving any complete task ranking in HEFT. The idea is to divide the task list into "groups" of independent (no precedence constraints between them) tasks 
as in the Hybrid Balanced Minimum Completion Time (HBMCT) heuristic of Zhao and Sakellariou (2004), and schedule them according to the classic min-min heuristic. This effectively
corresponds to HEFT with a different task priority list and consistently improved on the original. However, gains were minor and the additional computational cost significant so we 
ultimately elected not to include this investigation in the final paper.

`existing_heuristic_comparison`: Comparison of several existing static scheduling heuristics. Not referenced in the paper but this motivated our focus on HEFT in particular, as it was the best-performing (along with PEFT) and its structure offered more possibilities for optimization.  






