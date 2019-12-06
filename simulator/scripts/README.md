Scripts used to generate all results presented in the paper (and some others that we decided not to include).

benchmarking_HEFT:
Benchmarking the performance of HEFT (Section 4.1 of paper).

HOFT:
Comparison of HOFT with HEFT and HEFT-WM (Section 6).

motivation:
Investigating how useful static scheduling actually is for dynamic environments. Never referenced in paper but provided the main motivation for why we decided to look at static scheduling. 

min_min_for_HEFT:
Method we investigated for improving any complete task ranking in HEFT. The idea is to divide the task list into "groups" of independent (no precedence constraints between them) tasks 
as in the Hybrid Balanced Minimum Completion Time (HBMCT) heuristic of Zhao and Sakellariou (2004), and schedule them according to the classic min-min heuristic. This effectively
corresponds to HEFT with a different task priority list and consistently improved on the original but the gains were only minor and the additional computational cost significant so we 
ultimately elected not to include this investigation in the final paper.

low_CCR:
Heuristics targeting low CCR/high-data DAGs for which HEFT (and HOFT) are likely to fail. Referred to in the conclusion of paper but this was not promising so we didn't pursue it any further.  

existing_heuristic_comparison:
Comparison of five existing static scheduling heuristics - HEFT, HBMCT, PEFT, PETS and HCPT. Not referenced in the paper but this motivated our focus on HEFT in particular, as it was the best-performing along with PEFT and its structure offered more possibilities for optimization.  






