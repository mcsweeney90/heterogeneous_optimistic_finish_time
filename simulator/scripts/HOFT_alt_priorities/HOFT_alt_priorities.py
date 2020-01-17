#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

TODO: run this, random DAGs as well?

An alternative task prioritization phase for HOFT, using the Optimistic Cost Table from the PEFT heuristic
by Arabnejad and Barbosa (2014) instead.

The OFT table used in HOFT is extremely similar to the OCT table but traverses the DAG from the opposite direction.
Conceptually, the difference between the OFT and OCT values is like that between upward and downward rank: the former
is the optimal distance from the entry task to the current task (inclusive), whereas the OCT is the optimal distance 
from the task to an exit task (excluding the task itself). 
(There's another minor difference in that the OFT doesn't use approximate communication costs because of the
less general assumptions made about communication costs - i.e., that they depend only on the communicating Worker types.)   

In many ways the OCT alternative is actually much more intuitive since comparing the OCT value of a task on CPU and GPU 
tells us how much we expect to save in the long run by scheduling a task on one rather than the other, but in practice
we typically found the OFT version performed slightly better in general, although the difference was minor. Since we use
the OFT anyway for the processor selection in HOFT, it seems sensible to use the OFT-based prioritization as well. 
(Alternatively, the OCT could be used as the basis for alternative processor selection phases but all the options we tried
were typically worse than the OFT-based one, although this may be worth considering again in the future.)

"""

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
import dill
from collections import defaultdict 
from timeit import default_timer as timer
import sys
sys.path.append('../../') 
import Environment    
from Heuristics import HEFT, HOFT

# Set some parameters for plots.
# See here: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12
#plt.ioff() # Uncomment to suppress plots.

####################################################################################################

# Define environments to be considered.
single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

####################################################################################################    

def OCT_priorities(dag):
    """
    Computes a scheduling list of tasks in a manner similar to the standard HOFT prioritization phase
    but using the Optimistic Cost Table, as defined in Arabnejad and Barbosa (2014), instead.
    
    Parameters
    ------------------------
    dag - DAG object (see Graph.py module)
    The task DAG.        

    Returns
    ------------------------                          
    priority_list - List of Task objects
    An ordered scheduling list of tasks.  

    Notes
    ------------------------ 
    1. Rather than using dag.optimistic_cost_table we use a slightly simpler version here similar to the 
       OFT that assumes comm costs from processors of the same type are always zero, rather than using
       platform.approximate_comm_cost. 
           
    """ 
    # Compute the OCT table - see Note 1.
    OCT = defaultdict(lambda: defaultdict(float))  
    backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
    for task in backward_traversal:
        if task.exit:
            for p in ["C", "G"]:
                OCT[task][p] = 0
            continue
        for p in ["C", "G"]:
            child_values = []
            for child in dag.DAG.successors(task):
                c1, c2 = OCT[child]["C"] + child.CPU_time, OCT[child]["G"] + child.GPU_time
                if p == "G":
                    c1 += task.comm_costs["{}C".format(p)][child.ID]
                else:
                    c2 += task.comm_costs["{}G".format(p)][child.ID]
                child_values.append(min(c1, c2))
            OCT[task][p] = max(child_values) 
            
    # Compute upward rank of all tasks to ensure precedence constraints are met.      
    backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
    task_ranks = {}
    for t in backward_traversal:
        c1 = OCT[t]["C"] + t.CPU_time
        c2 = OCT[t]["G"] + t.GPU_time
        task_ranks[t] = max(c1, c2) / min(c1, c2)
        try:
            task_ranks[t] += max(task_ranks[s] for s in dag.DAG.successors(t))
        except ValueError:
            pass             
    priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))
    return priority_list


#######################################################################

"""Cholesky DAGs."""

#######################################################################

start = timer()
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]
chol_speedups, chol_mkspans = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for nb in [128, 1024]:
    print("\nStarting tile size {}...".format(nb))
    for env in [single, multiple]:
        print("\nStarting environment {}...".format(env.name))
        with open("results/cholesky/{}_nb{}.txt".format(env.name, nb), "w") as dest:            
            env.print_info(filepath=dest)
            for nt in n_tasks:
                print("Starting size {}...".format(nt))
                dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))
                dag.print_info(filepath=dest)
                mst = dag.minimal_serial_time(platform=env)
                print("Minimal serial time: {}\n".format(mst), file=dest)   
                chol_mkspans[env.name][nb]["MST"].append(mst)
                
                OFT = dag.optimistic_finish_times()    

                heft_mkspan = HEFT(dag, platform=env)                
                chol_mkspans[env.name][nb]["HEFT"].append(heft_mkspan)
                print("HEFT makespan: {}\n".format(heft_mkspan), file=dest)  
                
                for h in ["HOFT", "alt_HOFT"]:
                    if h == "HOFT":
                        mkspan = HOFT(dag, platform=env, table=OFT)
                    elif h == "alt_HOFT":
                        task_list = OCT_priorities(dag)
                        mkspan = HOFT(dag, platform=env, table=OFT, priority_list=task_list) 
                    chol_mkspans[env.name][nb][h].append(mkspan)  
                    sp = 100 - (mkspan / heft_mkspan) * 100
                    chol_speedups[env.name][nb][h].append(sp)
                    print("{} makespan: {}".format(h, mkspan), file=dest)           
                    print("Speedup (%) over HEFT: {}\n".format(sp), file=dest)     
                print("--------------------------------------------------------\n", file=dest)                  
    
# Save the makespans and speedups so can use later.
with open('results/chol_mkspans.dill'.format(nb), 'wb') as handle:
    dill.dump(chol_mkspans, handle)
with open('results/chol_speedups.dill'.format(nb), 'wb') as handle:
    dill.dump(chol_speedups, handle)
    
elapsed = timer() - start
print("Cholesky part took {} minutes".format(elapsed / 60))


"""Plotting..."""
try:
    chol_speedups
except NameError:
     with open('results/chol_speedups.dill', 'rb') as file:
         chol_speedups = dill.load(file)
         
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]

titles = {"Single_GPU" : "1 GPU, 1 CPU", "Multiple_GPU" : "4 GPUs, 4 CPUs"}
heuristics = ["HOFT", "alt_HOFT"]
preferences = {"HOFT" : [":", "o"], "alt_HOFT" : ["--", "s"]}
for nb in [128, 1024]:
    fig = plt.figure(dpi=400)            
    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("NUMBER OF TASKS", labelpad=10) 
    ax.set_ylabel("MAKESPAN REDUCTION VS HEFT (%)", labelpad=15)   
    
    ax1 = fig.add_subplot(211)
    plt.xscale('log')
    ax1.plot(n_tasks, len(n_tasks) * [0], linestyle='-', label="HEFT", color='#FBC15E')
    for h in heuristics:
        ax1.plot(n_tasks, chol_speedups["Single_GPU"][nb][h], linestyle=preferences[h][0], marker=preferences[h][1], label=h)
    ax1.set_xticklabels([])
    ax1.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
    ax1.set_title(titles["Single_GPU"], color="black", fontsize='large', family='serif')
    
    ax2 = fig.add_subplot(212)
    plt.xscale('log')
    ax2.plot(n_tasks, len(n_tasks) * [0], linestyle='-', label="HEFT", color='#FBC15E')
    for h in heuristics: 
        ax2.plot(n_tasks, chol_speedups["Multiple_GPU"][nb][h], linestyle=preferences[h][0], marker=preferences[h][1], label=h)        
    #ax2.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
    ax2.set_title(titles["Multiple_GPU"], color="black", fontsize='large', family='serif')
    
    plt.savefig('plots/speedup_cholesky_nb{}'.format(nb), bbox_inches='tight') 