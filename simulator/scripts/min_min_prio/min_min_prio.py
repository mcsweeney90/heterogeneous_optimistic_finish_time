#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:46:31 2019

TODO: re-labeled OFT-V to OFT-VI since changed numbering in OFT_priorities so should ideally rerun this (takes 2 hours and not really necessary
so this is low priority).

@author: Tom
"""

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
import dill
from collections import defaultdict 
from timeit import default_timer as timer
import sys
sys.path.append('../../') # Quick fix to let us import modules from main directory. 
import Environment    # Node classes and functions.
from Static_heuristics import HEFT, OFT_priorities, HBMCT


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
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12

####################################################################################################

# Environments.

single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

####################################################################################################

weightings = ["M", "WM-I", "OFT-VI"]

# Cholesky.
start = timer()
nb = 128
n_tasks = [35, 220]#, 680, 1540, 2925, 4960, 7770, 11480]
chol_reductions = defaultdict(lambda: defaultdict(list))
for env in [single, multiple]:
    with open("data/{}_cholesky_nb{}.txt".format(env.name, nb), "w") as dest:            
        env.print_info(filepath=dest)
        for nt in n_tasks:
            dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))
            dag.print_info(filepath=dest)
            mst = dag.minimal_serial_time(platform=env)
            print("Minimal serial time: {}\n".format(mst), file=dest)   
                        
            table = dag.optimistic_finish_times()   

            for w in weightings:
                if w == "OFT-VI":
                    task_list = OFT_priorities(dag, platform=env, selection=w, table=table)
                else:
                    task_list = dag.sort_by_upward_rank(platform=env, weighting=w)
                mkspan = HEFT(dag, platform=env, task_list=task_list)                
                print("HEFT with {} weighting makespan: {}".format(w, mkspan), file=dest) 
                
                mm_mkspan = HBMCT(dag, platform=env, task_list=task_list, batch_policy="min-min")
                print("HEFT-MM makespan: {}".format(mm_mkspan), file=dest)
                
                reduction = 100 - (mm_mkspan / mkspan) * 100
                chol_reductions[env.name][w].append(reduction)      
                print("Reduction: {}\n".format(reduction), file=dest)
                 
            print("--------------------------------------------------------\n", file=dest)             
    
        print("--------------------------------------------------------", file=dest)
        print("SUMMARY", file=dest)
        print("--------------------------------------------------------", file=dest)   
        n_dags = len(n_tasks)      
        for w in weightings:
            mean_reduction = np.mean(chol_reductions[env.name][w])
            best, worst = max(chol_reductions[env.name][w]), min(chol_reductions[env.name][w])
            print("Weighting: {}.".format(w), file=dest)
            print("Mean reduction: {}.".format(mean_reduction), file=dest)
            print("Best reduction: {}".format(best), file=dest)  
            print("Worst reduction: {}\n".format(worst), file=dest) 
            
# Save the reductions so can plot again later...
with open('data/reductions_cholesky_nb{}.dill'.format(nb), 'wb') as handle:
    dill.dump(chol_reductions, handle)
    
#"""Random."""
n_dags = 180
rand_reductions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for env in [single, multiple]:
    env.print_info()
    for ccr in ["1_5", "5_100"]:
        with open("data/{}_random_ccr_{}.txt".format(env.name, ccr), "w") as dest:            
            env.print_info(filepath=dest)
            count = 0
            for app in os.listdir('../../graphs/random/{}/CCR_{}'.format(env.name, ccr)):
                count += 1
#                if count > 1:
#                    break
                print("Starting DAG number {}...".format(count))
                dag = nx.read_gpickle('../../graphs/random/{}/CCR_{}/{}'.format(env.name, ccr, app))
                dag.print_info(platform=env, filepath=dest)   
                                
                # Need to compute table for the OFT weighting.
                table = dag.optimistic_finish_times() 
                              
                for w in weightings:
                    if w == "OFT-VI":
                        task_list = OFT_priorities(dag, platform=env, selection=w, table=table)
                    else:
                        task_list = dag.sort_by_upward_rank(platform=env, weighting=w)         
                    
                    mkspan = HEFT(dag, platform=env, task_list=task_list)                
                    print("HEFT with {} weighting makespan: {}".format(w, mkspan), file=dest) 
                    
                    mm_mkspan = HBMCT(dag, platform=env, task_list=task_list, batch_policy="min-min")
                    print("HEFT-MM makespan: {}".format(mm_mkspan), file=dest)
                    
                    reduction = 100 - (mm_mkspan / mkspan) * 100
                    rand_reductions[env.name][ccr][w].append(reduction)      
                    print("Reduction: {}\n".format(reduction), file=dest)
                    
                print("--------------------------------------------------------\n", file=dest)                
                
            print("--------------------------------------------------------", file=dest)
            print("SUMMARY", file=dest)
            print("--------------------------------------------------------", file=dest)            
            for w in weightings:
                mean_reduction = np.mean(rand_reductions[env.name][ccr][w])
                best, worst = max(rand_reductions[env.name][ccr][w]), min(rand_reductions[env.name][ccr][w])
                print("Weighting: {}.".format(w), file=dest)
                print("Mean reduction: {}.".format(mean_reduction), file=dest)
                print("Best reduction: {}".format(best), file=dest)  
                print("Worst reduction: {}\n".format(worst), file=dest)            
                

# Save the reductions so can plot later if I want...
with open('data/reductions_random.dill', 'wb') as handle:
    dill.dump(rand_reductions, handle)
    
elapsed = timer() - start
print("This took {} minutes".format(elapsed / 60))