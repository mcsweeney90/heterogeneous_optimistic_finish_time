#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:46:31 2019

Method we investigated for improving any complete task ranking in HEFT. The idea is to divide the task list into "groups" of independent (no precedence constraints between them) tasks 
as in the Hybrid Balanced Minimum Completion Time (HBMCT) heuristic of Zhao and Sakellariou (2004), and schedule them according to the classic min-min heuristic. This effectively
corresponds to HEFt with a different task priority list and consistently improved on the original but the gains were only minor and the additional computational cost significant so we 
ultimately elected not to include this investigation in the final paper.

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
from Static_heuristics import HEFT, HBMCT

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

weightings = ["HEFT", "HEFT-WM"]

#######################################################################

"""Cholesky DAGs."""

#######################################################################

start = timer()
nb = 128
n_tasks = [35, 220, 680]#, 1540, 2925, 4960, 7770, 11480, 16215, 22100]
chol_reductions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for nb in [128, 1024]:
    for env in [single, multiple]:
        with open("results/{}/cholesky_nb{}.txt".format(env.name, nb), "w") as dest:            
            env.print_info(filepath=dest)
            for nt in n_tasks:
                dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))
                dag.print_info(filepath=dest)
                mst = dag.minimal_serial_time(platform=env)
                print("Minimal serial time: {}\n".format(mst), file=dest)   
                            
                table = dag.optimistic_finish_times()   
    
                for w in weightings:
                    task_list = dag.sort_by_upward_rank(platform=env, weighting=w)
                    mkspan = HEFT(dag, platform=env, task_list=task_list)                
                    print("{} weighting makespan: {}".format(w, mkspan), file=dest) 
                    
                    mm_mkspan = HBMCT(dag, platform=env, task_list=task_list, batch_policy="min-min")
                    print("HBMCT-MM makespan: {}".format(mm_mkspan), file=dest)
                    
                    reduction = 100 - (mm_mkspan / mkspan) * 100
                    chol_reductions[env.name][nb][w].append(reduction)      
                    print("Reduction: {}\n".format(reduction), file=dest)
                     
                print("--------------------------------------------------------\n", file=dest)             
        
            print("--------------------------------------------------------", file=dest)
            print("SUMMARY", file=dest)
            print("--------------------------------------------------------", file=dest)   
            n_dags = len(n_tasks)      
            for w in weightings:
                mean_reduction = np.mean(chol_reductions[env.name][nb][w])
                best, worst = max(chol_reductions[env.name][nb][w]), min(chol_reductions[env.name][nb][w])
                print("Weighting: {}.".format(w), file=dest)
                print("Mean reduction: {}.".format(mean_reduction), file=dest)
                print("Best reduction: {}".format(best), file=dest)  
                print("Worst reduction: {}\n".format(worst), file=dest) 
            
# Save the reductions so can plot again later...
with open('data/cholesky_reductions.dill', 'wb') as handle:
    dill.dump(chol_reductions, handle)
    
#######################################################################

"""Random DAGs."""

#######################################################################

#start = timer()
#n_dags = 180
#rand_reductions = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
#for env in [single, multiple]:
#    env.print_info()
#    for acc in ["low_acc", "high_acc"]:
#        for ccr in ["0_10", "10_20", "20_50"]:
#            with open("results/{}/{}_CCR_{}.txt".format(env.name, acc, ccr), "w") as dest:            
#                env.print_info(filepath=dest)
#                count = 0
#                for app in os.listdir('../../graphs/random/{}/{}/CCR_{}'.format(env.name, acc, ccr)):
#                    count += 1
##                    if count > 1: # MAKE SURE TO UNCOMMENT BEFORE LEAVING TO RUN!
##                        break
#                    print("Starting DAG number {}...".format(count))
#                    dag = nx.read_gpickle('../../graphs/random/{}/{}/CCR_{}/{}'.format(env.name, acc, ccr, app))
#                    dag.print_info(platform=env, filepath=dest)                                       
#                                  
#                    for w in weightings:
#                        task_list = dag.sort_by_upward_rank(platform=env, weighting=w)                                 
#                        mkspan = HEFT(dag, platform=env, task_list=task_list)                
#                        print("{} weighting makespan: {}".format(w, mkspan), file=dest) 
#                        
#                        mm_mkspan = HBMCT(dag, platform=env, task_list=task_list, batch_policy="min-min")
#                        print("HBMCT-MM makespan: {}".format(mm_mkspan), file=dest)
#                        
#                        reduction = 100 - (mm_mkspan / mkspan) * 100
#                        rand_reductions[env.name][acc][ccr][w].append(reduction)      
#                        print("Reduction: {}\n".format(reduction), file=dest)                        
#                    print("--------------------------------------------------------\n", file=dest)                
#                    
#                print("--------------------------------------------------------", file=dest)
#                print("SUMMARY", file=dest)
#                print("--------------------------------------------------------", file=dest)            
#                for w in weightings:
#                    mean_reduction = np.mean(rand_reductions[env.name][acc][ccr][w])
#                    best, worst = max(rand_reductions[env.name][acc][ccr][w]), min(rand_reductions[env.name][acc][ccr][w])
#                    print("Weighting: {}.".format(w), file=dest)
#                    print("Mean reduction: {}.".format(mean_reduction), file=dest)
#                    print("Best reduction: {}".format(best), file=dest)  
#                    print("Worst reduction: {}\n".format(worst), file=dest)            
#                
#
## Save the reductions so can plot later if I want...
#with open('data/random_reductions.dill', 'wb') as handle:
#    dill.dump(rand_reductions, handle)
#    
#elapsed = timer() - start
#print("This took {} minutes".format(elapsed / 60))