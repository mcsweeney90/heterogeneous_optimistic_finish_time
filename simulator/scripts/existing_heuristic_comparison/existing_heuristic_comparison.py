#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of five existing static scheduling heuristics - HEFT, HBMCT, PEFT, PETS and HCPT. 

Estimated runtime: ~22 hours on a machine with an Intel i7. (SO might be a good idea to run in parts!)
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
from Heuristics import HEFT, HBMCT, PEFT, PETS, HCPT

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

heuristics = ["HEFT", "HBMCT", "PEFT", "PETS", "HCPT"]

#######################################################################

"""Cholesky DAGs."""

#######################################################################

start = timer()
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]
chol_mkspans = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for nb in [128, 1024]:
    for env in [single, multiple]:
        with open("results/{}/cholesky_nb{}.txt".format(env.name, nb), "w") as dest:            
            env.print_info(filepath=dest)
            for nt in n_tasks:
                dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))
                dag.print_info(filepath=dest)
                mst = dag.minimal_serial_time(platform=env)
                print("Minimal serial time: {}\n".format(mst), file=dest)   
                chol_mkspans[env.name][nb]["MST"].append(mst)                    
                for h in heuristics:
                    if h == "HEFT":
                        mkspan = HEFT(dag, platform=env)
                    elif h == "HBMCT":
                        mkspan = HBMCT(dag, platform=env)
                    elif h == "PEFT":
                        mkspan = PEFT(dag, platform=env)
                    elif h == "PETS":
                        mkspan = PETS(dag, platform=env)
                    elif h == "HCPT":
                        mkspan = HCPT(dag, platform=env)
                    chol_mkspans[env.name][nb][h].append(mkspan)  
                    print("{} makespan: {}\n".format(h, mkspan), file=dest)       
                print("--------------------------------------------------------\n", file=dest)                  
    
# Save the makespans so can use later.
with open('results/chol_mkspans.dill'.format(nb), 'wb') as handle:
    dill.dump(chol_mkspans, handle)
        
elapsed = timer() - start
print("Cholesky part took {} minutes".format(elapsed / 60))

#######################################################################

"""Random DAGs."""

#######################################################################

start = timer()
n_dags = 180
rand_mkspans = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
for env in [single, multiple]:
    print("\nStarting environment: {}".format(env.name))
    for acc in ["low_acc", "high_acc"]:
        print("\nStarting {} DAGs...".format(acc))
        for ccr in ["0_10", "10_20", "20_50"]:
            print("\nStarting CCR interval {}...".format(ccr))
            with open("results/{}/{}_CCR_{}.txt".format(env.name, acc, ccr), "w") as dest:            
                env.print_info(filepath=dest)
                bests = defaultdict(int)
                count = 0
                for app in os.listdir('../../graphs/random/{}/{}/CCR_{}'.format(env.name, acc, ccr)):
                    count += 1
                    # if count > 1:
                    #     break
                    print("Starting DAG number {}...".format(count))
                    dag = nx.read_gpickle('../../graphs/random/{}/{}/CCR_{}/{}'.format(env.name, acc, ccr, app))
                    dag.print_info(platform=env, filepath=dest) 

                    best = float('inf')   
                    for h in heuristics:
                        if h == "HEFT":
                            mkspan = HEFT(dag, platform=env)
                        elif h == "HBMCT":
                            mkspan = HBMCT(dag, platform=env)
                        elif h == "PEFT":
                            mkspan = PEFT(dag, platform=env)
                        elif h == "PETS":
                            mkspan = PETS(dag, platform=env)
                        elif h == "HCPT":
                            mkspan = HCPT(dag, platform=env)
                        rand_mkspans[env.name][acc][ccr][h].append(mkspan)  
                        best = min(best, mkspan)
                        print("{} makespan: {}\n".format(h, mkspan), file=dest) 
                        
                    for h in heuristics:
                        if rand_mkspans[env.name][acc][ccr][h][-1] == best:
                            bests[h] += 1
                                               
                    print("--------------------------------------------------------\n", file=dest)   
                print("--------------------------------------------------------", file=dest)
                print("SUMMARY", file=dest)
                print("--------------------------------------------------------", file=dest)            
                for h in heuristics:
                    mean_makespan = np.mean(rand_mkspans[env.name][acc][ccr][h])
                    print("Heuristic: {}.".format(h), file=dest)
                    print("Mean makespan: {}.".format(mean_makespan), file=dest)  
                    print("Number of best occurences: {}.\n".format(bests[h]), file=dest)                

# Save the makespans so can use later.
with open('results/rand_mkspans.dill', 'wb') as handle:
    dill.dump(rand_mkspans, handle)
    
elapsed = timer() - start
print("Random DAG part took {} minutes".format(elapsed / 60))