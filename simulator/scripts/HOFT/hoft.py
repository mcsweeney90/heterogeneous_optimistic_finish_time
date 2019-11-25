#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:17:56 2019

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
from Static_heuristics import HEFT, OFT_priorities, HOFT

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

####################################################################################################

# Environments.

single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

####################################################################################################

"""Cholesky."""

#start = timer()
#heuristics = ["HEFT-WM", "HOFT"]
#n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]
#chol_speedup, chol_mkspans = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
#for nb in [128, 1024]:
#    for env in [single, multiple]:
#        with open("results/cholesky/{}_nb{}.txt".format(env.name, nb), "w") as dest:            
#            env.print_info(filepath=dest)
#            for nt in n_tasks:
#                dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))
#                dag.print_info(filepath=dest)
#                mst = dag.minimal_serial_time(platform=env)
#                print("Minimal serial time: {}\n".format(mst), file=dest)   
#                chol_mkspans[env.name][nb]["MST"].append(mst)
#                
#                OFT = dag.optimistic_finish_times()    
#
#                heft_mkspan = HEFT(dag, platform=env)                
#                chol_mkspans[env.name][nb]["HEFT"].append(heft_mkspan)
#                print("HEFT makespan: {}\n".format(heft_mkspan), file=dest)  
#                
#                for h in heuristics:
#                    if h == "HEFT-WM":
#                        task_list = dag.sort_by_upward_rank(platform=env, weighting="WM-I")
#                        mkspan = HEFT(dag, platform=env, task_list=task_list)
#                    elif h == "HOFT":
#                        task_list = OFT_priorities(dag, platform=env, selection="HOFT", table=OFT)
#                        mkspan = HOFT(dag, platform=env, policy="HOFT", task_list=task_list)
#                    chol_mkspans[env.name][nb][h].append(mkspan)  
#                    sp = 100 - (mkspan / heft_mkspan) * 100
#                    chol_speedup[env.name][nb][h].append(sp)
#                    print("{} makespan: {}".format(h, mkspan), file=dest)           
#                    print("Speedup (%) over HEFT: {}\n".format(sp), file=dest)                          
#                
#                print("--------------------------------------------------------\n", file=dest)                  
#    
## Save the makespans so can plot later if I want.
#with open('data/chol_mkspans.dill'.format(nb), 'wb') as handle:
#    dill.dump(chol_mkspans, handle)
#    
## Save the speedups so can plot later if I want.
#with open('data/chol_speedups.dill'.format(nb), 'wb') as handle:
#    dill.dump(chol_speedup, handle)
#    
#elapsed = timer() - start
#print("Cholesky part took {} minutes".format(elapsed / 60))

"""Plot the Cholesky speedups over HEFT."""

#with open('data/chol_speedups.dill', 'rb') as file:
#    chol_speedups = dill.load(file)
#n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]

## Paper.
#for nb in [128, 1024]:
#    fig = plt.figure(dpi=400)            
#    ax = fig.add_subplot(111, frameon=False)
#    # hide tick and tick label of the big axes
#    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#    ax.set_xlabel("Number of tasks", labelpad=10) 
#    ax.set_ylabel("Makespan reduction vs HEFT (%)", labelpad=15)        
#    
#    preferences = {"HEFT-WM" : [":", "o"], "HOFT" : ["--", "s"]}
#        
#    ax1 = fig.add_subplot(211)
#    plt.xscale('log')
#    for h in heuristics:
#        ax1.plot(n_tasks, chol_speedups["Single_GPU"][nb][h], linestyle=preferences[h][0], marker=preferences[h][1], label=h)
#    ax1.set_xticklabels([])
#    ax1.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
#    ax1.set_title("Single GPU", color="black", fontsize='large', family='serif')
#    
#    ax2 = fig.add_subplot(212)
#    plt.xscale('log')
#    for h in heuristics: 
#        ax2.plot(n_tasks, chol_speedups["Multiple_GPU"][nb][h], linestyle=preferences[h][0], marker=preferences[h][1], label=h)        
#    #ax2.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
#    ax2.set_title("Multiple GPU", color="black", fontsize='large', family='serif')
#    
#    plt.savefig('plots/hoft_speedup_cholesky_nb{}'.format(nb), bbox_inches='tight') 

## Talk.
#titles = {"Single_GPU" : "1 GPU, 1 CPU", "Multiple_GPU" : "4 GPUs, 4 CPUs"}
#for nb in [128, 1024]:
#    fig = plt.figure(dpi=400)            
#    ax = fig.add_subplot(111, frameon=False)
#    # hide tick and tick label of the big axes
#    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#    ax.set_xlabel("NUMBER OF TASKS", labelpad=10) 
#    ax.set_ylabel("MAKESPAN REDUCTION VS HEFT (%)", labelpad=15)        
#    
#    preferences = {"HEFT-WM" : [":", "o"], "HOFT" : ["--", "s"]}
#        
#    ax1 = fig.add_subplot(211)
#    plt.xscale('log')
#    ax1.plot(n_tasks, len(n_tasks) * [0], linestyle='-', label="HEFT", color='#FBC15E')
#    for h in heuristics:
#        ax1.plot(n_tasks, chol_speedups["Single_GPU"][nb][h], linestyle=preferences[h][0], marker=preferences[h][1], label=h)
#    ax1.set_xticklabels([])
#    ax1.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
#    ax1.set_title(titles["Single_GPU"], color="black", fontsize='large', family='serif')
#    
#    ax2 = fig.add_subplot(212)
#    plt.xscale('log')
#    ax2.plot(n_tasks, len(n_tasks) * [0], linestyle='-', label="HEFT", color='#FBC15E')
#    for h in heuristics: 
#        ax2.plot(n_tasks, chol_speedups["Multiple_GPU"][nb][h], linestyle=preferences[h][0], marker=preferences[h][1], label=h)        
#    #ax2.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
#    ax2.set_title(titles["Multiple_GPU"], color="black", fontsize='large', family='serif')
#    
#    plt.savefig('plots/hoft_speedup_cholesky_nb{}_TALK'.format(nb), bbox_inches='tight') 




"""Random."""
start = timer()
heuristics = ["HEFT-WM", "HOFT", "HOFT-WM"]
n_dags = 180
rand_mkspans = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
rand_speedups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
rand_ccrs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for env in [single, multiple]:
    env.print_info()
    for acc in ["low_acc", "high_acc"]:
        for ccr in ["0_10", "10_20", "20_50"]:
            makespans = defaultdict(list)
            best_occurences, failures = defaultdict(int), defaultdict(int)
            with open("results/random/{}/{}_CCR_{}.txt".format(env.name, acc, ccr), "w") as dest:            
                env.print_info(filepath=dest)
                count = 0
                for app in os.listdir('../../graphs/random/{}/{}/CCR_{}'.format(env.name, acc, ccr)):
                    count += 1
                    if count > 2: # REMEMBER TO REMOVE THIS BEFORE LEAVING TO RUN!
                        break
                    print("Starting DAG number {}...".format(count))
                    dag = nx.read_gpickle('../../graphs/random/{}/{}/CCR_{}/{}'.format(env.name, acc, ccr, app))
                    dag.print_info(platform=env, filepath=dest) 
                    rand_ccrs[env.name][acc][ccr].append(dag.CCR[env.name])
                    mst = dag.minimal_serial_time(env) 
                    rand_mkspans[env.name][acc][ccr]["MST"].append(mst)
                    
                    heft_mkspan = HEFT(dag, platform=env)                
                    rand_mkspans[env.name][acc][ccr]["HEFT"].append(heft_mkspan)
                    print("HEFT makespan: {}\n".format(heft_mkspan), file=dest)  
                                    
                    # Need to compute table for the OFT weightings.
                    OFT = dag.optimistic_finish_times() 
                    
                    best = float('inf')                
                    for h in heuristics:
                        if h == "HEFT-WM":
                            task_list = dag.sort_by_upward_rank(platform=env, weighting="WM-I")
                            mkspan = HEFT(dag, platform=env, task_list=task_list)
                        elif h == "HOFT":
                            task_list = OFT_priorities(dag, platform=env, selection="HOFT", table=OFT)
                            mkspan = HOFT(dag, platform=env, policy="HOFT", table=OFT, task_list=task_list)  
                        elif h == "HOFT-WM":
                            task_list = dag.sort_by_upward_rank(platform=env, weighting="WM-I")
                            mkspan = HOFT(dag, platform=env, policy="HOFT", table=OFT, task_list=task_list)  
                        rand_mkspans[env.name][acc][ccr][h].append(mkspan)
                        sp = 100 - (mkspan / heft_mkspan) * 100
                        rand_speedups[env.name][acc][ccr][h].append(sp)
                        print("{} makespan: {}".format(h, mkspan), file=dest)           
                        print("Speedup (%) over HEFT: {}\n".format(sp), file=dest) 
                        best = min(best, mkspan)                  
                    print("--------------------------------------------------------\n", file=dest)                
                    for h in ["HEFT"] + heuristics:
                        m = rand_mkspans[env.name][acc][ccr][h][-1]
                        if m == best:
                            best_occurences[h] += 1
                        if m > mst:
                            failures[h] += 1                     
                            
                print("--------------------------------------------------------", file=dest)
                print("SUMMARY", file=dest)
                print("--------------------------------------------------------", file=dest)            
                for h in ["HEFT"] + heuristics:
                    print("{}.".format(h), file=dest)
                    if h != "HEFT":
                        print("Average improvement compared to HEFT: {}%".format(np.mean(rand_speedups[env.name][acc][ccr][h])), file=dest)
                        print("Number of times better than HEFT: {}/{}".format(sum(1 for s in rand_speedups[env.name][acc][ccr][h] if s >= 1.0), n_dags), file=dest)               
                    print("Number of best occurences: {}/{}".format(best_occurences[h], n_dags), file=dest)
                    print("Number of failures: {}/{}\n".format(failures[h], n_dags), file=dest)          
                
elapsed = timer() - start
print("This took {} minutes".format(elapsed / 60))

# Save the makespans so can plot again later...
with open('data/rand_mkspans.dill', 'wb') as handle:
    dill.dump(rand_mkspans, handle)
    
# Save the speedups so can plot again later...
with open('data/rand_speedups.dill', 'wb') as handle:
    dill.dump(rand_speedups, handle)
    
# Save the CCRs so can plot again later...
with open('data/rand_ccrs.dill', 'wb') as handle:
    dill.dump(rand_ccrs, handle)
    
    
"""Plot the speedup over HEFT for the random DAGs."""

with open('data/rand_speedups.dill', 'rb') as file:
    rand_speedups = dill.load(file)
    
with open('data/rand_ccrs.dill', 'rb') as file:
    rand_ccrs = dill.load(file)
    
with open('data/rand_mkspans.dill', 'rb') as file:
    rand_mkspans = dill.load(file)

all_ccrs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
all_speedups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for env in [single, multiple]:
    for acc in ["low_acc", "high_acc"]:
        for ccr in ["0_10", "10_20", "20_50"]:
            for i in range(len(rand_mkspans[env.name][acc][ccr]["MST"])):
                m = rand_mkspans[env.name][acc][ccr]["HEFT"][i]
                for h in heuristics:
                    m1 = rand_mkspans[env.name][acc][ccr][h][i]
                    d1, d2 = rand_ccrs[env.name][acc][ccr][i], rand_speedups[env.name][acc][ccr][h][i]
                    all_ccrs[env.name][acc][h].append(d1)
                    all_speedups[env.name][acc][h].append(d2)
                    
                    
with open("results/random/summary.txt", "w") as dest: 
    for env in [single, multiple]:  
        print("Platform: {}".format(env.name.replace('_', ' ')), file=dest)                
        for acc in ["low_acc", "high_acc"]:
            print("\n{} ACCELERATION".format("LOW" if acc == "low_acc" else "HIGH"), file=dest)
            for h in heuristics:
                print("{} mean speedup: {}".format(h, np.mean(all_speedups[env.name][acc][h])), file=dest)
        print("\n\n\n\n\n", file=dest)      
                

#markers = {"HOFT" : '.', "HEFT-WM" : '.'}
#
#fig = plt.figure(dpi=400)            
#ax = fig.add_subplot(111, frameon=False)
## hide tick and tick label of the big axes
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#ax.set_xlabel("CCR", labelpad=10) 
#ax.set_ylabel("Makespan reduction vs HEFT (%)", labelpad=15)    
#
#ax1 = fig.add_subplot(211)
##plt.xscale('log')
#ax1.plot(valid_ccrs["Single_GPU"]["HEFT-WM"], len(valid_ccrs["Single_GPU"]["HEFT-WM"]) * [0], linestyle='-', color='#FBC15E')
#for h in heuristics:
#    ax1.scatter(valid_ccrs["Single_GPU"][h], valid_speedups["Single_GPU"][h], marker=markers[h], label=h)
#    next(ax1._get_lines.prop_cycler) 
#ax1.set_xticklabels([])
#ax1.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
#ax1.set_title("Single GPU", color="black", fontsize='large', family='serif')
#
#ax2 = fig.add_subplot(212)
##plt.xscale('log')
#ax2.plot(valid_ccrs["Single_GPU"]["HEFT-WM"], len(valid_ccrs["Single_GPU"]["HEFT-WM"]) * [0], linestyle='-', color='#FBC15E')
#for h in heuristics:
#    ax2.scatter(valid_ccrs["Multiple_GPU"][h], valid_speedups["Multiple_GPU"][h], marker=markers[h], label=h)
#    next(ax2._get_lines.prop_cycler) 
##ax2.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
#ax2.set_title("Multiple GPU", color="black", fontsize='large', family='serif')
#
#plt.savefig('plots/random_speedups', bbox_inches='tight')     



