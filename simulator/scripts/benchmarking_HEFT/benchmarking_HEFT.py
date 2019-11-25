#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:10:55 2019

@author: Tom
"""

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
from collections import defaultdict 
from timeit import default_timer as timer
import dill
import sys
sys.path.append('../../') # Quick fix to let us import modules from main directory. 
import Environment    # Node classes and functions.
import Graph    # DAG classes and functions.
from Static_heuristics import HEFT


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
plt.rcParams['axes.titlepad'] = 5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 12

####################################################################################################

# Environments.

single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

####################################################################################################

##########################################

# A. Speedup.

##########################################

"""Cholesky."""
#start = timer()
#n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]
#chol_speedups = defaultdict(lambda: defaultdict(list))
#chol_mkspans = defaultdict(lambda: defaultdict(list))
#for env in [single, multiple]:
#    env.print_info()
#    for nb in [128, 1024]:
#        with open("data/cholesky_{}_nb{}.txt".format(env.name, nb), "w") as dest:            
#            env.print_info(filepath=dest)
#            for nt in n_tasks:
#                dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))
#                dag.print_info(filepath=dest)
#                mst = dag.minimal_serial_time(platform=env)
#                print("Minimal serial time: {}".format(mst), file=dest)     
#                
#                # Standard HEFT.
#                heft_mkspan = HEFT(dag, platform=env)
#                chol_mkspans[env.name][nb].append(heft_mkspan)
#                print("Standard HEFT makespan: {}".format(heft_mkspan), file=dest)
#                s = mst / heft_mkspan
#                print("Speedup: {}".format(s), file=dest)
#                chol_speedups[env.name][nb].append(s)                 
#                print("--------------------------------------------------------\n", file=dest)                  
#                
## Plot the speedups.
#preferences = {"Single_GPU" : ["-", "o"], "Multiple_GPU": ["--", "s"]}
#for nb in [128, 1024]:               
#    fig1 = plt.figure(dpi=400) 
#    ax1 = fig1.add_subplot(111)
#    ax1.set_xlabel("Number of tasks", labelpad=10) 
#    ax1.set_ylabel("Speedup", labelpad=10)  
#    for env in ["Single_GPU", "Multiple_GPU"]:
#        ax1.plot(n_tasks, chol_speedups[env][nb], linestyle=preferences[env][0], marker=preferences[env][1], label="{}".format(env.replace('_', ' ')))
#    ax1.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
#    #ax1.set_title("Cholesky (nb = {})".format(nb), color="black", fontsize='large', family='serif')    
#    plt.savefig('plots/cholesky_nb{}_speedups'.format(nb), bbox_inches='tight') 
#    
## Save the speedups so can plot again later if I want...
#with open('data/cholesky_speedups.dill', 'wb') as handle:
#    dill.dump(chol_speedups, handle)
#    
#with open('data/cholesky_mkspans.dill', 'wb') as handle:
#    dill.dump(chol_mkspans, handle)
#    
#elapsed = timer() - start
#print("This took {} minutes".format(elapsed / 60))

"""Random."""

#all_speedups = defaultdict(lambda: defaultdict(list))
#for env in [single, multiple]:
#    env.print_info()
#    for ccr in ["0_1", "1_5", "5_100"]:
#        speedups, failures = [], 0
#        with open("data/random_{}_CCR_{}.txt".format(env.name, ccr), "w") as dest:            
#            env.print_info(filepath=dest)
#            count = 0
#            for app in os.listdir('../../graphs/random/{}/CCR_{}'.format(env.name, ccr)):
#                count += 1
##                if count > 3:
##                    break
#                print("Starting DAG number {}...".format(count))
#                dag = nx.read_gpickle('../../graphs/random/{}/CCR_{}/{}'.format(env.name, ccr, app))
#                dag.print_info(platform=env, filepath=dest)  
#                mst = dag.minimal_serial_time(platform=env)       
#                print("Minimal serial time: {}".format(mst), file=dest) 
#                
#                # HEFT.
#                heft_mkspan = HEFT(dag, platform=env)
#                if heft_mkspan > mst:
#                    failures += 1
#                print("HEFT makespan: {}".format(heft_mkspan), file=dest)                
#                s = mst / heft_mkspan
#                print("Speedup: {}".format(s), file=dest)
#                speedups.append(s)
#                all_speedups[env.name][dag.CCR[env.name]].append(s)                                                
#                print("--------------------------------------------------------\n", file=dest)                
#                
#            print("--------------------------------------------------------", file=dest)
#            print("SUMMARY", file=dest)
#            print("--------------------------------------------------------", file=dest)  
#            print("Avg. speedup: {}.".format(np.mean(speedups)), file=dest)
#            print("Best speedup: {}.".format(max(speedups)), file=dest)
#            print("Worst speedup: {}.".format(min(speedups)), file=dest)            
#            print("Number of HEFT failures: {}/180".format(failures), file=dest)
#
#markers = {"Single_GPU" : 'o', "Multiple_GPU" : 's'}
#fig1 = plt.figure(dpi=400) 
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel("Speedup", labelpad=10)  
#ax1.set_xlabel("CCR", labelpad=10) 
#for env in ["Single_GPU", "Multiple_GPU"]:
#    ax1.scatter(all_speedups[env].keys(), all_speedups[env].values(), marker=markers[env], label="{}".format(env.replace('_', ' ')))
#    next(ax1._get_lines.prop_cycler)                
#ax1.legend(handletextpad=0.1, loc='best', fancybox=True)
##ax1.set_title("Random", color="black", fontsize='large', family='serif')    
#plt.savefig('plots/random_speedups', bbox_inches='tight') 
#
### Save the speedups so can plot again later if I want...
#with open('data/rand_speedups.dill', 'wb') as handle:
#    dill.dump(all_speedups, handle)

"""Playing with different plotting styles."""

# Cholesky.
nb = 128
with open('data/cholesky_speedups.dill'.format(nb), 'rb') as file:
    chol_speedups = dill.load(file)
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]

preferences = {"Single_GPU" : ["-", "o"], "Multiple_GPU": ["-", "o"]}
for nb in [128, 1024]:               
    fig1 = plt.figure(dpi=400) 
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Number of tasks", labelpad=10) 
    ax1.set_ylabel("Speedup", labelpad=10)  
    plt.xscale('log')
    for env in ["Single_GPU", "Multiple_GPU"]:
        ax1.plot(n_tasks, chol_speedups[env][nb], linestyle=preferences[env][0], marker=preferences[env][1], label="{}".format(env.replace('_', ' ')))
    ax1.set_ylim(bottom=0)
    plt.yticks(np.arange(0, 9, 1.0)) # Bit of a hack to make it look nicer, should really calculate the 0 and 9 values.
    ax1.legend(handlelength=1.8, handletextpad=0.4, loc='best', fancybox=True)
    #ax1.set_title("Cholesky (nb = {})".format(nb), color="black", fontsize='large', family='serif')    
    plt.savefig('plots/cholesky_nb{}_speedups'.format(nb), bbox_inches='tight') 
    
    
## Random.
    
#with open('data/rand_speedups.dill', 'rb') as file:
#    all_speedups = dill.load(file)
#markers = {"Single_GPU" : '.', "Multiple_GPU" : '.'}
#fig1 = plt.figure(dpi=400) 
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel("Speedup", labelpad=10)  
#ax1.set_xlabel("CCR", labelpad=10) 
#plt.xscale('log')
#for env in ["Single_GPU", "Multiple_GPU"]:
#    ax1.scatter(all_speedups[env].keys(), all_speedups[env].values(), marker=markers[env], label="{}".format(env.replace('_', ' ')))
#    next(ax1._get_lines.prop_cycler)                
##ax1.legend(handletextpad=0.1, loc='best', fancybox=True) 
#plt.savefig('plots/random_speedups', bbox_inches='tight') 