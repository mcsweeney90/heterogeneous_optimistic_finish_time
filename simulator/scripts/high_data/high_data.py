#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:49:29 2019

19/11: Modify this according to changes in the CCR intervals we consider.

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
from Static_heuristics import HEFT, HEFT_L

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
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['figure.titlesize'] = 12

####################################################################################################

# Environments.

single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

####################################################################################################

"""Sampling based lookahead, single child with the highest priority."""

start = timer()
samplings = ["1P"]
n_dags = 180
rand_mkspans, rand_speedups = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
for env in [multiple]:
    env.print_info()
    for ccr in ["0_1"]:
        heft_failures = 0
        successes, ties, failures = defaultdict(int), defaultdict(int), defaultdict(int)
        with open("data/{}_random_CCR_{}.txt".format(env.name, ccr), "w") as dest:            
            env.print_info(filepath=dest)
            count = 0
            for app in os.listdir('../../graphs/random/{}/CCR_{}'.format(env.name, ccr)):
                count += 1
#                if count > 1: # REMEMBER TO REMOVE THIS BEFORE LEAVING TO RUN!
#                    break
                print("Starting DAG number {}...".format(count))
                dag = nx.read_gpickle('../../graphs/random/{}/CCR_{}/{}'.format(env.name, ccr, app))
                dag.print_info(platform=env, filepath=dest)  
                mst = dag.minimal_serial_time(env) 
                rand_mkspans[env.name]["MST"].append(mst)
                
                heft_mkspan = HEFT(dag, platform=env)
                rand_mkspans[env.name]["HEFT"].append(heft_mkspan)
                print("HEFT makespan: {}".format(heft_mkspan), file=dest)  
                if heft_mkspan > mst:
                    heft_failures += 1
                    
                for h in samplings:
                    mkspan = HEFT_L(dag, platform=env, weighted_average=True, child_sampling_policy=h)
                    print("HEFT-L with {} child sampling policy makespan: {}".format(h, mkspan), file=dest) 
                    rand_mkspans[env.name][h].append(mkspan)                                  
                                     
                print("--------------------------------------------------------\n", file=dest)                
                for h in samplings:
                    m = rand_mkspans[env.name][h][-1]
                    rand_speedups[env.name][h].append(100 - (m / heft_mkspan) * 100) 
                    
                    if m > mst:
                        failures[h] += 1
                    elif m == mst:
                        ties[h] += 1
                    elif m < mst:
                        successes[h] += 1
                        
            print("--------------------------------------------------------", file=dest)
            print("SUMMARY", file=dest)
            print("--------------------------------------------------------", file=dest) 
            print("HEFT failures: {}\n".format(heft_failures), file=dest)
            for h in samplings:
                print("Sampling policy: {}.".format(h), file=dest)
                print("Average improvement compared to HEFT: {}%".format(np.mean(rand_speedups[env.name][h])), file=dest)
                print("Number of times better than HEFT: {}/{}".format(sum(1 for s in rand_speedups[env.name][h] if s >= 1.0), n_dags), file=dest) 
                print("Number of successes: {}/{}".format(successes[h], n_dags), file=dest) 
                print("Number of ties: {}/{}".format(ties[h], n_dags), file=dest) 
                print("Number of failures: {}/{}\n".format(failures[h], n_dags), file=dest)  
                    
## Save the speedups so can plot later...
with open('data/speedups_random.dill', 'wb') as handle:
    dill.dump(rand_speedups, handle)
    
with open('data/makespans_random.dill', 'wb') as handle:
    dill.dump(rand_mkspans, handle)
    
elapsed = timer() - start
print("This took {} minutes.".format(elapsed / 60))


"""Relationship between MST/CP (called MSLR here) and speedup. Short answer: there isn't one, this wasn't useful."""

#start = timer()
#n_dags = 180
#mslrs = defaultdict(list)
#for env in [single, multiple]:
#    env.print_info()
#    for ccr in ["0_1"]:
#        heft_failures, heft_ties, heft_wins = 0, 0, 0
#        with open("data/{}_oft_check.txt".format(env.name), "w") as dest:            
#            env.print_info(filepath=dest)
#            count = 0
#            for app in os.listdir('../../graphs/random/{}/CCR_{}'.format(env.name, ccr)):
#                count += 1
##                if count > 10: # REMEMBER TO REMOVE THIS BEFORE LEAVING TO RUN!
##                    break
#                print("Starting DAG number {}...".format(count))
#                dag = nx.read_gpickle('../../graphs/random/{}/CCR_{}/{}'.format(env.name, ccr, app))
#                dag.print_info(platform=env, filepath=dest)  
#                mst = dag.minimal_serial_time(env) 
#                
#                OFT = dag.optimistic_finish_times() 
#                cp = max(min(OFT[task][p] for p in OFT[task]) for task in OFT if task.exit) 
#                print("Critical path: {}".format(cp), file=dest)
#                mslr = mst / cp  
#                
#                heft_mkspan = HEFT(dag, platform=env)
#                print("HEFT makespan: {}".format(heft_mkspan), file=dest)  
#                if heft_mkspan > mst:
#                    heft_failures += 1
#                elif heft_mkspan == mst:
#                    heft_ties += 1
#                else:
#                    heft_wins += 1      
#
#                mslrs[env.name].append((mslr, mst/heft_mkspan))                                             
#                                     
#                print("--------------------------------------------------------\n", file=dest)                
#                
#                        
#            print("--------------------------------------------------------", file=dest)
#            print("SUMMARY", file=dest)
#            print("--------------------------------------------------------", file=dest) 
#            print("HEFT failures: {}".format(heft_failures), file=dest)
#            print("HEFT ties: {}".format(heft_ties), file=dest)
#            print("HEFT wins: {}\n".format(heft_wins), file=dest)            
#                    
#    
#elapsed = timer() - start
#print("This took {} minutes.".format(elapsed / 60))
#
#with open('data/mslrs.dill', 'wb') as handle:
#    dill.dump(mslrs, handle)
    
# Now plot the relationship between mslr and speedup.

#with open('data/mslrs.dill', 'rb') as file:
#    mslr_data = dill.load(file)
#
#mslrs, speedups = defaultdict(list), defaultdict(list)
#for env in [single, multiple]:
#    for m, s in mslr_data[env.name]:
#        mslrs[env.name].append(m)
#        speedups[env.name].append(s)
#
#fig = plt.figure(dpi=400)            
#ax = fig.add_subplot(111, frameon=False)
## hide tick and tick label of the big axes
#plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#ax.set_xlabel("MSLR", labelpad=10) 
#ax.set_ylabel("Speedup", labelpad=15)        
#
#preferences = {"HEFT" : [":", "o"], "HEFT-WM" : ["--", "s"], "HEFT-OFT" : ["-.", "v"]}
#    
#ax1 = fig.add_subplot(211)
#ax1.scatter(mslrs["Single_GPU"], speedups["Single_GPU"], marker='.')
#ax1.set_xticklabels([])
#ax1.set_title("Single GPU", color="black", fontsize='large', family='serif')
#
#ax2 = fig.add_subplot(212)
#ax2.scatter(mslrs["Multiple_GPU"], speedups["Multiple_GPU"], marker='.')  
#ax2.set_title("Multiple GPU", color="black", fontsize='large', family='serif')
#
#plt.savefig('plots/mslr_vs_speedup', bbox_inches='tight') 
        
