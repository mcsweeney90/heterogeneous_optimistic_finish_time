#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling-based lookahead for low CCR/high-data DAGs. 
Referred to in the conclusion of paper but this was not promising so we didn't pursue it very far.  

Estimated runtime: ~5 hours on a machine with an Intel i7.
"""

import os
import networkx as nx
import numpy as np
import dill
from collections import defaultdict 
from timeit import default_timer as timer
import sys
sys.path.append('../../') 
import Environment    
from Heuristics import HEFT, HEFT_L

####################################################################################################

# Define environments to be considered.
single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

####################################################################################################
            
start = timer()

with open('../HOFT/results/rand_mkspans.dill', 'rb') as file:
    rand_mkspans = dill.load(file)

# Just "1P" option shown here since this is quite slow but results for others are similar.
samplings = ["1P"] 
n_dags = 180
rand_speedups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for env in [single, multiple]:
    env.print_info()
    for acc in ["low_acc", "high_acc"]:
        heft_failures = 0
        successes, ties, failures = defaultdict(int), defaultdict(int), defaultdict(int)
        with open("results/{}_{}.txt".format(env.name, acc), "w") as dest:            
            env.print_info(filepath=dest)
            count = 0
            for app in os.listdir('../../graphs/random/{}/{}/CCR_0_10'.format(env.name, acc)):
                count += 1
                print("Starting DAG number {}...".format(count))
                dag = nx.read_gpickle('../../graphs/random/{}/{}/CCR_0_10/{}'.format(env.name, acc, app))
                dag.print_info(platform=env, filepath=dest)  
                mst = dag.minimal_serial_time(env) 
                
                heft_mkspan = HEFT(dag, platform=env)
                print("HEFT makespan: {}".format(heft_mkspan), file=dest)   
                if heft_mkspan > mst:
                    heft_failures += 1
                    
                for h in samplings:
                    mkspan = HEFT_L(dag, platform=env, weighted_average=True, child_sampling_policy=h)
                    print("HEFT-L with {} child sampling policy makespan: {}".format(h, mkspan), file=dest) 
                    rand_mkspans[env.name][acc]["0_10"][h].append(mkspan) 
                    rand_speedups[env.name][acc][h].append(100 - (mkspan / heft_mkspan) * 100)                     
                    if mkspan > mst:
                        failures[h] += 1
                    elif mkspan == mst:
                        ties[h] += 1
                    elif mkspan < mst:
                        successes[h] += 1  
                print("--------------------------------------------------------\n", file=dest)    
            print("--------------------------------------------------------", file=dest)
            print("SUMMARY", file=dest)
            print("--------------------------------------------------------", file=dest) 
            print("HEFT failures: {}\n".format(heft_failures), file=dest)
            for h in samplings:
                print("Sampling policy: {}.".format(h), file=dest)
                print("Average improvement compared to HEFT: {}%".format(np.mean(rand_speedups[env.name][acc][h])), file=dest)
                print("Number of times better than HEFT: {}/{}".format(sum(1 for s in rand_speedups[env.name][acc][h] if s >= 1.0), n_dags), file=dest) 
                print("Number of successes: {}/{}".format(successes[h], n_dags), file=dest) 
                print("Number of ties: {}/{}".format(ties[h], n_dags), file=dest) 
                print("Number of failures: {}/{}\n".format(failures[h], n_dags), file=dest)  
                    
# Save speedups and makespans so can use later.
with open('data/rand_speedups.dill', 'wb') as handle:
    dill.dump(rand_speedups, handle)    
with open('data/rand_mkspans.dill', 'wb') as handle:
    dill.dump(rand_mkspans, handle)
    
elapsed = timer() - start
print("This took {} minutes.".format(elapsed / 60))


#######################################################################

"""Print summary of failure information."""

#######################################################################

heuristics = ["HEFT-WM", "HOFT", "HOFT-WM", "1P"]   
# Load data if necessary.
try:
    rand_mkspans
except NameError:
    with open('results/rand_mkspans.dill', 'rb') as file:
        rand_mkspans = dill.load(file)

with open("results/summary.txt", "w") as dest:     
    for env in [single, multiple]:
        env.print_info(filepath=dest)
        for acc in ["low_acc", "high_acc"]:
            print("{} ACCELERATION\n".format("LOW" if acc == "low_acc" else "HIGH"), file=dest)
            
            heft_failures = set()
            for i, hft in enumerate(rand_mkspans[env.name][acc]["0_10"]["HEFT"]):
                if rand_mkspans[env.name][acc]["0_10"]["MST"][i] < rand_mkspans[env.name][acc]["0_10"]["HEFT"][i]:
                    heft_failures.add(i)        
            print("Number of HEFT failures: {}\n".format(len(heft_failures)), file=dest)
            
            failures = defaultdict(int)
            heft_corrections, new_failures = defaultdict(int), defaultdict(int)        
            
            for i, mst in enumerate(rand_mkspans[env.name][acc]["0_10"]["MST"]):
                for h in heuristics:
                    if mst < rand_mkspans[env.name][acc]["0_10"][h][i]:
                        failures[h] += 1
                        if i not in heft_failures:
                            new_failures[h] += 1
                    else:
                        if i in heft_failures:
                            heft_corrections[h] += 1
            for h in heuristics:
                print("Heuristic: {}".format(h), file=dest)
                print("Number of failures: {}".format(failures[h]), file=dest)
                print("Number of HEFT corrections: {}".format(heft_corrections[h]), file=dest)
                print("Number of new failures: {}\n".format(new_failures[h]), file=dest)                
            print("--------------------------------------------------------", file=dest)

