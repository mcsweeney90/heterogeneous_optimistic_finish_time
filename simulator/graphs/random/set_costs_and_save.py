#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:52:19 2019

Read DAG objects from topologies folder, set computation and communication costs and save.

@author: Tom
"""

import os
import sys
import networkx as nx
import numpy as np
from timeit import default_timer as timer
sys.path.append('../../') # Quick fix to let us import modules from main directory.
import Environment    # Node classes and functions.

# Set up the environments - needed to ensure CCR is correct for the target platform.  
single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

# Set DAG costs and save.
start = timer()
for orig in os.listdir('topologies/'):       
    print(orig)
    for env in [single, multiple]:  
        for acc in ["low_acc", "high_acc"]:
            s = 5 if acc == "low_acc" else 50 # Acceleration values based on BLAS/LAPACK kernels with nb = 128 or 1024.
            for comm in [(0, 10), (10, 20), (20, 50)]: # CCR intervals chosen based on range for Cholesky DAGs but extended to cover other applications.
                dag = nx.read_gpickle('topologies/{}'.format(orig))  
                dag.set_costs(platform=env, target_ccr=np.random.uniform(comm[0], comm[1]), ratio_dist=("gamma", s))     
                nx.write_gpickle(dag, '{}/{}/CCR_{}_{}/{}.gpickle'.format(env.name, acc, comm[0], comm[1], dag.app))         
print("\n")
elapsed = timer() - start     
print("Time taken to read DAG topologies and set costs: {} minutes.".format(elapsed / 60)) 

# Check and fix any DAGs with CCR in wrong interval. 
start = timer()
for env in [single, multiple]:
    print(env.name)
    for acc in ["low_acc", "high_acc"]:
        print(acc)
        for comm in ["CCR_0_10", "CCR_10_20", "CCR_20_50"]:
            print(comm)
            for app in os.listdir('{}/{}/{}/'.format(env.name, acc, comm)):
                dag = nx.read_gpickle('{}/{}/{}/{}'.format(env.name, acc, comm, app)) 
                if comm == "CCR_0_10":
                    l, u = 0, 10
                    while dag.CCR[env.name] < l or dag.CCR[env.name] > u:
                        print(app)
                        dag.set_costs(platform=env, target_ccr=np.random.uniform(0, 10))   
                        print("New CCR: {}".format(dag.CCR[env.name]))
                        nx.write_gpickle(dag, '{}/{}/{}/{}.gpickle'.format(env.name, acc, comm, dag.app)) 
                elif comm == "CCR_10_20":   
                    l, u = 10, 20
                    while dag.CCR[env.name] < l or dag.CCR[env.name] > u:
                        print(app)
                        dag.set_costs(platform=env, target_ccr=np.random.uniform(10, 20))   
                        print("New CCR: {}".format(dag.CCR[env.name]))
                        nx.write_gpickle(dag, '{}/{}/{}/{}.gpickle'.format(env.name, acc, comm, dag.app))  
                else:
                    l, u = 20, 50
                    while dag.CCR[env.name] < l or dag.CCR[env.name] > u:
                        print(app)
                        dag.set_costs(platform=env, target_ccr=np.random.uniform(20, 50))   
                        print("New CCR: {}".format(dag.CCR[env.name]))
                        nx.write_gpickle(dag, '{}/{}/{}/{}.gpickle'.format(env.name, acc, comm, dag.app)) 
print("Time taken to check and correct DAG CCRs: {} minutes.".format(elapsed / 60))                
  
# Save summaries of DAG info.          
start = timer()
for env in [single, multiple]:
    for acc in ["low_acc", "high_acc"]:
        for comm in [(0, 10), (10, 20), (20, 50)]:
            with open("{}/{}/summaries/CCR_{}_{}.txt".format(env.name, acc, comm[0], comm[1]), "w") as dest:            
                env.print_info(filepath=dest)
                for app in os.listdir('{}/{}/CCR_{}_{}/'.format(env.name, acc, comm[0], comm[1])):
                    print(app)
                    dag = nx.read_gpickle('{}/{}/CCR_{}_{}/{}'.format(env.name, acc, comm[0], comm[1], app))            
                    dag.print_info(filepath=dest, platform=env)
elapsed = timer() - start     
print("Time taken to compute and save DAG info: {} minutes.".format(elapsed / 60))         
