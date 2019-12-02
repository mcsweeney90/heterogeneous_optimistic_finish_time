#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:15:20 2019

@author: Tom
"""

import os
import numpy as np
import networkx as nx
from collections import defaultdict
from NLA_DAGs import cholesky
from timeit import default_timer as timer
import sys
sys.path.append('../../') # Quick fix to let us import modules from main directory.
import Environment    # Cluster classes and functions.

# Set up the environments - needed for computing CCR.  
single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

#min_tile = 32
#max_tile = 1024
#inc_factor = 2
#iterations = 1001
#sizes = [32, 64, 128, 256, 512, 1024]
#cpu_data = defaultdict(lambda: defaultdict(list))
#gpu_data = defaultdict(lambda: defaultdict(list))
#comm_data = defaultdict(lambda: defaultdict(list))
#
#
## Load CPU data.
#name = "skylake"
#dgemm_raw = np.genfromtxt('data/{}/DGEMM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
#dpotrf_raw = np.genfromtxt('data/{}/DPOTRF_{}.csv'.format(name, name), delimiter=',', skip_header=1)
#dsyrk_raw = np.genfromtxt('data/{}/DSYRK_{}.csv'.format(name, name), delimiter=',', skip_header=1)
#dtrsm_raw = np.genfromtxt('data/{}/DTRSM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
#
#x = min_tile
#i = 1 # Ignore the first iteration for each tile size - more expensive than later iterations because of libraries being loaded, etc.
#while x < max_tile + 1:
#    cpu_data["GEMM"][x] = [y[2] for y in dgemm_raw[i: i + iterations - 1]]
#    cpu_data["POTRF"][x] = [y[2] for y in dpotrf_raw[i: i + iterations - 1]]
#    cpu_data["SYRK"][x]= [y[2] for y in dsyrk_raw[i: i + iterations - 1]]
#    cpu_data["TRSM"][x] = [y[2] for y in dtrsm_raw[i: i + iterations - 1]]            
#    i += iterations
#    x *= inc_factor    
#    
#
## Load GPU data.
#name = "V100"
#dgemm_raw = np.genfromtxt('data/{}/DGEMM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
#dpotrf_raw = np.genfromtxt('data/{}/DPOTRF_{}.csv'.format(name, name), delimiter=',', skip_header=1)
#dsyrk_raw = np.genfromtxt('data/{}/DSYRK_{}.csv'.format(name, name), delimiter=',', skip_header=1)
#dtrsm_raw = np.genfromtxt('data/{}/DTRSM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
#
#x = min_tile
#i = 1 # Ignore the first iteration for each tile size - more expensive than later iterations because of libraries being loaded, etc.
#while x < max_tile + 1:
#    gpu_data["GEMM"][x] = [y[2] for y in dgemm_raw[i: i + iterations - 1]]
#    gpu_data["POTRF"][x] = [y[2] for y in dpotrf_raw[i: i + iterations - 1]]
#    gpu_data["SYRK"][x]= [y[2] for y in dsyrk_raw[i: i + iterations - 1]]
#    gpu_data["TRSM"][x] = [y[2] for y in dtrsm_raw[i: i + iterations - 1]]  
#    comm_data["GEMM"][x] = [y[3] - y[2] for y in dgemm_raw[i: i + iterations - 1]]  
#    comm_data["POTRF"][x] = [y[3] - y[2] for y in dpotrf_raw[i: i + iterations - 1]]
#    comm_data["SYRK"][x] = [y[3] - y[2] for y in dsyrk_raw[i: i + iterations - 1]]
#    comm_data["TRSM"][x] = [y[3] - y[2] for y in dtrsm_raw[i: i + iterations - 1]]   
#    i += iterations
#    x *= inc_factor
#
#for nt in range(5, 51, 5):
#    print(nt)
#    dag = cholesky(num_tiles=nt)
#    for s in sizes:
#        for task in dag.DAG:
#            task.CPU_time = np.mean(cpu_data[task.type][s])
#            task.GPU_time = np.mean(gpu_data[task.type][s])
#            task.acceleration_ratio = task.CPU_time / task.GPU_time
#            cc_comm_costs, cg_comm_costs, gc_comm_costs, gg_comm_costs = {}, {}, {}, {}
#            for child in dag.DAG.successors(task):
#                cc_comm_costs[child.ID] = 0
#                x = np.mean(comm_data[child.type][s])
#                cg_comm_costs[child.ID] = x
#                gc_comm_costs[child.ID] = x
#                gg_comm_costs[child.ID] = x
#            task.comm_costs["CC"] = cc_comm_costs
#            task.comm_costs["CG"] = cg_comm_costs
#            task.comm_costs["GC"] = gc_comm_costs
#            task.comm_costs["GG"] = gg_comm_costs
#        # Compute and save the CCR for both platforms.
#        dag.app = "Cholesky (nb {})".format(s)
#        for env in [single, multiple]:
#            dag.compute_CCR(platform=env)
#        nx.write_gpickle(dag, 'nb{}/{}tasks.gpickle'.format(s, dag.num_tasks))
        
"""Print summaries of all Cholesky DAGs."""

start = timer()
for env in [single, multiple]:  
    for nb in [32, 64, 128, 256, 512, 1024]:
        print(nb)
        with open("summaries/{}/nb{}.txt".format(env.name, nb), "w") as dest:            
            env.print_info(filepath=dest)
            for nt in [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]:
                dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))          
                dag.print_info(filepath=dest, platform=env)

elapsed = timer() - start     
print("Time taken to compute and save DAG info: {} minutes.".format(elapsed / 60)) 