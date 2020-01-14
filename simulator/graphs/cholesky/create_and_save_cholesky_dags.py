#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:15:20 2019

Create and store Cholesky DAGs with computation and communication costs given by sample means of real timings on CSF3 node. 

@author: Tom
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from timeit import default_timer as timer
# Quick fix to let us import modules from main directory...
import sys
sys.path.append('../../') 
import Environment    
from Graph import Task, DAG  

def cholesky(num_tiles, draw=False):
    """
    Returns a DAG object representing a tiled Cholesky factorization.
    """
    
    last_acted_on = {} # Useful for keeping track of things...
    
    G = nx.DiGraph()
    
    for k in range(num_tiles): # Grow the DAG column by column.     
        
        node1 = Task(task_type="POTRF")
        if k == 0:
            node1.entry = True
        else:
            # Find the task which last acted on the tile.            
            for node in G: 
                if last_acted_on[(k, k)] == node:
                    G.add_edge(node, node1)
                    break     
        last_acted_on[(k, k)] = node1                                    
        
        for i in range(k + 1, num_tiles):
            node2 = Task(task_type="TRSM")
            G.add_edge(node1, node2)
            try:
                for node in G:
                    if last_acted_on[(i, k)] == node:
                        G.add_edge(node, node2)
                        break
            except KeyError:
                pass
            last_acted_on[(i, k)] = node2            
            
        for i in range(k + 1, num_tiles): 
            node3 = Task(task_type="SYRK")
            try:
                for node in G:
                    if last_acted_on[(i, i)] == node:
                        G.add_edge(node, node3)
                        break
            except KeyError:
                pass
            last_acted_on[(i, i)] = node3
            
            try:
                for node in G:
                    if last_acted_on[(i, k)] == node:
                        G.add_edge(node, node3)
                        break
            except KeyError:
                pass
                
            for j in range(k + 1, i):               
                node4 = Task(task_type="GEMM") 
                try:
                    for node in G:
                        if last_acted_on[(i, j)] == node:
                            G.add_edge(node, node4)
                            break
                except KeyError:
                    pass
                last_acted_on[(i, j)] = node4
                
                try:
                    for node in G:
                        if last_acted_on[(i, k)] == node:
                            G.add_edge(node, node4)
                            break
                except KeyError:
                    pass
                
                try:
                    for node in G:
                        if last_acted_on[(j, k)] == node:
                            G.add_edge(node, node4)
                            break
                except KeyError:
                    pass   
                
    # Create the DAG object. 
    dag = DAG(app="Cholesky")
    dag.DAG = G
    n = 0
    for t in dag.DAG:
        # Set the exit tasks.
        if not list(dag.DAG.successors(t)):
            t.exit = True
        t.ID = n # Give each task an ID number.
        n += 1    
    dag.num_tasks = n   # Number of tasks in DAG, often useful. 
    dag.num_edges = dag.DAG.number_of_edges()     
    max_edges = (n * (n - 1)) / 2 # Maximum number of edges for DAG with n vertices.
    dag.edge_density = dag.num_edges / max_edges  
    if draw:
        dag.draw_graph()    
    return dag 

"""Load the real timing data."""

min_tile = 32
max_tile = 1024
inc_factor = 2
iterations = 1001
sizes = [32, 64, 128, 256, 512, 1024]
cpu_data = defaultdict(lambda: defaultdict(list))
gpu_data = defaultdict(lambda: defaultdict(list))
comm_data = defaultdict(lambda: defaultdict(list))


# Load CPU data.
name = "skylake"
dgemm_raw = np.genfromtxt('data/{}/DGEMM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dpotrf_raw = np.genfromtxt('data/{}/DPOTRF_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dsyrk_raw = np.genfromtxt('data/{}/DSYRK_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dtrsm_raw = np.genfromtxt('data/{}/DTRSM_{}.csv'.format(name, name), delimiter=',', skip_header=1)

x = min_tile
i = 1 # Ignore the first iteration for each tile size - more expensive than later iterations because of libraries being loaded, etc.
while x < max_tile + 1:
    cpu_data["GEMM"][x] = [y[2] for y in dgemm_raw[i: i + iterations - 1]]
    cpu_data["POTRF"][x] = [y[2] for y in dpotrf_raw[i: i + iterations - 1]]
    cpu_data["SYRK"][x]= [y[2] for y in dsyrk_raw[i: i + iterations - 1]]
    cpu_data["TRSM"][x] = [y[2] for y in dtrsm_raw[i: i + iterations - 1]]            
    i += iterations
    x *= inc_factor        

# Load GPU data.
name = "V100"
dgemm_raw = np.genfromtxt('data/{}/DGEMM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dpotrf_raw = np.genfromtxt('data/{}/DPOTRF_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dsyrk_raw = np.genfromtxt('data/{}/DSYRK_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dtrsm_raw = np.genfromtxt('data/{}/DTRSM_{}.csv'.format(name, name), delimiter=',', skip_header=1)

x = min_tile
i = 1 
while x < max_tile + 1:
    gpu_data["GEMM"][x] = [y[2] for y in dgemm_raw[i: i + iterations - 1]]
    gpu_data["POTRF"][x] = [y[2] for y in dpotrf_raw[i: i + iterations - 1]]
    gpu_data["SYRK"][x]= [y[2] for y in dsyrk_raw[i: i + iterations - 1]]
    gpu_data["TRSM"][x] = [y[2] for y in dtrsm_raw[i: i + iterations - 1]]  
    comm_data["GEMM"][x] = [y[3] - y[2] for y in dgemm_raw[i: i + iterations - 1]]  
    comm_data["POTRF"][x] = [y[3] - y[2] for y in dpotrf_raw[i: i + iterations - 1]]
    comm_data["SYRK"][x] = [y[3] - y[2] for y in dsyrk_raw[i: i + iterations - 1]]
    comm_data["TRSM"][x] = [y[3] - y[2] for y in dtrsm_raw[i: i + iterations - 1]]   
    i += iterations
    x *= inc_factor
    
"""Create the Cholesky DAGs."""

# Set up the target environments (needed for computing CCR).  
single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

# Create and save the actual DAG objects.
for nt in range(5, 51, 5):
    print(nt)
    # Construct DAG topology.
    dag = cholesky(num_tiles=nt)
    for s in sizes:
        for task in dag.DAG:
            # Set computation and communication costs.
            task.CPU_time = np.mean(cpu_data[task.type][s])
            task.GPU_time = np.mean(gpu_data[task.type][s])
            task.acceleration_ratio = task.CPU_time / task.GPU_time
            cc_comm_costs, cg_comm_costs, gc_comm_costs, gg_comm_costs = {}, {}, {}, {}
            for child in dag.DAG.successors(task):
                cc_comm_costs[child.ID] = 0
                x = np.mean(comm_data[child.type][s])
                cg_comm_costs[child.ID] = x
                gc_comm_costs[child.ID] = x
                gg_comm_costs[child.ID] = x
            task.comm_costs["CC"] = cc_comm_costs
            task.comm_costs["CG"] = cg_comm_costs
            task.comm_costs["GC"] = gc_comm_costs
            task.comm_costs["GG"] = gg_comm_costs
        # Compute and save the CCR for both platforms.
        dag.app = "Cholesky (nb {})".format(s)
        for env in [single, multiple]:
            dag.compute_CCR(platform=env)
        # Save DAG.
        nx.write_gpickle(dag, 'nb{}/{}tasks.gpickle'.format(s, dag.num_tasks))
        
# Save summaries of all Cholesky DAGs.
start = timer()
for nb in [32, 64, 128, 256, 512, 1024]:
    print(nb)
    with open("summaries/nb{}.txt".format(nb), "w") as dest: 
        for nt in [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]:
            dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))          
            dag.print_info(filepath=dest, platform=[single, multiple])
elapsed = timer() - start     
print("Time taken to compute and save DAG info: {} minutes.".format(elapsed / 60)) 

"""Draw small DAGs to get an idea of the topology."""

# # Choose tile size nb and number of tiles nt. (Tile size not relevant for drawing unless labels are defined.)
# nb, nt = 128, 220
# # Load the DAG.
# cholesky = nx.read_gpickle('nb{}/{}tasks.gpickle'.format(nb, nt))
# # Draw the DAG.
# cholesky.draw_graph(filepath="images")