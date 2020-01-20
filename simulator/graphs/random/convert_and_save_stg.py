#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read .stg files, construct and save corresponding DAG topologies (without computation and communication costs).
"""

import os
import networkx as nx
import re
from timeit import default_timer as timer
import sys
sys.path.append('../../') # Quick fix to let us import modules from main directory.   
from Graph import Task, DAG

start = timer()
# Read stg files.
for orig in os.listdir('STG/'):    
    if orig.endswith('.' + 'stg'):  
        print(orig)
        dag = DAG(app=orig.split('.')[0])        
        with open("STG/{}".format(orig)) as f:
            next(f) # Skip first line.            
            for row in f:
                if row[0] == "#":
                    if row[6] == "x" and row[9] == "P":
                        dag.max_task_predecessors = int(row.split(":")[1].strip())
                        continue
                    elif row[4] == "A" and row[9] == "P":
                        info = row.split(":")
                        if len(info) == 2:
                            dag.avg_task_predecessors = float(info[1].strip())
                        elif len(info) == 3:                            
                            dag.avg_task_predecessors = float(info[2].split(")")[0].strip())                        
                        break                    
                    continue
                # Remove all whitespace - there is probably a nicer way to do this...
                info = " ".join(re.split("\s+", row, flags=re.UNICODE)).strip().split() 
                # Create task.        
                nd = Task()
                nd.ID = int(info[0])
                if info[2] == '0':
                    nd.entry = True
                    dag.DAG.add_node(nd)
                    continue
                if nd.ID == 1001: # Assumes only one exit task, which is true for the STG but may not be in general!
                    nd.exit = True
                # Add connections to predecessors.
                predecessors = list(n for n in dag.DAG if str(n.ID) in info[3:])
                for p in predecessors:
                    dag.DAG.add_edge(p, nd)
        dag.num_tasks = len(dag.DAG) # Number of tasks in DAG, often useful.
        dag.num_edges = dag.DAG.number_of_edges()
        max_edges = (dag.num_tasks * (dag.num_tasks - 1)) / 2 # Maximum number of edges for DAG with n vertices.
        dag.edge_density = dag.num_edges / max_edges
        dag.print_info()
        
        # Save DAG topology.
        nx.write_gpickle(dag, "topologies/{}.gpickle".format(dag.app))
        
elapsed = timer() - start     
print("Time taken: {} seconds".format(elapsed))   
            
            
            
        