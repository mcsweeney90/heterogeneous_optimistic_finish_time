#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:50:14 2019

Functions for constructing NLA application DAGs.

@author: Tom
"""

import networkx as nx
import sys
sys.path.append('../../') # Quick fix to let us import modules from main directory.   
from Graph import Task, DAG    

def cholesky(num_tiles, draw=False):
    """
    Returns a DAG object representing a tiled Cholesky factorization.
    TODO: Ugly code with lots of duplication, tidy up. 
    """
    
    last_acted_on = {} # Useful for keeping track of 
    
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


def QR(num_tiles, draw=False):
    """
    Returns a DAG object representing a tiled QR factorization of a matrix.
    TODO. 
    """            
    return 

        
        
        
        