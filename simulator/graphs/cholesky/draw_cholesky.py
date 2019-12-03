#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:05:19 2019

Very short script for loading Cholesky DAGs, drawing the corresponding graphs and saving them. 

@author: Tom
"""

import networkx as nx
import sys
sys.path.append('../../') # Needed to avoid ModuleNotFoundError. TODO: investigate if we can remove this. 

# Choose tile size nb and number of tiles nt. (Tile size not relevant for drawing unless labels are defined.)
nb, nt = 128, 220
# Load the DAG.
cholesky = nx.read_gpickle('nb{}/{}tasks.gpickle'.format(nb, nt))
# Draw the DAG.
cholesky.draw_graph(filepath="images")
