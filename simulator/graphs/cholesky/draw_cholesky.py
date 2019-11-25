#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:05:19 2019

@author: Tom
"""

import networkx as nx

# Choose tile size nb and number of tile nt. (Tile size not relevant for drawing unless we want to add weight labels.)
nb, nt = 128, 35
# Load the DAG.
cholesky = nx.read_gpickle('real/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))
# Draw the DAG.
cholesky.draw_graph(filepath="images")
