#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:50:57 2018

Framework for describing a DAG. 

@author: Tom, Neil Walton
"""

import numpy as np
import networkx as nx
from copy import deepcopy
from networkx.drawing.nx_agraph import to_agraph
from networkx.drawing.nx_pydot import read_dot
from statistics import median
from collections import defaultdict

class Task:
    """
    Task class for representing nodes in a task dependency DAG.
    """ 
        
    def __init__(self, task_type=None):
        """
        Task attributes.
        """             
         
        self.type = task_type   # String identifying the type of the task, e.g., "GEMM". 
        self.ID = None    # Useful for printing and debugging, etc
        self.entry = False   # True if entry task.
        self.exit = False    # True if exit task. 
        
        # The following attributes are costs set by a platform.
        self.CPU_time = 0  # Execution time on CPU workers.
        self.GPU_time = 0  # Execution time on GPU workers.
        self.acceleration_ratio = 0 # CPU_time / GPU_time.
        self.comm_costs = defaultdict(dict) # Nested dict, e.g., comm_costs["CC"][child.ID] == CPU-CPU cost from parent to child.
        self.comm_costs["CC"], self.comm_costs["CG"], self.comm_costs["GC"], self.comm_costs["GG"] = {}, {}, {}, {} 
        
        # Set once task has been scheduled.
        self.AST = 0   # Actual start time.
        self.AFT = 0   # Actual finish time.
        self.scheduled = False   # True if task has been scheduled somewhere.
        self.where_scheduled = None   # ID of processor task is scheduled on, sometimes useful.                   
    
    def reset(self):
        """Resets some attributes to defaults so execution of the task can be simulated again. """
        self.AFT = 0
        self.AST = 0   
        self.scheduled = False
        self.where_scheduled = None           
        
    def approximate_execution_cost(self, platform, weighting="HEFT", r_bar=None):
        """ 
        Average execution cost of a task across all processors in a platform. Used in e.g., HEFT.
        Assumes only CPUs and GPUs and the cost is the same on all processors of each kind. 
        """
        # Let user set custom weights. Not really pursued in any depth so far.
        
        if weighting == "HEFT" or weighting == "mean" or weighting == "MEAN" or weighting == "M":
            return (self.CPU_time * platform.n_CPUs + self.GPU_time * platform.n_GPUs) / platform.n_workers
        elif weighting == "median" or weighting == "MEDIAN":
            execution_costs = [self.CPU_time for _ in range(platform.n_CPUs)] + [self.GPU_time for _ in range(platform.n_GPUs)]
            return median(execution_costs)
        elif weighting == "worst" or weighting == "W" or weighting == "simple worst" or weighting == "SW":
            return max(self.CPU_time, self.GPU_time)
        elif weighting == "best" or weighting == "B" or weighting == "simple best" or weighting == "sb":
            return min(self.CPU_time, self.GPU_time)   
        elif weighting == "HEFT-WM" or weighting == "WM-I":
            r = self.acceleration_ratio
            return (self.CPU_time * platform.n_CPUs + r * self.GPU_time * platform.n_GPUs) / (platform.n_CPUs + r * platform.n_GPUs) 
        elif weighting == "WM-II":
            if not r_bar:
                raise ValueError("Tried to use approximate_execution_cost with weighting == WM-II but haven't entered the mean acceleration ratio!")
            r = r_bar
            return (self.CPU_time * platform.n_CPUs + r * self.GPU_time * platform.n_GPUs) / (platform.n_CPUs + r * platform.n_GPUs)  
        
        elif weighting == "PS" or weighting == "ps": # "PS" == "processor speedup".
            fastest, slowest = min(self.CPU_time, self.GPU_time), max(self.CPU_time, self.GPU_time)
            if not fastest:
                return 0
            return slowest / fastest
        elif weighting == "D":
            fastest, slowest = min(self.CPU_time, self.GPU_time), max(self.CPU_time, self.GPU_time)
            return slowest - fastest
        elif weighting == "sfb" or weighting == "SFB":
            fastest, slowest = min(self.CPU_time, self.GPU_time), max(self.CPU_time, self.GPU_time)
            if not fastest:
                return 0
            return (slowest - fastest) / (slowest / fastest) 
        
        elif weighting == "EW-I":
            if not r_bar:
                raise ValueError("Tried to use approximate_execution_cost with weighting == EW-I but haven't entered the mean acceleration ratio!")
            r = r_bar # Alternatively, use mean child acceleration ratio or each child's acc ratio.
            w = platform.n_CPUs + r * platform.n_GPUs
            # Compute the effective CPU time.
            eff_cpu_cost = self.CPU_time
            eff_cpu_cost += (platform.n_CPUs  - 1) / w * sum(self.comm_costs["CC"].values())
            eff_cpu_cost += (r * platform.n_GPUs / w) * sum(self.comm_costs["CG"].values()) 
            # Compute the effective GPU time.
            eff_gpu_cost = self.GPU_time
            eff_gpu_cost += (platform.n_CPUs / w) * sum(self.comm_costs["GC"].values())
            eff_gpu_cost += (r * (platform.n_GPUs - 1) / w) * sum(self.comm_costs["GG"].values())               
            return (eff_cpu_cost * platform.n_CPUs + r * eff_gpu_cost * platform.n_GPUs) / w
        
        elif weighting == "EW-II":
            r = self.acceleration_ratio 
            w = platform.n_CPUs + r * platform.n_GPUs
            # Compute the effective CPU time.
            eff_cpu_cost = self.CPU_time
            eff_cpu_cost += (platform.n_CPUs  - 1) / w * sum(self.comm_costs["CC"].values())
            eff_cpu_cost += (r * platform.n_GPUs / w) * sum(self.comm_costs["CG"].values()) 
            # Compute the effective GPU time.
            eff_gpu_cost = self.GPU_time
            eff_gpu_cost += (platform.n_CPUs / w) * sum(self.comm_costs["GC"].values())
            eff_gpu_cost += (r * (platform.n_GPUs - 1) / w) * sum(self.comm_costs["GG"].values())               
            return (eff_cpu_cost * platform.n_CPUs + r * eff_gpu_cost * platform.n_GPUs) / w
        
        elif weighting == "EPS-I":
            if not r_bar:
                raise ValueError("Tried to use approximate_execution_cost with weighting == EPS-I but haven't entered the mean acceleration ratio!")
            r = r_bar # Alternatively could use mean acceleration ratio of the children, or each child's acc. ratio.
            w = platform.n_CPUs + r * platform.n_GPUs
            # Compute the effective CPU time.
            eff_cpu_cost = self.CPU_time
            eff_cpu_cost += (platform.n_CPUs  - 1) / w * sum(self.comm_costs["CC"].values())
            eff_cpu_cost += (r * platform.n_GPUs / w) * sum(self.comm_costs["CG"].values()) 
            # Compute the effective GPU time.
            eff_gpu_cost = self.GPU_time
            eff_gpu_cost += (platform.n_CPUs / w) * sum(self.comm_costs["GC"].values())
            eff_gpu_cost += (r * (platform.n_GPUs - 1) / w) * sum(self.comm_costs["GG"].values())              
            fastest, slowest = min(eff_cpu_cost, eff_gpu_cost), max(eff_cpu_cost, eff_gpu_cost)
            if not fastest:
                return 0
            return slowest / fastest  
        
        elif weighting == "EPS-II":
            r = self.acceleration_ratio 
            w = platform.n_CPUs + r * platform.n_GPUs
            # Compute the effective CPU time.
            eff_cpu_cost = self.CPU_time
            eff_cpu_cost += (platform.n_CPUs  - 1) / w * sum(self.comm_costs["CC"].values())
            eff_cpu_cost += (r * platform.n_GPUs / w) * sum(self.comm_costs["CG"].values()) 
            # Compute the effective GPU time.
            eff_gpu_cost = self.GPU_time
            eff_gpu_cost += (platform.n_CPUs / w) * sum(self.comm_costs["GC"].values())
            eff_gpu_cost += (r * (platform.n_GPUs - 1) / w) * sum(self.comm_costs["GG"].values())              
            fastest, slowest = min(eff_cpu_cost, eff_gpu_cost), max(eff_cpu_cost, eff_gpu_cost)
            if not fastest:
                return 0
            return slowest / fastest  
        
        elif weighting == "CC-I":
            if self.exit:
                return 0
            cc_mean = np.mean(list(self.comm_costs["CC"].values()))
            cg_mean = np.mean(list(self.comm_costs["CG"].values()))
            gc_mean = np.mean(list(self.comm_costs["GC"].values()))
            gg_mean = np.mean(list(self.comm_costs["GG"].values()))            
            A = platform.n_CPUs * (platform.n_CPUs - 1)
            B = platform.n_CPUs * platform.n_GPUs
            C = platform.n_GPUs * (platform.n_GPUs - 1)            
            return (A * cc_mean + B * (cg_mean + gc_mean) + C * gg_mean) / platform.n_workers**2           
        elif weighting == "CC-II":
            if self.exit:
                return 0
            return max(max(self.comm_costs["CC"].values()), max(self.comm_costs["CG"].values()), 
                       max(self.comm_costs["GC"].values()), max(self.comm_costs["GG"].values()))
        elif weighting == "CC-III":
            if self.exit:
                return 0
            return min(min(self.comm_costs["CC"].values()), min(self.comm_costs["CG"].values()), 
                       min(self.comm_costs["GC"].values()), min(self.comm_costs["GG"].values()))     
            
        raise ValueError('No weighting, e.g., "mean" or "median", specified for approximate_execution_cost.')   

    def ready_to_schedule(self, dag):
        """
        Returns True if the task is ready to schedule (i.e., all dependencies of the task have been satisfied or it is an entry task), 
        False otherwise (including if it has already actually been scheduled).
        """
        if self.scheduled:
            return False  # Already scheduled.
        if self.entry:  # Entry task.
            return True
        for p in dag.DAG.predecessors(self):
            if not p.scheduled:
                return False
        return True

class DAG:
    """
    A collection of tasks with weighted edges representing dependencies between them and weights the data amounts to be transferred.    
    """
    def __init__(self, app="Random"): 
        """
        - app is a string describing the application whose task graph the DAG object represents.
        """
        self.app = app 
        self.DAG = nx.DiGraph()
        self.num_tasks = 0  
        
        # These next are useful for e.g., I/O, debugging, etc.
        self.max_task_predecessors = None
        self.avg_task_predecessors = None
        self.num_edges = None
        self.edge_density = None # Num. edges / max possible edges.
        self.CCR = {} # {Platform.name : CCR}.

    def compute_topological_info(self):
        """Computes topological info. Assumes not known already."""  
        
        if self.max_task_predecessors is None and self.avg_task_predecessors is None:
            num_predecessors = list(len(list(self.DAG.predecessors(t))) for t in self.DAG)
            self.max_task_predecessors = max(num_predecessors)
            self.avg_task_predecessors = np.mean(num_predecessors)
        elif self.max_task_predecessors is None:
            num_predecessors = list(len(list(self.DAG.predecessors(t))) for t in self.DAG)
            self.max_task_predecessors = max(num_predecessors)
        elif self.avg_task_predecessors is None:
            num_predecessors = list(len(list(self.DAG.predecessors(t))) for t in self.DAG)
            self.avg_task_predecessors = np.mean(num_predecessors)
        if self.num_edges is None:
            self.num_edges = self.DAG.number_of_edges()
        if self.edge_density is None:
            max_edges = (self.num_tasks * (self.num_tasks - 1)) / 2 # If dummy entry and exit nodes should disregard these so assume this is not the case.
            self.edge_density = self.num_edges / max_edges         
            
    def compute_CCR(self, platform):
        """Computes and sets the CCR for platform."""
        cpu_times = list(task.CPU_time for task in self.DAG)
        gpu_times = list(task.GPU_time for task in self.DAG)
        mean_compute = sum(cpu_times) * platform.n_CPUs + sum(gpu_times) * platform.n_GPUs
        mean_compute /= platform.n_workers
        
        cc_comm, cg_comm, gc_comm, gg_comm = 0, 0, 0, 0
        for task in self.DAG:
            cc_comm += (sum(task.comm_costs["CC"].values()))
            cg_comm += (sum(task.comm_costs["CG"].values()))
            gc_comm += (sum(task.comm_costs["GC"].values()))
            gg_comm += (sum(task.comm_costs["GG"].values()))
           
        mean_comm = platform.n_CPUs * (platform.n_CPUs - 1) * cc_comm
        mean_comm += platform.n_CPUs * platform.n_GPUs * (cg_comm + gc_comm)
        mean_comm += platform.n_GPUs * (platform.n_GPUs - 1) * gg_comm
        mean_comm /= (platform.n_workers**2)
        
        self.CCR[platform.name] = mean_compute / mean_comm         
        

    def print_info(self, platform=None, detailed=False, filepath=None):
        """ Prints all the information about the DAG."""
        # Print to screen.
        print("--------------------------------------------------------")
        print("DAG INFO")
        print("--------------------------------------------------------")   
        print("Application: {}".format(self.app))
        print("Number of tasks: {}".format(self.num_tasks))
        self.compute_topological_info()
        print("Maximum number of task predecessors: {}".format(self.max_task_predecessors))
        print("Average number of task predecessors: {}".format(self.avg_task_predecessors))
        print("Number of edges: {}".format(self.num_edges))
        print("Edge density: {}".format(self.edge_density))
        
        if platform:
            cpu_times = list(task.CPU_time for task in self.DAG)
            gpu_times = list(task.GPU_time for task in self.DAG)
            acc_ratios = list(task.acceleration_ratio for task in self.DAG)
            cpu_mu, cpu_sigma = np.mean(cpu_times), np.std(cpu_times)
            print("Mean task CPU time: {}, standard deviation: {}".format(cpu_mu, cpu_sigma))
            gpu_mu, gpu_sigma = np.mean(gpu_times), np.std(gpu_times)
            print("Mean task GPU time: {}, standard deviation: {}".format(gpu_mu, gpu_sigma))
            task_mu = (platform.n_GPUs * gpu_mu + platform.n_CPUs * cpu_mu) / platform.n_workers
            print("Mean task execution time: {}".format(task_mu))
            acc_mu, acc_sigma = np.mean(acc_ratios), np.std(acc_ratios)
            print("Mean task acceleration ratio: {}, standard deviation: {}".format(acc_mu, acc_sigma)) 
            
            # Compute the CCR if not already set.
            try:
                ccr = self.CCR[platform.name]
            except KeyError:                 
                ccr = self.compute_CCR(platform)
            print("Computation-to-communication ratio: {}".format(ccr))
            
            mst = self.minimal_serial_time(platform)
            print("Minimal serial time: {}".format(mst))
                        
            if detailed:
                print("\nTASK INFO:")
                for task in self.DAG:
                    print("\nTask ID: {}".format(task.ID))
                    if task.entry:
                        print("Entry task.")
                    if task.exit:
                        print("Exit task.")
                    if task.type:
                        print("Task type: {}".format(task.type)) 
                    print("CPU time: {}".format(task.CPU_time))
                    print("GPU time: {}".format(task.GPU_time))
                    print("Acceleration ratio: {}".format(task.acceleration_ratio)) 
                    for child, time in task.comm_costs.items():
                        print("Child task ID: {}, comm cost (when nonzero): {}".format(child, time))               
        print("--------------------------------------------------------") 
        
        # If filepath, also print to file.
        if filepath:
            print("--------------------------------------------------------", file=filepath)
            print("DAG INFO", file=filepath)
            print("--------------------------------------------------------", file=filepath)   
            print("Application: {}".format(self.app), file=filepath)
            print("Number of tasks: {}".format(self.num_tasks), file=filepath)
            print("Maximum number of task predecessors: {}".format(self.max_task_predecessors), file=filepath)
            print("Average number of task predecessors: {}".format(self.avg_task_predecessors), file=filepath)
            print("Number of edges: {}".format(self.num_edges), file=filepath)
            print("Edge density: {}".format(self.edge_density), file=filepath)
            if platform: 
                print("Mean task CPU time: {}, standard deviation: {}".format(cpu_mu, cpu_sigma), file=filepath)
                print("Mean task GPU time: {}, standard deviation: {}".format(gpu_mu, gpu_sigma), file=filepath)
                print("Mean task acceleration ratio: {}, standard deviation: {}".format(acc_mu, acc_sigma), file=filepath)
                print("Mean task execution time: {}".format(task_mu), file=filepath)
                print("Computation-to-communication ratio: {}".format(ccr), file=filepath)
                print("Minimal serial time: {}".format(mst), file=filepath)
                if detailed:
                    print("\nTASK INFO:", file=filepath)
                    for task in self.DAG:
                        print("\nTask ID: {}".format(task.ID), file=filepath)
                        if task.entry:
                            print("Entry task.", file=filepath)
                        if task.exit:
                            print("Exit task.", file=filepath)
                        if task.type:
                            print("Task type: {}".format(task.type), file=filepath)
                        print("CPU time: {}".format(task.CPU_time), file=filepath)
                        print("GPU time: {}".format(task.GPU_time), file=filepath)
                        for child, time in task.comm_costs.items():
                            print("Child task ID: {}, comm cost (when nonzero): {}".format(child, time), file=filepath)                   
            print("--------------------------------------------------------", file=filepath) 
            
        
    def draw_graph(self, filepath="graphs/images", verbose=False):
        """
        Draws the graph and saves it as a png.        
        See https://stackoverflow.com/questions/39657395/how-to-draw-properly-networkx-graphs
        """          
        G = deepcopy(self.DAG)        
        G.graph['graph'] = {'rankdir':'TD'}  
        G.graph['node']={'shape':'circle', 'color':'#348ABD', 'style':'filled', 'fillcolor':'#E5E5E5', 'penwidth':'3.0'}
        G.graph['edges']={'arrowsize':'4.0', 'penwidth':'5.0'}       
        A = to_agraph(G)
        
        # Add identifying colors if task types are known.
        for task in G:
            if task.type == "GEMM":
                n = A.get_node(task)  
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#E24A33'
                n.attr['label'] = 'G'
            elif task.type == "POTRF":
                n = A.get_node(task)   
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#348ABD'
                n.attr['label'] = 'P'
            elif task.type == "SYRK":
                n = A.get_node(task)   
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#988ED5'
                n.attr['label'] = 'S'
            elif task.type == "TRSM":
                n = A.get_node(task)    
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#FBC15E'
                n.attr['label'] = 'T'       
        
        if verbose:
            print("The DOT file used to generate the graph looks like:")
            print(A)
        A.layout('dot')
        A.draw('{}/{}_{}tasks_DAG.png'.format(filepath, self.app.split(" ")[0], self.num_tasks))     # May need to change destination depending on current directory.        
            
    def set_costs(self, platform, target_ccr, history=None, ratio_dist=("gamma", 40.0)):
        """
        Sets computation and communication costs for randomly generated DAGs (e.g., from the STG).
        Platform is required to ensure that CCR is roughly correct. Note this that the actual CCR will only be approximately
        the input desired CCR because of stochasticity in how we define communication costs, so might need to clean up afterwards. 
        """  
        
        if self.num_edges is None:
            self.num_edges = self.DAG.number_of_edges()
        
        # Set the execution costs on CPU and GPU.
        for task in self.DAG:
            task.GPU_time = np.random.randint(1, 100) 
            if task.acceleration_ratio == 0:               
                try:
                    task.acceleration_ratio = history[task.type] 
                except:
                    if ratio_dist[0] == "gamma":
                        task.acceleration_ratio = np.random.gamma(shape=1.0, scale=ratio_dist[1])                    
            task.CPU_time = task.GPU_time * task.acceleration_ratio
        
        # Set the corresponding communication costs.
        
        # Compute the expected compute.
        cpu_times = list(task.CPU_time for task in self.DAG)
        gpu_times = list(task.GPU_time for task in self.DAG)
        expected_compute = sum(cpu_times) * platform.n_CPUs + sum(gpu_times) * platform.n_GPUs
        expected_compute /= platform.n_workers
        
        # Calculate the expected communication in the entire DAG - i.e., for all edges.        
        expected_comm = expected_compute / target_ccr
        # Compute the sum of all CPU-GPU comm costs, for all tasks. 
        total_cg_comm = expected_comm * platform.n_workers**2   
        total_cg_comm /= (2 * platform.n_CPUs * platform.n_GPUs + platform.n_GPUs * (platform.n_GPUs - 1))
        # Assumption here is that CG, GC, GG communications are roughly similar magnitudes so gc_comm and gg_comm are approx the same.
        
        comm_per_task = total_cg_comm / self.num_tasks
        
        for task in self.DAG:
            if task.exit:               
                continue
            # Calculate the number of task children.
            n_children = len(list(self.DAG.successors(task))) 
            comm_per_child = comm_per_task / n_children
            cc_comm_costs, cg_comm_costs, gc_comm_costs, gg_comm_costs = {}, {}, {}, {}
            for child in self.DAG.successors(task):
                cc_comm_costs[child.ID] = 0
                cg_comm_costs[child.ID] = np.random.gamma(shape=1.0, scale=comm_per_child)
                gc_comm_costs[child.ID] = np.random.gamma(shape=1.0, scale=comm_per_child)
                gg_comm_costs[child.ID] = np.random.gamma(shape=1.0, scale=comm_per_child)
            task.comm_costs["CC"] = cc_comm_costs
            task.comm_costs["CG"] = cg_comm_costs
            task.comm_costs["GC"] = gc_comm_costs
            task.comm_costs["GG"] = gg_comm_costs
        
        # Compute and save the actual CCR.
        mean_compute = sum(cpu_times) * platform.n_CPUs + sum(gpu_times) * platform.n_GPUs
        mean_compute /= platform.n_workers
        
        cc_comm, cg_comm, gc_comm, gg_comm = 0, 0, 0, 0
        for task in self.DAG:
            cc_comm += (sum(task.comm_costs["CC"].values()))
            cg_comm += (sum(task.comm_costs["CG"].values()))
            gc_comm += (sum(task.comm_costs["GC"].values()))
            gg_comm += (sum(task.comm_costs["GG"].values()))
           
        mean_comm = platform.n_CPUs * (platform.n_CPUs - 1) * cc_comm
        mean_comm += platform.n_CPUs * platform.n_GPUs * (cg_comm + gc_comm)
        mean_comm += platform.n_GPUs * (platform.n_GPUs - 1) * gg_comm
        mean_comm /= (platform.n_workers**2)
        
        self.CCR[platform.name] = mean_compute / mean_comm 
                                                            
                
    def all_tasks_scheduled(self):
        """Check if all the tasks in the DAG have been scheduled. Returns True if they have, False if not."""
        return all(task.scheduled for task in self.DAG)
    
    def reset(self):
        """ Resets some attributes to defaults so execution of the DAG object can be simulated again. """
        for task in self.DAG:
            task.reset()
            
    def makespan(self, partial=False):
        """Computes the makespan of the DAG."""
        if not partial:
            if not self.all_tasks_scheduled():
                raise ValueError('Error! There are tasks in the DAG which are not scheduled yet!')
            return max(t.AFT for t in self.DAG if t.exit)  
        return max(t.AFT for t in self.DAG)    
                   
    def sort_by_upward_rank(self, platform, weighting="mean", return_rank_values=False, verbose=False):
        """
        Returns a list of all tasks in the DAG sorted in nonincreasing order of upward rank.
        Note "upward rank" also sometimes called "bottom-level".
        """        
        
        # Traverse the DAG starting from the exit task.
        backward_traversal = list(reversed(list(nx.topological_sort(self.DAG))))
        
        # Compute the upward rank of all tasks recursively, starting with the exit task.
        task_ranks = {}
        for t in backward_traversal:
            task_ranks[t] = t.approximate_execution_cost(platform, weighting=weighting) 
            try:
                task_ranks[t] += max(platform.approximate_comm_cost(parent=t, child=s, weighting=weighting) + task_ranks[s] for s in self.DAG.successors(t))
            except ValueError:
                pass  
        
        priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))
            
        # If verbose, print the tasks in order of upward rank.
        if verbose:
            priority_list_ids = list(t.ID for t in priority_list)
            print("The priority list is: {}".format(priority_list_ids))
            
        # Return the tasks sorted in nonincreasing order of upward rank. 
        if return_rank_values:
            return priority_list, task_ranks
        return priority_list  
    
    def sort_by_downward_rank(self, platform, weighting="mean", return_rank_values=False, verbose=False):
        """
        Returns a list of all tasks in the DAG sorted in nondecreasing order of downward rank.
        Note "downward rank" also sometimes called "top-level".
        """
        
        # Traverse the DAG starting from the exit task.
        forward_traversal = list(nx.topological_sort(self.DAG))
        
        # Compute the downward rank of all tasks, starting with the entry task.
        task_ranks = {}
        for t in forward_traversal:
            task_ranks[t] = 0
            try:
                task_ranks[t] += max(p.approximate_execution_cost(platform, weighting) + platform.approximate_comm_cost(parent=p, child=t, weighting=weighting) +
                          task_ranks[p] for p in self.DAG.predecessors(t))
            except ValueError:
                pass  
        
        priority_list = list(sorted(task_ranks, key=task_ranks.get))
            
        # If verbose, print the tasks in order of upward rank.
        if verbose:
            priority_list_ids = list(t.ID for t in priority_list)
            print("The priority list is: {}".format(priority_list_ids))
            
        # Return the tasks sorted in nonincreasing order of upward rank. 
        if return_rank_values:
            return priority_list, task_ranks
        return priority_list 

    def optimistic_cost_table(self, platform):
        """Used in PEFT heuristic."""
        # Compute the OCT table.
        OCT = defaultdict(lambda: defaultdict(float))  
        # Traverse the DAG starting from the exit task(s).
        backward_traversal = list(reversed(list(nx.topological_sort(self.DAG))))
        for task in backward_traversal:
            if task.exit:
                for p in range(platform.n_workers):
                    OCT[task][p] = 0
                continue
            # Not an exit task...
            for p in range(platform.n_workers):
                child_values = []
                for child in self.DAG.successors(task):
                    # Calculate OCT(child, pw) + w(child, pw) for all processors pw.
                    proc_values = list(OCT[child][pw] + child.CPU_time if pw < platform.n_CPUs else OCT[child][pw] + child.GPU_time for pw in range(platform.n_workers))
                    # Add the (approximate) communication cost to the processor value unless pw == p.
                    for pw in range(platform.n_workers):
                        if pw != p: 
                            proc_values[pw] += platform.approximate_comm_cost(task, child)
                    # Calculate the minimum value over all processors.
                    child_values.append(min(proc_values))
                # OCT is the maximum of these processor minimums over all the child tasks.
                OCT[task][p] = max(child_values)  
        return OCT       
        
    def optimistic_finish_times(self):
        """
        Compute the optimistic finish time (i.e., the earliest possible time a task can be completed, assuming infinitely 
        many processors) for all tasks on all processors.
        """
        
        d = defaultdict(int)
        d["CC"], d["GG"] = 0, 0
        d["CG"], d["GC"]  = 1, 1     
        
        # Compute OFT for all tasks and processors. 
        OFT = defaultdict(lambda: defaultdict(float))            
        forward_traversal = list(nx.topological_sort(self.DAG))
        for task in forward_traversal:
            for p in ["C", "G"]:
                OFT[task][p] = task.CPU_time if p == "C" else task.GPU_time 
                if not task.entry:
                    parent_values = []
                    for parent in self.DAG.predecessors(task):
                        action_values = [OFT[parent][q] + d["{}".format(q + p)] * parent.comm_costs["{}".format(q + p)][task.ID] for q in ["C", "G"]]
                        parent_values.append(min(action_values))
                    OFT[task][p] += max(parent_values)   
        return OFT     
    
    def critical_path(self, verbose=False):
        """
        A lower bound on the makespan of the DAG on platform. Assume that every task can be executed immediately on the kind of processor
        that minimizes its execution time. Originally used nx.all_simple_paths but that was ~30 times slower.       
        """            
        
        OFT = self.optimistic_finish_times()
        cp = max(min(OFT[task][processor] for processor in OFT[task]) for task in OFT if task.exit)          
        if verbose:
            print("Length of critical path: {}".format(cp))                            
        return cp    
   
    def minimal_serial_time(self, platform):
        """
        Returns the minimum execution time of the DAG on the platform in serial (i.e., on a single processor). 
        Assumes platform has at least one CPU and GPU and only one kind of each.
        """        
        return min(sum(task.CPU_time for task in self.DAG), sum(task.GPU_time for task in self.DAG))
    
    def get_ready_tasks(self):
        """
        Return all the ready tasks. Slow and not usually recommended.
        """       
        return list(t for t in filter(lambda t: t.ready_to_schedule(self) == True, self.DAG))    
        

####################################################################################################    
# Functions to create DAG objects from dot files and Networkx DiGraphs.    
####################################################################################################        
        
def convert_from_dot(dot_path, app=None, draw=False):
    """ 
    Creates a DAG object from a dot file. 
    TODO: this is really slow for large DAGs (> 1000) so ideally need to find a way to optimize. 
    Unfortunately the bulk of the time seems to be taken by read_dot itself...
    """
        
    graph = nx.DiGraph(read_dot(dot_path))
    # Check if it's actually a DAG and make the graph directed if it isn't already.
    if graph.is_directed():
        G = graph
    else:
        G = nx.DiGraph()
        G.name = graph.name
        G.add_nodes_from(graph)    
        done = set() 
        for u, v in graph.edges():
            if (v, u) not in done:
                G.add_edge(u, v)
                done.add((u, v))   
        G.graph = deepcopy(graph.graph)
        G.node = deepcopy(graph.node)        
    # Look for cycles.
    try:
        nx.topological_sort(G)
    except nx.NetworkXUnfeasible:
        raise ValueError('Input graph in convert_from_dot has at least one cycle so is not a DAG!')    
    
    # Get app name from the filename (if not input).
    if not app:
        filename = dot_path.split('/')[-1]    
        app = filename.split('.')[0]    
    
    # Create the DAG object.    
    dag = DAG(app=app)
    done = set()    
    # Construct dag.DAG.   
    for t in nx.topological_sort(G):
        if t not in done:            
            nd = Task()            
            nd.ID = int(t)
            nd.entry = True
            done.add(t)
        else:
            for n in dag.DAG:
                if n.ID == int(t):
                    nd = n    
                    break
        count = 0
        for s in G.successors(t):
            count += 1
            if s not in done:                
                nd1 = Task()                
                nd1.ID = int(s)
                done.add(s) 
            else:
                for n in dag.DAG:
                    if n.ID == int(s):
                        nd1 = n
                        break
            dag.DAG.add_edge(nd, nd1) 
        if not count:
            nd.exit = True   
                
    # Number of tasks.
    dag.num_tasks = len(dag.DAG)  
    
    if draw:
        G.graph['graph'] = {'rankdir':'TD'}        
        G.graph['node']={'shape':'circle', 'color':'black', 'style':'filled', 'fillcolor':'lightcyan', 'penwidth':'3.0', 'height' : '2.0', 'fontsize' : '85.0'}
        G.graph['edges']={'arrowsize':'4.0', 'penwidth':'5.0'}
        A = to_agraph(G)
        print("The DOT file used to generate the graph looks like:")
        print(A)
        A.layout('dot')
        A.draw('../graphs/images/{}_{}tasks_STG.png'.format(dag.app, dag.num_tasks))     # May need to change destination depending on current directory.      
         
    return dag

def convert_from_nx_graph(graph, app="Random", single_exit=False, draw=False):
    """ 
    Creates a DAG object from an input DiGraph.
    """
    # Make the graph directed if it isn't already.
    if graph.is_directed():
        G = graph
    else:
        G = nx.DiGraph()
        G.name = graph.name
        G.add_nodes_from(graph)    
        done = set()
        for u, v in graph.edges():
            if (v, u) not in done:
                G.add_edge(u, v)
                done.add((u, v))  
        G.graph = deepcopy(graph.graph)     
    # Look for cycles...
    try:
        nx.topological_sort(G)
    except nx.NetworkXUnfeasible:
        raise ValueError('Input graph in convert_from_nx_graph has at least one cycle so is not a DAG!')
    
    # Add single exit node if desired.
    if single_exit:
        exits = set(nd for nd in G if not list(G.successors(nd)))
        num_exits = len(exits)
        if num_exits > 1:
            terminus = len(G)
            G.add_node(terminus)
            for nd in G:
                if nd in exits:
                    G.add_edge(nd, terminus)   
                    
        
    # Create the DAG object.
    dag = DAG(app=app)
    done = set()
    
    # Construct dag.DAG.    
    for t in nx.topological_sort(G):
        if t not in done:           
            nd = Task()                      
            nd.ID = int(t)
            nd.entry = True
            done.add(t)
        else:
            for n in dag.DAG:
                if n.ID == int(t):
                    nd = n 
                    break
        
        count = 0
        for s in G.successors(t):
            count += 1
            if s not in done:
                nd1 = Task()                               
                nd1.ID = int(s)
                done.add(s) 
            else:
                for n in dag.DAG:
                    if n.ID == int(s):
                        nd1 = n
                        break
            dag.DAG.add_edge(nd, nd1) 
        if not count:
            nd.exit = True     
    
    # Number of tasks.
    dag.num_tasks = len(dag.DAG)         
    
    if draw:
        G.graph['graph'] = {'rankdir':'TD'}        
        G.graph['node']={'shape':'circle', 'color':'black', 'style':'filled', 'fillcolor':'lightcyan', 'penwidth':'3.0', 'height' : '2.0', 'fontsize' : '85.0'}
        G.graph['edges']={'arrowsize':'4.0', 'penwidth':'5.0'}
        A = to_agraph(G)
        print("The DOT file used to generate the graph looks like:")
        print(A)
        A.layout('dot')    
        A.draw('{}_{}tasks.png'.format(dag.app, dag.num_tasks)) 
        
    return dag