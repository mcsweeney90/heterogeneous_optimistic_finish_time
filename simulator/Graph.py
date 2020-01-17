#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module contains classes which create a framework for describing task DAGs.  

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
    Represents static tasks.
    """         
    def __init__(self, task_type=None):
        """
        Create Task object.
        
        Parameters
        ------------------------
        task_type - None/string
        String identifying the name of the task, e.g., "GEMM".
        
        Attributes
        ------------------------
        type - None/string
        Initialized to task_type.
        
        ID - int
        Identification number of the Task in its DAG.
        
        entry - bool
        True if Task has no predecessors, False otherwise.
        
        exit - bool
        True if Task has no successors, False otherwise.
        
        The following 4 attributes are usually set after initialization by functions which
        take a Node object as a target platform.
        
        CPU_time - int/float
        The Task's execution time on CPU Workers. 
        
        GPU_time - int/float
        The Task's execution time on GPU Workers. 
        
        acceleration_ratio - int/float
        The ratio of the Task's execution time on CPU and GPU Workers. 
        
        comm_costs - defaultdict(dict)
        Nested dict {string identifying source and target processor types : {child ID : cost}}
        e.g., self.comm_costs["CG"][5] = 10 means that the communication cost between the Task
        and the child task with ID 5 is 10 when Task is scheduled on a CPU Worker and the child 
        is scheduled on a GPU Worker.
        
        The following 4 attributes are set once the task has actually been scheduled.
        
        AST - int/float
        The actual start time of the Task.
        
        AFT- int/float
        The actual finish time of the Task.
        
        scheduled - bool
        True if Task has been scheduled on a Worker, False otherwise.
        
        where_scheduled - None/int
        The numerical ID of the Worker that the Task has been scheduled on. Often useful.
        
        Comments
        ------------------------
        1. It would perhaps be more useful in general to take all attributes as parameters since this
           is more flexible but as we rarely work at the level of individual Tasks this isn't necessary
           for our purposes.        
        """           
         
        self.type = task_type  
        self.ID = None    
        self.entry = False 
        self.exit = False     
        
        self.CPU_time = 0  
        self.GPU_time = 0  
        self.acceleration_ratio = 0 
        self.comm_costs = defaultdict(dict) 
        self.comm_costs["CC"], self.comm_costs["CG"], self.comm_costs["GC"], self.comm_costs["GG"] = {}, {}, {}, {} 
        
        self.AST = 0   
        self.AFT = 0  
        self.scheduled = False  
        self.where_scheduled = None                 
    
    def reset(self):
        """Resets some attributes to defaults so execution of the task can be simulated again."""
        self.AFT = 0
        self.AST = 0   
        self.scheduled = False
        self.where_scheduled = None           
        
    def approximate_execution_cost(self, platform, weighting="HEFT"):
        """
        Compute the "approximate" computation time of the Task. 
        Usually used for setting priorities in HEFT and similar heuristics.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.
                
        weighting - string
        How the approximation should be computed. 
        Options:
            - "HEFT", use mean values over all processors as in HEFT.
            - "median", use median values over all processors. 
            - "worst", always use largest possible computation cost.
            - "simple worst", always use largest possible computation cost.
            - "best", always use smallest possible computation cost.
            - "simple best", always use smallest possible computation cost.
            - "HEFT-WM", compute mean over all processors, weighted by acceleration ratio.
            - "PS", processor speedup. Cost = max(CPU time, GPU time) / min(CPU time, GPU time).
            - "D", difference. Cost = max(CPU time, GPU time) - min(CPU time, GPU time).
            - "SFB". Cost = ( max(CPU time, GPU time) - min(CPU time, GPU time) ) / ( max(CPU time, GPU time) / min(CPU time, GPU time) ). 
                                         
        Returns
        ------------------------
        float 
        The approximate computation cost of the Task. 
        
        Notes
        ------------------------
        1. "median", "worst", "simple worst", "best", "simple best" were all considered by Zhao and Sakellariou, 2003. 
        2. "PS", "D" and "SFB" are from Shetti, Fahmy and Bretschneider, 2013.
        """
        
        if weighting == "HEFT" or weighting == "mean" or weighting == "MEAN" or weighting == "M":
            return (self.CPU_time * platform.n_CPUs + self.GPU_time * platform.n_GPUs) / platform.n_workers
        elif weighting == "median" or weighting == "MEDIAN":
            execution_costs = [self.CPU_time for _ in range(platform.n_CPUs)] + [self.GPU_time for _ in range(platform.n_GPUs)]
            return median(execution_costs)
        elif weighting == "worst" or weighting == "W" or weighting == "simple worst" or weighting == "SW":
            return max(self.CPU_time, self.GPU_time)
        elif weighting == "best" or weighting == "B" or weighting == "simple best" or weighting == "sb":
            return min(self.CPU_time, self.GPU_time)   
        elif weighting == "HEFT-WM" or weighting == "WM":
            r = self.acceleration_ratio
            return (self.CPU_time * platform.n_CPUs + r * self.GPU_time * platform.n_GPUs) / (platform.n_CPUs + r * platform.n_GPUs)         
        elif weighting == "PS" or weighting == "ps":
            fastest, slowest = min(self.CPU_time, self.GPU_time), max(self.CPU_time, self.GPU_time)
            if not fastest:
                return 0
            return slowest / fastest
        elif weighting == "D" or weighting == "d":
            fastest, slowest = min(self.CPU_time, self.GPU_time), max(self.CPU_time, self.GPU_time)
            return slowest - fastest
        elif weighting == "SFB" or weighting == "sfb":
            fastest, slowest = min(self.CPU_time, self.GPU_time), max(self.CPU_time, self.GPU_time)
            if not fastest:
                return 0
            return (slowest - fastest) / (slowest / fastest)             
        raise ValueError('No weighting, e.g., "mean" or "median", specified for approximate_execution_cost.')   

    def ready_to_schedule(self, dag):
        """
        Determine if Task is ready to schedule - i.e., all precedence constraints have been 
        satisfied or it is an entry task.
        
        Parameters
        ------------------------
        dag - DAG object
        The DAG to which the task belongs.                
                                         
        Returns
        ------------------------
        bool
        True if Task can be scheduled, False otherwise.         
        
        Notes
        ------------------------
        1. Returns False if Task has already been scheduled.
        """
        if self.scheduled:
            return False  
        if self.entry: 
            return True
        for p in dag.DAG.predecessors(self):
            if not p.scheduled:
                return False
        return True

class DAG:
    """
    Represents an application task DAG.   
    """
    def __init__(self, app="Random"): 
        """
        The DAG is a collection of Tasks with a topology defined by a Networkx DiGraph object.        
        
        Parameters
        ------------------------
        app - string
        The name of application the DAG represents, e.g., "Cholesky".
        
        Attributes
        ------------------------
        app - string
        Ditto above.
        
        DAG - DiGraph from Networkx module
        Represents the topology of the DAG.
        
        num_tasks - int
        The number of tasks in the DAG.
        
        The following attributes summarize topological information and are usually set
        by compute_topological_info when necessary.
        
        max_task_predecessors - None/int
        The maximum number of predecessors possessed by any task in the DAG.    
        
        avg_task_predecessors - None/int
        The average number of predecessors possessed by all tasks in the DAG.
               
        num_edges - None/int
        The number of edges in the DAG. 
        
        edge_density - None/float
        The ratio of the number of edges in the DAG to the maximum possible for a DAG with the same
        number of tasks. 
        
        CCR - dict {string : float}
        Summarizes the computation-to-communication ratio (CCR) values for different platforms in the
        form {platform name : DAG CCR}.         
        
        Comments
        ------------------------
        1. It seems a little strange to make the CCR a dict but it avoided having to compute it over 
           and over again for the same platforms in some scripts.
        """   
        self.app = app 
        self.DAG = nx.DiGraph()
        self.num_tasks = 0  
        self.max_task_predecessors = None
        self.avg_task_predecessors = None
        self.num_edges = None
        self.edge_density = None 
        self.CCR = {}

    def compute_topological_info(self):
        """
        Compute information about the DAG's topology and set the corresponding attributes.                
             
        Notes
        ------------------------
        1. The maximum number of edges for a DAG with n tasks is 1/2 * n * (n - 1).
           This can be proven e.g., by considering each vertex in turn and determining
           the maximum number of possible new edges.
        """  
        
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
            max_edges = (self.num_tasks * (self.num_tasks - 1)) / 2 
            self.edge_density = self.num_edges / max_edges         
            
    def compute_CCR(self, platform):
        """
        Compute the computation-to-communication ratio (CCR) for the DAG on the target platform.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.           
        """
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
        """
        Print basic information about the DAG, either to screen or as txt file.
        
        Parameters
        ------------------------
        platform - None/Node object (see Environment.py module)/list
        Compute more specific information about the DAG when executed on the platform (if Node)
        or multiple platforms (if list of Nodes).
        
        detailed - bool
        If True, print information about individual Tasks.
        
        filepath - string
        Destination for txt file.                           
        """
        print("--------------------------------------------------------", file=filepath)
        print("DAG INFO", file=filepath)
        print("--------------------------------------------------------", file=filepath)   
        print("Application: {}".format(self.app), file=filepath)
        print("Number of tasks: {}".format(self.num_tasks), file=filepath)
        self.compute_topological_info()
        print("Maximum number of task predecessors: {}".format(self.max_task_predecessors), file=filepath)
        print("Average number of task predecessors: {}".format(self.avg_task_predecessors), file=filepath)
        print("Number of edges: {}".format(self.num_edges), file=filepath)
        print("Edge density: {}".format(self.edge_density), file=filepath)
        
        if platform is not None: 
            cpu_times = list(task.CPU_time for task in self.DAG)
            gpu_times = list(task.GPU_time for task in self.DAG)
            acc_ratios = list(task.acceleration_ratio for task in self.DAG)
            cpu_mu, cpu_sigma = np.mean(cpu_times), np.std(cpu_times)
            print("Mean task CPU time: {}, standard deviation: {}".format(cpu_mu, cpu_sigma), file=filepath)
            gpu_mu, gpu_sigma = np.mean(gpu_times), np.std(gpu_times)
            print("Mean task GPU time: {}, standard deviation: {}".format(gpu_mu, gpu_sigma), file=filepath)            
            acc_mu, acc_sigma = np.mean(acc_ratios), np.std(acc_ratios)
            print("Mean task acceleration ratio: {}, standard deviation: {}".format(acc_mu, acc_sigma), file=filepath) 
            
            if isinstance(platform, list):
                for p in platform:
                    task_mu = (p.n_GPUs * gpu_mu + p.n_CPUs * cpu_mu) / p.n_workers
                    print("\nMean task execution time on {} platform: {}".format(p.name, task_mu), file=filepath)
                    try:
                        ccr = self.CCR[p.name]
                    except KeyError:                 
                        ccr = self.compute_CCR(p)
                    print("Computation-to-communication ratio on {} platform: {}".format(p.name, ccr), file=filepath)
                    mst = self.minimal_serial_time(platform)
                    print("Minimal serial time on {} platform: {}".format(p.name, mst), file=filepath)
            else:
                task_mu = (platform.n_GPUs * gpu_mu + platform.n_CPUs * cpu_mu) / platform.n_workers
                print("Mean task execution time on {} platform: {}".format(platform.name, task_mu), file=filepath)            
                # Compute the CCR if not already set.
                try:
                    ccr = self.CCR[platform.name]
                except KeyError:                 
                    ccr = self.compute_CCR(platform)
                print("Computation-to-communication ratio: {}".format(ccr), file=filepath)
            
                mst = self.minimal_serial_time(platform)
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
                    print("Acceleration ratio: {}".format(task.acceleration_ratio), file=filepath)               
        print("--------------------------------------------------------", file=filepath)        
            
        
    def draw_graph(self, filepath="graphs/images"):
        """
        Draws the DAG and saves the image.
        
        Parameters
        ------------------------        
        filepath - string
        Destination for image. 

        Notes
        ------------------------                           
        1. See https://stackoverflow.com/questions/39657395/how-to-draw-properly-networkx-graphs       
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
        
        A.layout('dot')
        A.draw('{}/{}_{}tasks_DAG.png'.format(filepath, self.app.split(" ")[0], self.num_tasks))            
            
    def set_costs(self, platform, target_ccr, ratios=None, ratio_dist=("gamma", 40.0)):
        """
        Sets computation and communication costs for randomly generated DAGs (e.g., from the STG).
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.           
        
        target_ccr - float/int
        The CCR we want the DAG to have on the target platform. Due to stochasticity in how we choose 
        communication costs this is not precise so we might need to double check afterwards.
        
        ratios - None/dict
        Record of acceleration ratios for task types on target platform, {task type : acc. ratio}.
        
        ratio_dist - tuple/list
        Used to set task acceleration ratios if ratios is not input. 
        First entry is string giving name of probability distribution to sample from and other entries
        describe the relevant moments.
        Options:
            - ("gamma", x). Gamma distribution with mean and variance x.
            - ("normal", x, y). Normal distribution with mean x and standard deviation y.        
                      
        Notes
        ------------------------
        1. We assume that GPU times are uniformly distributed integers between 1 and 100.
        2. We assume that CPU-CPU communication costs are zero and all others are of similar magnitude to
           one another (as we typically have throughout).
        3. Communication costs are sampled from a Gamma distribution with a computed mean and standard deviation
           to try and achieve the desired CCR value for the DAG.
        """
        
        if self.num_edges is None:
            self.num_edges = self.DAG.number_of_edges()
        
        # Set the computation costs.
        for task in self.DAG:
            task.GPU_time = np.random.randint(1, 100) 
            if task.acceleration_ratio == 0: # If acceleration ratio not known...         
                if isinstance(ratios, dict):
                    task.acceleration_ratio = ratios[task.type] 
                else:
                    if ratio_dist[0] == "gamma":
                        task.acceleration_ratio = np.random.gamma(shape=1.0, scale=ratio_dist[1]) 
                    elif ratio_dist[0] == "normal":
                        task.acceleration_ratio = np.random.normal(ratio_dist[1], ratio_dist[2]) 
                    else:
                        raise ValueError('Unrecognized ratio_dist entered in set_costs!')
                    
            task.CPU_time = task.GPU_time * task.acceleration_ratio
        
        # Set the communication costs.
        
        # Compute the expected total compute of the entire DAG.
        cpu_times = list(task.CPU_time for task in self.DAG)
        gpu_times = list(task.GPU_time for task in self.DAG)
        expected_compute = sum(cpu_times) * platform.n_CPUs + sum(gpu_times) * platform.n_GPUs
        expected_compute /= platform.n_workers
        
        # Calculate the expected communication cost of the entire DAG - i.e., for all edges.        
        expected_comm = expected_compute / target_ccr
        # Compute the sum of all CPU-GPU comm costs, for all tasks. 
        total_cg_comm = expected_comm * platform.n_workers**2   
        total_cg_comm /= (2 * platform.n_CPUs * platform.n_GPUs + platform.n_GPUs * (platform.n_GPUs - 1))
        # Assume that gc_comm and gg_comm are approximately the same.
        
        comm_per_task = total_cg_comm / self.num_tasks        
        for task in self.DAG:
            if task.exit:               
                continue
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
        
        # Compute the actual CCR.
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
        
        # Save CCR for platform as the actual value.
        self.CCR[platform.name] = mean_compute / mean_comm                                                             
                
    def all_tasks_scheduled(self):
        """Returns True all the tasks in the DAG have been scheduled, False if not."""
        return all(task.scheduled for task in self.DAG)
    
    def reset(self):
        """Resets some Task attributes to defaults so scheduling of the DAG can be simulated again."""
        for task in self.DAG:
            task.reset()
            
    def makespan(self, partial=False):
        """
        Compute the makespan of the DAG.
        
        Parameters
        ------------------------        
        partial - bool
        If True, only computes makespan of all tasks that have been scheduled so far, not the entire DAG. 

        Returns
        ------------------------         
        int/float
        The makespan of the (possibly incomplete) DAG.        
        """ 
        if not partial:
            if not self.all_tasks_scheduled():
                raise ValueError('Error! There are tasks in the DAG which are not scheduled yet!')
            return max(t.AFT for t in self.DAG if t.exit)  
        return max(t.AFT for t in self.DAG)    
                   
    def sort_by_upward_rank(self, platform, weighting="HEFT", return_rank_values=False, verbose=False):
        """
        Sorts all tasks in the DAG by decreasing/non-increasing order of upward rank.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.
        
        Weighting - string
        How the tasks and edges should be weighted in platform.approximate_comm_cost and task.approximate_execution_cost.
        Default is "HEFT" which is mean values over all processors. See referenced methods for more options.
        
        return_rank_values - bool
        If True, method also returns the upward rank values for all tasks.
        
        verbose - bool
        If True, print the ordering of all tasks to the screen. Useful for debugging. 

        Returns
        ------------------------                          
        priority_list - list
        Scheduling list of all Task objects prioritized by upward rank.
        
        If return_rank_values == True:
        task_ranks - dict
        Gives the actual upward ranks of all tasks in the form {task : rank_u}.
        
        Notes
        ------------------------ 
        1. "Upward rank" is also called "bottom-level".        
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
        
        if verbose:
            priority_list_ids = list(t.ID for t in priority_list)
            print("The priority list is: {}".format(priority_list_ids))
            
        if return_rank_values:
            return priority_list, task_ranks
        return priority_list  
    
    def sort_by_downward_rank(self, platform, weighting="HEFT", return_rank_values=False, verbose=False):
        """
        Sorts all tasks in the DAG by increasing/non-decreasing order of downward rank.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.
        
        Weighting - string
        How the tasks and edges should be weighted in platform.approximate_comm_cost and task.approximate_execution_cost.
        Default is "HEFT" which is mean values over all processors. See referenced methods for more options.
        
        return_rank_values - bool
        If True, method also returns the downward rank values for all tasks.
        
        verbose - bool
        If True, print the ordering of all tasks to the screen. Useful for debugging. 

        Returns
        ------------------------                          
        priority_list - list
        Scheduling list of all Task objects prioritized by downward rank.
        
        If return_rank_values == True:
        task_ranks - dict
        Gives the actual downward ranks of all tasks in the form {task : rank_d}.
        
        Notes
        ------------------------ 
        1. "Downward rank" is also called "top-level".        
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
            
        if verbose:
            priority_list_ids = list(t.ID for t in priority_list)
            print("The priority list is: {}".format(priority_list_ids))
            
        if return_rank_values:
            return priority_list, task_ranks
        return priority_list 

    def optimistic_cost_table(self, platform):
        """
        Computes the Optimistic Cost Table, as defined in Arabnejad and Barbosa (2014), for the given platform.
        Used in the PEFT heuristic - see Heuristics.py.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.        

        Returns
        ------------------------                          
        OCT - Nested defaultdict
        The optimistic cost table in the form {Task 1: {Worker 1 : c1, Worker 2 : c2, ...}, ...}.             
        """   
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
        Computes the optimistic finish time, as defined in the Heterogeneous Optimistic Finish Time (HOFT) algorithm,
        of all tasks assuming they are scheduled on either CPU or GPU. 
        Used in the HOFT heuristic - see Heuristics.py.                  

        Returns
        ------------------------                          
        OFT - Nested defaultdict
        The optimistic finish time table in the form {Task 1: {Worker 1 : c1, Worker 2 : c2, ...}, ...}.         
        
        Notes
        ------------------------ 
        1. No target platform is necessary as parameter since task.CPU_time and GPU_time are assumed to be set for all tasks. 
           Likewise, task.comm_costs is assumed to be set for all tasks. 
        """  
        
        d = defaultdict(int)
        d["CC"], d["GG"] = 0, 0
        d["CG"], d["GC"]  = 1, 1     
        
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
    
    def critical_path(self):
        """
        Computes the critical path, a lower bound on the makespan of the DAG.               

        Returns
        ------------------------                          
        cp - float
        The length of the critical path.        
        
        Notes
        ------------------------ 
        1. No target platform is necessary as input since task.CPU_time and GPU_time are assumed to be set for all tasks. 
           Likewise, task.comm_costs is assumed to be set for all tasks. 
        2. There are alternative ways to compute the critical path but unlike some others this approach takes
           communication costs into account.
        """        
        OFT = self.optimistic_finish_times()
        cp = max(min(OFT[task][processor] for processor in OFT[task]) for task in OFT if task.exit)                                
        return cp    
   
    def minimal_serial_time(self, platform):
        """
        Computes the minimum makespan of the DAG on a single Worker of the platform.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.        

        Returns
        ------------------------                          
        float
        The minimal serial time.      
        
        Notes
        ------------------------                          
        1. Assumes all task CPU and GPU times are set.        
        """        
        return min(sum(task.CPU_time for task in self.DAG), sum(task.GPU_time for task in self.DAG))
    
    def get_ready_tasks(self):
        """
        Identify the tasks that are ready to schedule.               

        Returns
        ------------------------                          
        List
        All tasks in the DAG that are ready to be scheduled.                 
        """       
        return list(t for t in filter(lambda t: t.ready_to_schedule(self) == True, self.DAG))    
        

####################################################################################################    
# Functions to create DAG objects from dot files and Networkx DiGraphs.    
####################################################################################################        
        
def convert_from_dot(dot_path, app=None):
    """
    Create a DAG object from a graph stored as a dot file.
    
    Parameters
    ------------------------
    dot_path - string
    Where the dot file is located.

    app - None/string
    The application that the graph represents, e.g., "Cholesky".   

    Returns
    ------------------------                          
    dag - DAG object
    Converted version of the graph described by the dot file.      
    
    Notes
    ------------------------                          
    1. This is very slow so isn't recommended for even medium-sized DAGs. The vast majority of the time 
       seems to be taken by read_dot (from Networkx) itself so think there may be a bug somewhere.       
    """  
    # Use read_dot from Networkx to load the graph.        
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
    dag.num_tasks = len(dag.DAG)      
         
    return dag

def convert_from_nx_graph(graph, app="Random", single_exit=False):
    """
    Create a DAG object from a graph stored as a dot file.
    
    Parameters
    ------------------------
    graph - Networkx Graph (ideally DiGraph)
    The graph to be converted to a DAG object.

    app - None/string
    The application that the graph represents, e.g., "Cholesky".  
    
    single_exit - bool
    If True, add an artificial single exit task.

    Returns
    ------------------------                          
    dag - DAG object
    Converted version of the graph.           
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
    
    # Add single exit node if specified.
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
    dag.num_tasks = len(dag.DAG)       
        
    return dag