#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module contains classes which create a framework for describing arbitrary CPU and GPU computing environments. 

Notes:
    1. Although "GPU" is used throughout, both processor types are defined entirely by their task
       processing/communication times so this can easily be used for any computing environment with only two different 
       types of processing resources.
       
"""

import numpy as np
from statistics import median
from math import inf
from collections import defaultdict

class Worker:
    """
    Represents any CPU or GPU processing resource. 
    """
    def __init__(self, GPU=False, ID=None):
        """
        Create the Worker object.
        
        Parameters
        --------------------
        GPU - bool
        True if Worker is a GPU. Assumed to be a CPU unless specified otherwise.
        
        ID - Int
        Assigns an integer ID to the task. Often very useful.        
        """        
        self.GPU = True if GPU else False
        self.CPU = not self.GPU   
        self.ID = ID   
        self.load = [] # Tasks scheduled on the processor.
        self.idle = True # True if no tasks currently scheduled on the processor. 
                
    def earliest_start_time(self, task, dag, platform, insertion=True):
        """
        Returns the estimated earliest start time for a task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
        
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Node object
        The Node object to which the Worker belongs.
        Needed for calculating communication costs.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        
        Returns
        ------------------------
        float 
        The earliest start time for task on Worker.        
        """        
        
        if self.idle:   # If no tasks scheduled on processor...
            if task.entry: # If an entry task...
                return 0   
            else:
                return max(p.AFT + platform.comm_cost(p, task, p.where_scheduled, self.ID) 
                                    for p in dag.DAG.predecessors(task))  
                
        # Find earliest time all task predecessors have finished and the task can theoretically start.     
        est = 0
        if not task.entry:                    
            predecessors = dag.DAG.predecessors(task) 
            est += max(p.AFT + platform.comm_cost(p, task, p.where_scheduled, self.ID) 
                                    for p in predecessors)
            
        processing_time = task.CPU_time if self.CPU else task.GPU_time            
        
        # At least one task already scheduled on processor... 
        # Check if it can be scheduled before any task.
        prev_finish_time = 0
        for t in self.load:
            if t.AST < est:
                prev_finish_time = t.AFT
                continue
            poss_start_time = max(prev_finish_time, est)
            if poss_start_time + processing_time <= t.AST:
                return poss_start_time
            prev_finish_time = t.AFT
        
        # No valid gap found.
        return max(self.load[-1].AFT, est)    
        
    def earliest_finish_time(self, task, dag, platform, insertion=True, start_time=None): 
        """
        Returns the estimated earliest finish time for a task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
        
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Node object
        The Node to which the Worker belongs. 
        Needed for calculating communication costs.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        
        start_time - float
        If not None, taken to be task's actual start time. Validity is checked with valid_start_time which raises
        ValueError if it fails. Should be used very carefully!              
        
        Returns
        ------------------------
        float 
        The earliest finish time for task on Worker. 
        """   
        processing_time = task.CPU_time if self.CPU else task.GPU_time  
        # If start_time, make sure it is valid.
        if start_time: 
            if not self.valid_start_time(task, start_time, dag, platform):
                raise ValueError('Invalid input start time! Processor ID: {}, task ID, attempted start time: {}'.format(self.ID, task.ID, start_time))
            return processing_time + start_time
        return processing_time + self.earliest_start_time(task, dag, platform, insertion=insertion)      

    def valid_start_time(self, task, start_time, dag, platform, insertion=True):
        """
        Check if input start time is valid.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
        
        start_time - float
        If not None, taken to be task's actual start time. Validity is checked with 
        valid_start_time which raises ValueError if it fails. Should be used very carefully!
        
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Node object
        The Node object to which the Worker belongs. 
        Needed for calculating communication costs, although this is a bit unconventional.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        
        Returns
        ------------------------
        bool
        True if the input start time is valid (respects all dependencies, etc), False otherwise.                  
        """          
        
        # Compute earliest possible start time. 
        est = max(p.AFT + platform.comm_cost(p, task, p.where_scheduled, self.ID) 
                                    for p in dag.DAG.predecessors(task)) 
        
        if start_time < est:
            return False
        
        if not insertion:
            if start_time < self.load[-1].AFT:
                return False
            return True
        else:
            if start_time > self.load[-1].AFT:
                return True
            processing_time = task.CPU_time if self.CPU else task.GPU_time
            for i, t in enumerate(self.load[:-1]):
                if start_time > t.AFT and start_time + processing_time < self.load[i + 1].AST:
                    return True
                if t.AST > start_time:
                    break
            return False            
        return False
        
    def schedule_task(self, task, dag, platform, insertion=True, start_time=None, finish_time=None):
        """
        Schedules the task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
                
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Node object
        The Node object to which the Worker belongs. 
        Needed for calculating communication costs, although this is a bit unconventional.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        
        start_time - float
        If not None, schedules task at this start time. Validity is checked with 
        valid_start_time which raises ValueError if it fails. Should be used very carefully!
        
        finish_time - float
        If not None, taken to be task's actual finish time. 
        Should be used with great care (see note below!)
        
        Notes
        ------------------------
        1. If finish_time, doesn't check that all task predecessors have actually been scheduled.
           This is so we can do lookahead in e.g., platform.estimate_finish_times and to save repeated
           calculations in some circumstances but should be used very, very carefully!
                 
        """                    
        if start_time:
            if not self.valid_start_time(task, start_time, dag, platform):
                raise ValueError('Invalid input start time! Processor ID: {}, task ID, attempted start time: {}'.format(self.ID, task.ID, start_time))
        
        # Set task attributes.
        if finish_time:
            task.AFT = finish_time
        else:
            task.AFT = self.earliest_finish_time(task, dag, platform, insertion=insertion, start_time=start_time)        
        task.AST = task.AFT - task.CPU_time if self.CPU else task.AFT - task.GPU_time 
        task.scheduled = True
        task.where_scheduled = self.ID 

        if self.idle:
            self.idle = False 
            self.load.append(task)
            return
        
        if not insertion:             
            self.load.append(task)            
        else: 
            idx = -1
            for t in self.load:
                if task.AST < t.AST:
                    idx = self.load.index(t)
                    break
            if idx > -1:
                self.load.insert(idx, task)
            else:
                self.load.append(task)         

    def unschedule_task(self, task):
        """
        Unschedules the task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.                 
        """
        # Remove task from the load.
        self.load.remove(task)
        # Revert Worker to idle if necessary.
        if not len(self.load):
            self.idle = True
        # Reset the task itself.    
        task.reset()                         
        
    def print_schedule(self, filepath=None):
        """
        Print the current tasks scheduled on the Worker, either to screen or as txt file.
        
        Parameters
        ------------------------
        filepath - string
        Destination for schedule txt file.                           
        """
        proc_type = "CPU" if self.CPU else "GPU"
        print("PROCESSOR {}, TYPE {}: ".format(self.ID, proc_type), file=filepath)
        for t in self.load:
            task_info = "Task type: {},".format(t.type) if t.type else ""
            print("Task ID: {}, {}, AST = {}, AFT = {}.".format(t.ID, task_info, t.AST, t.AFT), file=filepath)     
 
class Node:
    """          
    A Node is basically just a collection of CPU and GPU Worker objects.
    """
    def __init__(self, CPUs, GPUs, name="generic", communication=True, adt=False):
        """
        Initialize the Node by giving the number of CPUs and GPUs.
        
        Parameters
        ------------------------
        CPUs - int
        The number of CPUs.

        GPUs - int
        The number of GPUs.
        
        name - string
        An identifying name for the Node. Often useful.
        
        communication - bool
        If False, disregard all communication - all costs are taken to be zero.
        
        adt - bool
        If True, simulates the effect of asynchronous data transfers. 
        Not used anywhere but may be used in future.
        """
        self.name = name  
        self.communication = communication         
        self.adt = adt
        
        self.n_CPUs, self.n_GPUs = CPUs, GPUs 
        self.n_workers = self.n_CPUs + self.n_GPUs # Often useful.
        self.workers = [] # List of all Worker objects.
        for i in range(self.n_CPUs):
            self.workers.append(Worker(ID=i))          
        for j in range(self.n_GPUs):
            self.workers.append(Worker(GPU=True, ID=self.n_CPUs + j))                             
    
    def print_info(self, filepath=None):
        """
        Print basic information about the Node, either to screen or as txt file.
        
        Parameters
        ------------------------
        filepath - string
        Destination for txt file.                           
        """
        print("--------------------------------------------------------", file=filepath)
        print("NODE INFO", file=filepath)
        print("--------------------------------------------------------", file=filepath)
        print("Name: {}".format(self.name), file=filepath)
        print("{} CPUs, {} GPUs".format(self.n_CPUs, self.n_GPUs), file=filepath)
        print("Communication: {}".format(self.communication), file=filepath) 
        print("Asynchronous data transfer: {}".format(self.adt), file=filepath)
        print("--------------------------------------------------------\n", file=filepath)            
       
    def where_scheduled(self, task):
        """
        Which Worker the task is currently scheduled on.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task. 

        Returns
        ------------------------ 
        None/int
        ID of the Worker on which task is scheduled, None if task is not scheduled anywhere.               
        """
        for p in range(self.n_workers):
            if task in self.workers[p].load:
                return p
        return None       
    
    def reset(self):
        """Resets some attributes to defaults so we can simulate the execution of another DAG. """
        for w in self.workers:
            w.load = []   
            w.idle = True 
            
    def print_schedule(self, heuristic_name="", filepath=None):
        """
        Print the current schedule, all tasks scheduled on each Worker, either to screen or as txt file.
        
        Parameters
        ------------------------
        heuristic_name - string
        Name of the heuristic which produced the current schedule. Often helpful.
        
        filepath - string
        Destination for schedule txt file.                           
        """
        print("--------------------------------------------------------", file=filepath)
        print("{} SCHEDULE".format(heuristic_name), file=filepath)
        print("--------------------------------------------------------", file=filepath)
        for w in self.workers:
            w.print_schedule(filepath=filepath)  
        makespan = max(w.load[-1].AFT for w in self.workers if w.load) 
        print("\n{} MAKESPAN: {}".format(heuristic_name, makespan), file=filepath)            
        print("--------------------------------------------------------\n", file=filepath)
        
    
    def valid_schedule(self, schedule, dag):
        """
        Check if input schedule is valid.
        
        Parameters
        ------------------------
        schedule - list of the form [(task, processor scheduled on, start time), ...]
        Represents a (static) task.        
        
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
        
        Returns
        ------------------------
        bool
        True if the input schedule is valid (doesn't violate any task dependencies or 
        schedule multiple tasks on same processor at same time), False otherwise.                  
        """    
        finish_times, where_scheduled, processor_loads, start_times = {}, {}, {}, {}
        for t, p, st in schedule:
            start_times[t] = st
            finish_times[t] = st + t.CPU_time if p < self.n_CPUs else st + t.GPU_time
            where_scheduled[t] = p
            try:
                processor_loads[p].append(t)
            except KeyError:
                processor_loads[p] = [t]
        
        # Check all dependencies are satisfied.
        # Note that we don't assume that task.AFT and task.AST are set.
        for t, p, st in schedule:
            for pred in dag.DAG.predecessors(t):                
                if st < finish_times[pred] + self.comm_cost(pred, t, where_scheduled[pred], p): 
                    return False        
        
        # Sort the loads of each processor and check that there's no overlap.
        for proc in processor_loads:
            sorted(processor_loads[proc], key = lambda t: finish_times[t])
            for i, task in processor_loads[proc[:-1]]:
                s = processor_loads[proc][i + 1]
                if finish_times[t] > start_times[s]:
                    return False            
        return True  
    
    def comm_cost(self, parent, child, source_id, target_id):   
        """
        Compute the communication time from a parent task to a child.
        
        Parameters
        ------------------------
        parent - Task object (see Graph.py module)
        The parent task that is sending its data.
        
        child - Task object (see Graph.py module)
        The child task that is receiving data.
        
        source_id - int
        The ID of the Worker on which parent is scheduled.
        
        target_id - int
        The ID of the Worker on which child may be scheduled.
        
        Returns
        ------------------------
        float 
        The communication time between parent and child.        
        """       
        if source_id == target_id:
            return 0 
        if not self.communication:
            return 0          
        
        source_type = "G" if source_id > self.n_CPUs - 1 else "C"
        target_type = "G" if target_id > self.n_CPUs - 1 else "C"
        
        if self.adt:
            parent_comp_time = parent.GPU_time if source_type == "G" else parent.CPU_time
            return max(parent.comm_costs["{}".format(source_type + target_type)][child.ID] - parent_comp_time, 0)
        
        return parent.comm_costs["{}".format(source_type + target_type)][child.ID]    
                
    
    def approximate_comm_cost(self, parent, child, weighting="HEFT"): 
        """
        Compute the "approximate" communication time from parent to child tasks. 
        Usually used for setting priorities in HEFT and similar heuristics.
        
        Parameters
        ------------------------
        parent - Task object (see Graph.py module)
        The parent task that is sending its data.
        
        child - Task object (see Graph.py module)
        The child task that is receiving data.
        
        weighting - string
        How the approximation should be computed. 
        Options:
            - "HEFT", use mean values over all processors as in HEFT.
            - "median", use median values over all processors. 
            - "worst", assume each task is on its slowest processor type and compute corresponding communication cost.
            - "simple worst", always use largest possible communication cost.
            - "best", assume each task is on its fastest processor type and compute corresponding communication cost.
            - "simple best", always use smallest possible communication cost.
            - "HEFT-WM", compute mean over all processors, weighted by task acceleration ratios.
            - "PS", "D", "SFB" - speedup-based weightings from Shetti, Fahmy and Bretschneider, 2013. 
               Returns zero in all three cases so definitions can be found in approximate_execution_cost
               method in the Task class in Graph.py.
                                         
        Returns
        ------------------------
        float 
        The approximate communication cost between parent and child. 
        
        Notes
        ------------------------
        1. "median", "worst", "simple worst", "best", "simple best" were all considered by Zhao and Sakellariou, 2003. 
        """
        if not self.communication:
            return 0
        
        if weighting == "HEFT" or weighting == "mean" or weighting == "MEAN" or weighting == "M":            
            c_bar = self.n_CPUs * (self.n_CPUs - 1) * parent.comm_costs["CC"][child.ID] 
            c_bar += self.n_CPUs * self.n_GPUs * parent.comm_costs["CG"][child.ID]
            c_bar += self.n_CPUs * self.n_GPUs * parent.comm_costs["GC"][child.ID]
            c_bar += self.n_GPUs * (self.n_GPUs - 1) * parent.comm_costs["GG"][child.ID]
            c_bar /= (self.n_workers**2)
            return c_bar            
            
        elif weighting == "median" or weighting == "MEDIAN":
            costs = self.n_CPUs * (self.n_CPUs - 1) * [parent.comm_costs["CC"][child.ID]] 
            costs += self.n_CPUs * self.n_GPUs * [parent.comm_costs["CG"][child.ID]]
            costs += self.n_CPUs * self.n_GPUs * [parent.comm_costs["GC"][child.ID]]
            costs += self.n_GPUs * (self.n_GPUs - 1) * [parent.comm_costs["GG"][child.ID]]
            costs += self.n_workers * [0]
            return median(costs)
        
        elif weighting == "worst" or weighting == "WORST":
            parent_worst_proc = "C" if parent.CPU_time > parent.GPU_time else "G"
            child_worst_proc = "C" if child.CPU_time > child.GPU_time else "G"
            if parent_worst_proc == "C" and child_worst_proc == "C" and self.n_CPUs == 1:
                return 0
            if parent_worst_proc == "G" and child_worst_proc == "G" and self.n_GPUs == 1:
                return 0
            return parent.comm_costs["{}".format(parent_worst_proc + child_worst_proc)][child.ID]
        
        elif weighting == "simple worst" or weighting == "SW":
            return max(parent.comm_costs["CC"][child.ID], parent.comm_costs["CG"][child.ID], parent.comm_costs["GC"][child.ID], parent.comm_costs["GG"][child.ID])
        
        elif weighting == "best" or weighting == "BEST":
            parent_best_proc = "G" if parent.CPU_time > parent.GPU_time else "C"
            child_best_proc = "G" if child.CPU_time > child.GPU_time else "C"
            if parent_best_proc == child_best_proc:
                return 0
            return parent.comm_costs["{}".format(parent_best_proc + child_best_proc)][child.ID]
        
        elif weighting == "simple best" or weighting == "sb":
            return min(parent.comm_costs["CC"][child.ID], parent.comm_costs["CG"][child.ID], parent.comm_costs["GC"][child.ID], parent.comm_costs["GG"][child.ID])         
                
        elif weighting == "HEFT-WM" or weighting == "WM":
            A, B = parent.acceleration_ratio, child.acceleration_ratio
            c_bar = self.n_CPUs * (self.n_CPUs - 1) * parent.comm_costs["CC"][child.ID] 
            c_bar += self.n_CPUs * B * self.n_GPUs * parent.comm_costs["CG"][child.ID]
            c_bar += A * self.n_GPUs * self.n_CPUs * parent.comm_costs["GC"][child.ID]
            c_bar += A * self.n_GPUs * B * (self.n_GPUs - 1) * parent.comm_costs["GG"][child.ID]
            c_bar /= ((self.n_CPUs + A * self.n_GPUs) * (self.n_CPUs + B * self.n_GPUs))
            return c_bar              
            
        elif weighting == "PS" or weighting == "ps" or weighting == "D" or weighting == "d" or weighting == "SFB" or weighting == "sfb": 
            return 0
        
        raise ValueError('No weighting (e.g., "mean" or "median") specified for approximate_comm_cost.')
        
    def fastest_processor_of_type(self, task, dag, gpu=False):
        """
        Finds the fastest Worker of a CPU or GPU type for the task.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
        
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        gpu - bool
        If True, finds fastest GPU for task, else CPU.
        
        Returns
        ------------------------
        tuple (int, float)
        ID of the Worker of the input type expected to complete it at the earliest time and the 
        estimated time it can start. 
        """         
        if gpu:
            finish_times = [p.earliest_finish_time(task, dag, self) for p in self.workers if p.GPU]
            return np.argmin(finish_times) + self.n_CPUs, np.amin(finish_times) - task.GPU_time           
        finish_times = [p.earliest_finish_time(task, dag, self) for p in self.workers if p.CPU]
        return np.argmin(finish_times), np.amin(finish_times) - task.CPU_time    
        
    def schedule_batch(self, dag, batch, policy="eft", bmct_initial_allocation="met", mapping=None, batch_sort=None):
        """
        Schedule a batch of independent (no precedence constraints between them) tasks from a 
        DAG according to some input rule.
        
        Parameters
        ------------------------
        
        dag - DAG object (see Graph.py module)
        The DAG to which the tasks belongs.
              
        batch - list of Task objects
        The tasks to be scheduled.
        
        policy - string
        Describes the rule used for scheduling the tasks.
        Options:
            - "eft", use the classic Earliest Finish Time heuristic.
            - "met", use the classic Minimum Execution Time heuristic.
            - "bmct", use Sakellariou and Zhao's Balanced Minimum Completion Time heuristic.
            - "custom", expect a mapping {task : processor} to be specified and throw an error otherwise. 
            
        bmct_initial_allocation - string
        How the tasks should be initially scheduled when following the BMCT policy.
        Options:
            - "eft", use the classic Earliest Finish Time heuristic.
            - "met", use the classic Minimum Execution Time heuristic.
            - "random", assign tasks randomly, weighted by processor speed.
            
        mapping - dict, {task ID : processor ID}
        Which Worker each task should be scheduled on.
        
        batch_sort - None/string
        Optional, how to weight each task to compute priorities. Can use any weighting defined in task.approximate_execution_cost (see Graph.py).        
        """ 
        
        if batch_sort:
            weight = {}
            for task in batch:
                weight[task.ID] = task.approximate_execution_cost(self, weighting=batch_sort) 
            batch = list(reversed(sorted(batch, key=lambda t : weight[t.ID]))) # Descending order.            
    
        if policy == "eft":  
            for task in batch:
                min_processor = np.argmin([p.earliest_finish_time(task, dag, self) for p in self.workers])           
                self.workers[min_processor].schedule_task(task, dag, self)  
                
        elif policy == "met":
            for task in batch:
                if task.CPU_time < task.GPU_time:
                    rand_processor = np.random.choice(range(self.n_CPUs))
                else:
                    rand_processor = np.random.choice(range(self.n_CPUs, self.n_CPUs + self.n_GPUs))           
                self.workers[rand_processor].schedule_task(task, dag, self)                
                
        elif policy == "min-min" or policy == "max-min":
            
            # Compute the earliest completion times for all tasks in batch on all processors.
            ct = defaultdict(list)
            for t in batch:
                for p in self.workers:
                    ct[t.ID].append(p.earliest_finish_time(t, dag, self))
                    
            scheduled = {t.ID : False for t in batch}
            while True:
                # Compute minimum completion times for all processors.
                M = {t : min(ct[t.ID]) for t in batch if not scheduled[t.ID]}    
                if policy == "min-min":
                    # Find the task with the minimum completion time. 
                    next_task = min(M, key=M.get)
                else:
                    # Find the task with the maximum completion time. 
                    next_task = max(M, key=M.get)
                # Find the processor to schedule it on.
                proc = np.argmin(ct[next_task.ID])
                self.workers[proc].schedule_task(next_task, dag, self)
                scheduled[next_task.ID] = True
                
                # Check if we're finished.
                if all(scheduled[t.ID] for t in batch):
                    break               
                
                for t in batch:
                    # If already scheduled, no need to update.
                    if scheduled[t.ID]:
                        continue
                    # Update completion time estimate for proc.
                    ct[t.ID][proc] = self.workers[proc].earliest_finish_time(t, dag, self)              
                    
        elif policy == "bmct": 
            
            if bmct_initial_allocation == "random": 
                # Find mean acceleration ratio over all tasks in batch.
                mean_acc_ratio = np.mean(t.acceleration_ratio for t in batch)
                s = self.n_CPUs + self.n_GPUs * mean_acc_ratio
                relative_speeds = [1 / s] * self.n_CPUs + [mean_acc_ratio / s] * self.n_GPUs          
            
            # Schedule the tasks.
            for task in batch:
                # Schedule all tasks on their fastest processor initially (classic Minimum Execution Time heuristic).
                if bmct_initial_allocation == "met":                
                    if task.CPU_time < task.GPU_time:
                        rand_processor = np.random.choice(range(self.n_CPUs))
                    else:
                        rand_processor = np.random.choice(range(self.n_CPUs, self.n_CPUs + self.n_GPUs))           
                    self.workers[rand_processor].schedule_task(task, dag, self) 
                
                elif bmct_initial_allocation == "eft":
                    finish_times = list([p.earliest_finish_time(task, dag, self) for p in self.workers])
                    min_processor = np.argmin(finish_times) 
                    self.workers[min_processor].schedule_task(task, dag, self, finish_time=finish_times[min_processor])
                    
                elif bmct_initial_allocation == "random":
                    chosen_processor = np.random.choice(range(self.n_workers), p=relative_speeds)
                    self.workers[chosen_processor].schedule_task(task, dag, self)                    
                    
            # Compute average finish times for all tasks in batch across all processors.
            avg_fts = {}
            for task in batch:
                efts = []
                for p in self.workers:
                    if task.where_scheduled == p.ID:
                        efts.append(task.AFT)
                        continue
                    efts.append(p.earliest_finish_time(task, dag, self))
                avg_fts[task] = sum(efts) / self.n_workers                 
                
            # Sort tasks in batch in ascending order of their average finish time across all processors.
            batch_avgs = list(sorted(avg_fts, key=avg_fts.get))                        
            
            moved_task = True
            while moved_task:
                # Find processor m with the latest finish time.
                processor_finish_times = []
                for p in self.workers:
                    processor_finish_times.append(p.load[-1].AFT) if not p.idle else processor_finish_times.append(0)
                m = np.argmax(processor_finish_times)
                # Check all tasks on processor m in order of average finish time across all processors.
                moved_task = False
                for task in batch_avgs:
                    if task not in self.workers[m].load:
                        continue
                    # Estimate new finish time of all processors assuming that the task is scheduled there.
                    fts = []
                    for p in self.workers:
                        if p.ID == m:
                            fts.append(inf) 
                        elif p.idle:
                            fts.append(p.earliest_finish_time(task, dag, self))
                        else:
                            fts.append(max(p.earliest_finish_time(task, dag, self), p.load[-1].AFT))                      
                    # Find the processor with the earliest finish time.
                    j = np.argmin(fts)
                    # If finish time of processor j is before the finish time of processor m, then move the task there.
                    if fts[j] < processor_finish_times[m]:
                        self.workers[m].unschedule_task(task)
                        self.workers[j].schedule_task(task, dag, self)
                        moved_task = True
                    if moved_task:
                        break                    
            
        elif policy == "custom":
            if mapping is None:
                raise ValueError("Custom processor selection policy selected for schedule_batch but no mapping specified!")
            for task in batch:
                processor = mapping[task.ID]
                self.workers[processor].schedule_task(task, dag, self)                          
        else:
            raise ValueError('Unrecognised processor selection policy in schedule_batch!')
        
    def estimate_makespan(self, dag, batch, policy="eft", mapping=None, just_batch=False):
        """
        Estimate the new makespan after scheduling a batch of independent tasks.
        At the moment, basically just schedule the tasks in the batch then reset everything back to the initial state
        but in the future may try something else.
        
        Parameters
        ------------------------
        
        dag - DAG object (see Graph.py module)
        The DAG to which the tasks belongs.
              
        batch - list of Task objects
        The tasks to be scheduled.
        
        policy - string
        Describes the rule used for scheduling the tasks.
        Options:
            - "eft", use the classic Earliest Finish Time heuristic.
            - "met", use the classic Minimum Execution Time heuristic.
            - "bmct", use Sakellariou and Zhao's Balanced Minimum Completion Time heuristic.
            - "custom", expect a mapping {task : processor} to be specified and throw an error otherwise. 
                        
        mapping - dict, {task ID : processor ID}
        Which Worker each task should be scheduled on.
        
        just_batch - bool
        If True, return the estimated maximum finish time of all tasks in the batch only, else 
        return current makespan of entire DAG.      
        
        Returns
        ------------------------
        makespan - float
        The current makespan of the DAG.        
        """                                 
        self.schedule_batch(dag, batch, policy=policy, mapping=mapping)              
            
        # Compute the makespan.
        if just_batch:
            makespan = max(t.AFT for t in batch)
        else:
            makespan = dag.makespan(partial=True)
            
        # Revert back to the original state of the system.
        for task in batch:             
            p = self.workers[task.where_scheduled]
            p.unschedule_task(task)          
        
        return makespan
    
    def estimate_finish_times(self, dag, batch, policy="EFT", where_scheduled=False, cpu_samples=None, gpu_samples=None):
        """
        Estimate the finish times of all tasks in batch. 
        Effectively a helper function for HEFT_L function in Heuristics.py module.
        Assumes that all tasks in batch are independent and that they are ready to schedule,
        so in particular disregards any unscheduled parents.
        
        Parameters
        ------------------------
        
        dag - DAG object (see Graph.py module)
        The DAG to which the tasks belongs.
              
        batch - list of Task objects
        The tasks to be scheduled.
        
        policy - string
        Describes the rule used for scheduling the tasks.
        Options:
            - "eft", use the classic Earliest Finish Time heuristic.
                        
        where_scheduled - bool
        If True also return a dict describing which Worker each task is to be scheduled on.
        
        cpu_samples - None/int
        The number of CPU Workers to sample.
        
        gpu_samples - None/int
        The number of GPU Workers to sample.    
        
        Returns
        ------------------------
        makespan - float
        The current makespan of the DAG.  
        
        If where_scheduled:
        destinations - dict {task ID : (Worker ID, estimated finish time of task on Worker)}
        Where each task is expected to be scheduled.
        """ 
        
        if policy != "EFT":
            raise ValueError("Sorry, estimate_finish_times only works for EFT processor selection now (but will extend in future).")
            # TODO: add more indepedent task scheduling heuristics.
            
        if cpu_samples and cpu_samples > self.n_CPUs:
            raise ValueError("Error: tried to sample more CPUs than there actually are!")  
            
        if gpu_samples and gpu_samples > self.n_GPUs:
            raise ValueError("Error: tried to sample more GPUs than there actually are!") 
        
        if where_scheduled:
            destinations = {}
        
        finish_times = defaultdict(float)
        for task in batch:
            est_finish_times = []
            scheduled_predecessors = list(t for t in dag.DAG.predecessors(task) if t.scheduled)
            if cpu_samples:
                cpu_indices = np.random.choice(range(self.n_CPUs), cpu_samples, replace=False)
            if gpu_samples:
                gpu_indices = np.random.choice(range(self.n_CPUs, self.n_workers), gpu_samples, replace=False)
            for p in self.workers:
                if p.CPU and cpu_samples and p.ID not in cpu_indices:
                    est_finish_times.append(float('inf'))
                    continue
                if p.GPU and gpu_samples and p.ID not in gpu_indices:
                    est_finish_times.append(float('inf'))
                    continue                   
                
                est = max(t.AFT + self.comm_cost(t, task, t.where_scheduled, p.ID) 
                                    for t in scheduled_predecessors)
                processing_time = task.CPU_time if p.CPU else task.GPU_time 
                if p.idle:
                    est_finish_times.append(est + processing_time)
                else:        
                    found = False
                    prev_finish_time = 0
                    for t in p.load:
                        if t.AST < est:
                            prev_finish_time = t.AFT
                            continue
                        poss_start_time = max(prev_finish_time, est)
                        if poss_start_time + processing_time <= t.AST:
                            est_finish_times.append(poss_start_time + processing_time)
                            found = True
                            break
                        prev_finish_time = t.AFT                    
                    # No valid gap found.
                    if not found:
                        est_finish_times.append(max(p.load[-1].AFT, est) + processing_time)                    
                        
            # Select processor which minimizes child's finish time.
            chosen_processor = np.argmin(est_finish_times) 
            # Schedule the child so we can estimate the next child's finish times fairly. 
            self.workers[chosen_processor].schedule_task(task, dag, self, finish_time=est_finish_times[chosen_processor])
            finish_times[task.ID] = task.AFT                 
        
        # Unschedule all the child tasks...
        for task in batch:
            if where_scheduled:
                destinations[task.ID] = (task.where_scheduled, task.AFT)
            self.workers[task.where_scheduled].unschedule_task(task)
        
        if where_scheduled:
            return finish_times, destinations
        return finish_times     
        
        
    