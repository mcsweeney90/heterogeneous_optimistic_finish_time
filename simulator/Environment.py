#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:00:45 2018

Framework for describing a CPU-GPU environment.

@author: Neil Walton, Tom
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
        
        # Is a CPU unless stated otherwise.
        self.GPU = True if GPU else False
        self.CPU = not self.GPU   
        self.ID = ID   # Useful for debugging, etc.
        self.load = [] # Tasks scheduled on the processor.
        self.idle = True # True if no tasks currently scheduled on the processor. 
                
    def earliest_start_time(self, task, dag, platform, insertion=True, comm_estimates=None, comp_estimates=None):
        """
        Returns the estimated earliest start time for task on the processor.
        If insertion, uses the insertion-based policy - i.e., a task can be scheduled between two already scheduled tasks (if permitted by dependencies).
        """        
        
        if self.idle:   # If no tasks scheduled on processor...
            if task.entry: # If an entry task...
                return 0   
            else:
                return max(p.AFT + platform.comm_cost(p, task, p.where_scheduled, self.ID, estimates=comm_estimates) 
                                    for p in dag.DAG.predecessors(task))  
                
        # Find earliest time all task predecessors have finished and the task can theoretically start.     
        est = 0
        if not task.entry:                    
            predecessors = dag.DAG.predecessors(task) 
            est += max(p.AFT + platform.comm_cost(p, task, p.where_scheduled, self.ID, estimates=comm_estimates) 
                                    for p in predecessors)
            
        if comp_estimates:
            processing_time = comp_estimates["CPU"][task.ID] if self.CPU else comp_estimates["GPU"][task.ID]
        else:
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
        
    def earliest_finish_time(self, task, dag, platform, insertion=True, start_time=None, comm_estimates=None, comp_estimates=None): 
        """
        Returns the estimated earliest finish time for task on the processor.
        If start_time is not None, checks it is valid and, if so, takes this to be task's actual start time.
        """       
        if comp_estimates:
            processing_time = comp_estimates["CPU"][task.ID] if self.CPU else comp_estimates["GPU"][task.ID]
        else:
            processing_time = task.CPU_time if self.CPU else task.GPU_time 
        if start_time: 
            if not self.valid_start_time(task, start_time, dag, platform):
                raise ValueError('Invalid input start time! Processor ID: {}, task ID, attempted start time: {}'.format(self.ID, task.ID, start_time))
            return processing_time + start_time
        return processing_time + self.earliest_start_time(task, dag, platform, insertion=insertion, comm_estimates=comm_estimates, comp_estimates=comp_estimates)      

    def valid_start_time(self, task, start_time, dag, platform, insertion=True):
        """
        True if the input start time is possible (respects all dependencies, etc), False otherwise.
        """
        
        # Compute earliest possible start time, when all predecessors have completed and relevant data has been transferred. 
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
        Schedule the task on the processor. Tasks are stored in a list in ascending order of start time.
        If insertion, uses insertion-based scheduling policy. 
        If start_time is not None, checks if task can be executed at this time and, if so, schedules it then.
        If finish_time is not None, schedules it at that time. 
        NOTE: if finish_time, doesn't check that all task predecessors have been scheduled. (This is so we can do lookahead in e.g., platform.estimate_finish_times.) 
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
        """Unschedules the task on the processor."""
        # Remove task from the load and revert the processor to idle if necessary.
        self.load.remove(task)
        if not len(self.load):
            self.idle = True
        # Reset the task itself.    
        task.reset()                         
        
    def print_schedule(self, filepath=None):
        """Prints the tasks scheduled on the processor."""
        proc_type = "CPU" if self.CPU else "GPU"
        if filepath:
            print("Processor {}, {}: ".format(self.ID, proc_type), file=filepath)
            for t in self.load:
                type_info = " Task type: {},".format(t.type) if t.type else ""
                print("Task ID: {},{} AST = {}, AFT = {}.".format(t.ID, type_info, t.AST, t.AFT), file=filepath) 
        else:
            print("Processor {}, {}: ".format(self.ID, proc_type))
            for t in self.load:
                print("Task ID: {}, AST = {}, AFT = {}.".format(t.ID, t.AST, t.AFT))     
 
class Node:
    """          
    A node is basically just a collection of CPUs and GPUs.
    Inputs:
        - CPUs, integer, the number of CPUs.
        - GPUs, integer, the number of GPUs.
        - name, string, used to identify node.
        - communication, bool, can turn communication costs on and off.
        - adt, bool, can turn asynchronous data transfers on and off.
    """
    def __init__(self, CPUs, GPUs, name="generic", communication=True, adt=False):
        
        self.name = name  
        self.communication = communication         
        self.adt = adt
        
        # Workers.
        self.n_CPUs, self.n_GPUs = CPUs, GPUs 
        self.n_workers = self.n_CPUs + self.n_GPUs
        self.workers = []
        for i in range(self.n_CPUs):
            self.workers.append(Worker(ID=i))          
        for j in range(self.n_GPUs):
            self.workers.append(Worker(GPU=True, ID=self.n_CPUs + j))  
                           
    
    def print_info(self, filepath=None):
        """Print basic information about the Node."""
        # Print to screen.
        print("--------------------------------------------------------")
        print("NODE INFO")
        print("--------------------------------------------------------")
        print("Name: {}".format(self.name))
        print("{} CPUs, {} GPUs".format(self.n_CPUs, self.n_GPUs))
        print("Communication: {}".format(self.communication))
        print("Asynchronous data transfer: {}".format(self.adt))
        print("--------------------------------------------------------\n")
        
        # If filepath, also print to file.
        if filepath:
            print("--------------------------------------------------------", file=filepath)
            print("NODE INFO", file=filepath)
            print("--------------------------------------------------------", file=filepath)
            print("Name: {}".format(self.name), file=filepath)
            print("{} CPUs, {} GPUs".format(self.n_CPUs, self.n_GPUs), file=filepath)
            print("Communication: {}".format(self.communication), file=filepath) 
            print("Asynchronous data transfer: {}".format(self.adt), file=filepath)
            print("--------------------------------------------------------\n", file=filepath)            
       
    def where_scheduled(self, task):
        """Return the ID of the processor where task is scheduled, or None if it isn't."""
        for p in range(self.n_workers):
            if task in self.workers[p].load:
                return p
        return None       
    
    def reset(self):
        """ Resets some attributes to defaults so we can simulate the execution of another job. """
        for w in self.workers:
            w.load = []   
            w.idle = True 
            
    def print_schedule(self, name="", filepath=None):
        """Prints the current schedule for all processors."""
        if filepath:
            print("--------------------------------------------------------", file=filepath)
            print("{} SCHEDULE".format(name), file=filepath)
            print("--------------------------------------------------------", file=filepath)
            for w in self.workers:
                w.print_schedule(filepath=filepath)  
            makespan = max(w.load[-1].AFT for w in self.workers if w.load) 
            print("\n{} makespan: {}".format(name, makespan), file=filepath)            
            print("--------------------------------------------------------\n", file=filepath)
        else:
            print("--------------------------------------------------------")
            print("{} SCHEDULE".format(name))
            print("--------------------------------------------------------")
            for w in self.workers:
                w.print_schedule()  
            makespan = max(w.load[-1].AFT for w in self.workers if w.load) 
            print("\n{} makespan: {}".format(name, makespan))            
            print("--------------------------------------------------------\n")
    
    def valid_schedule(self, schedule, dag):
        """
        Inputs:
            - schedule, list of the form [(task, processor scheduled on, start time), ...].
        Returns True if schedule is valid (doesn't violate any task dependencies or 
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
    
    def evaluate_schedule(self, schedule, dag):
        """
        Inputs:
            - schedule, list of the form [[(task, processor scheduled on, start time), ...].
            - dag, the DAG the schedule corresponds to.        
        Checks that an input schedule is valid and, if so, returns its makespan.
        """
        # Check if it's a valid schedule.
        if not self.valid_schedule(schedule, dag):
            raise ValueError('Schedule you are trying to evaluate is invalid! Check it again...') 
        
        # Compute makespan.  
        # Note that here we assume that task.AST, task.AFT aren't set so can't just use dag.makespan().
        # Assume that task.exit is set so only need to check the exit tasks. If not, comment/remove "if not t.exit" statement in the loop.
        mkspan =  0.0
        for t, p, st in schedule:
            # If task.exit is not set, comment/remove next statement.
            if not t.exit:
                continue
            finish_time = st + t.CPU_time if p < self.n_CPUs else st + t.GPU_time
            mkspan = max(finish_time, mkspan)
        return mkspan
        
    def comm_cost(self, parent, child, source_id, target_id, estimates=None):    
        """
        Communication cost between parent and child tasks (assumes one exists).         
        """        
        if source_id == target_id:
            return 0 
        if not self.communication:
            return 0          
        
        source_type = "G" if source_id > self.n_CPUs - 1 else "C"
        target_type = "G" if target_id > self.n_CPUs - 1 else "C"
        
        if estimates:
            return estimates["{}".format(source_type + target_type)][parent.ID][child.ID]
        
        if self.adt:
            parent_comp_time = parent.GPU_time if source_type == "G" else parent.CPU_time
            return max(parent.comm_costs["{}".format(source_type + target_type)][child.ID] - parent_comp_time, 0)
        
        return parent.comm_costs["{}".format(source_type + target_type)][child.ID]    
                
    
    def approximate_comm_cost(self, parent, child, weighting="HEFT", r_bar=None):  
        """
        Approximate communication cost of the edge from a parent task to a child task. Used in HEFT and similar heuristics. 
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
                
        elif weighting == "WM-I":
            A, B = parent.acceleration_ratio, child.acceleration_ratio
            c_bar = self.n_CPUs * (self.n_CPUs - 1) * parent.comm_costs["CC"][child.ID] 
            c_bar += self.n_CPUs * B * self.n_GPUs * parent.comm_costs["CG"][child.ID]
            c_bar += A * self.n_GPUs * self.n_CPUs * parent.comm_costs["GC"][child.ID]
            c_bar += A * self.n_GPUs * B * (self.n_GPUs - 1) * parent.comm_costs["GG"][child.ID]
            c_bar /= ((self.n_CPUs + A * self.n_GPUs) * (self.n_CPUs + B * self.n_GPUs))
            return c_bar            
        
        elif weighting == "WM-II":
            if not r_bar:
                raise ValueError("Tried to use approximate_comm_cost with weighting == WM-II but haven't entered the mean acceleration ratio!")
            r = r_bar
            c_bar = self.n_CPUs * (self.n_CPUs - 1) * parent.comm_costs["CC"][child.ID] 
            c_bar += self.n_CPUs * r * self.n_GPUs * parent.comm_costs["CG"][child.ID]
            c_bar += r * self.n_GPUs * self.n_CPUs * parent.comm_costs["GC"][child.ID]
            c_bar += r * self.n_GPUs * r * (self.n_GPUs - 1) * parent.comm_costs["GG"][child.ID]
            c_bar /= ((self.n_CPUs + r * self.n_GPUs)**2)
            return c_bar 
            
        elif weighting == "PS" or weighting == "ps" or weighting == "diff" or weighting == "SFB" or weighting == "sfb" or weighting == "EPS": 
            return 0
        elif weighting[:2] == "EW":
            return 0        
        elif weighting == "CC-I" or weighting == "CC-II" or weighting == "CC-III":
            return 0
        
        raise ValueError('No weighting (e.g., "mean" or "median") specified for approximate_comm_cost.')
        
    def fastest_processor_of_type(self, task, dag, gpu=False):
        """
        Returns (ID of the processor of the given type which will complete the task at the earliest time, estimated time it will start).
        """        
        if gpu:
            finish_times = [p.earliest_finish_time(task, dag, self) for p in self.workers if p.GPU]
            return np.argmin(finish_times) + self.n_CPUs, np.amin(finish_times) - task.GPU_time           
        finish_times = [p.earliest_finish_time(task, dag, self) for p in self.workers if p.CPU]
        return np.argmin(finish_times), np.amin(finish_times) - task.CPU_time    
        
    def schedule_batch(self, dag, batch, policy="eft", bmct_initial_allocation="met", mapping=None, batch_sort=None):
        """
        Schedule an independent batch of the tasks in the DAG according to some processor selection policy.
        Inputs:
            - policy describes how we schedule the tasks. Default is "eft" which corresponds to scheduling each task on the processor estimated
              to complete it at the earliest time. 
              If policy == "eft" use the classic Earliest Finish Time heuristic.
              If policy == "met" then use the classic Minimum Execution Time heuristic. 
              If policy == "bmct" then we use Sakellariou and Zhao's Balanced Minimum Completion Time. 
              If policy == "custom" then we expect a mapping to be specified and throw an error otherwise.
            - mapping is a dict {task ID : processor ID} that describes where we want to schedule each task.
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
                # Schedule all tasks on their fastest processor initially (classic Minimum Execution Time heuristic). No suggestion on how to break ties, so do it randomly.
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
        Estimate the new makespan resulting after scheduling a batch of independent tasks from dag on platform.        
        At the moment, basically just schedule the tasks in the batch then reset everything back to the initial state but in the future may 
        try something else.
        
        Inputs:
            - policy describes how we schedule the tasks. Default is "eft" which corresponds to scheduling each task on the processor estimated
              to complete it at the earliest time. If heuristic == "custom" then we expect a mapping to be specified and throw an error otherwise.
            - mapping is a dict {task ID : processor ID} that describes where we want to schedule each task.
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
        Assumes that all tasks in batch are independent and that they are ready to schedule (so disregards any unscheduled parents).
        Helper function for HEFT_L (HEFT with lookahead).
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
        

# TODO: Working only at node-level now but may develop a Cluster class later.
class Cluster:
    def __init__(self, nodes, interconnect):
        """Cluster is a collection of nodes."""
        self.nodes = nodes
        self.interconnect = interconnect
        
        
    