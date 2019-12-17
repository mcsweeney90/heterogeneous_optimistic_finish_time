#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:11:27 2018

Implementations of classic static listing heuristics, as well as Heterogeneous Optimistic Finish Time (HOFT)
and a few experimental ones that I may pursue further in the future.

@author: Tom
"""

import numpy as np
import networkx as nx
from collections import defaultdict 

####################################################################################################
    
"""Classic heuristics.""" 
    
####################################################################################################   
    
def HEFT(dag, platform, task_list=None, weighting="mean", return_schedule=False, schedule_dest=None):
    """
    Heterogeneous Earliest Finish Time (Topcuoglu, Hariri and Wu, 2002).
    If return_schedule == True, returns the schedule explictly as a dict {task ID : processor ID}.
    """   
    
    if return_schedule:
        pi = defaultdict(int)  
    
    # Get a list of the tasks in nonincreasing order of upward rank.
    if task_list:
        priority_list = task_list
    else:
        priority_list = dag.sort_by_upward_rank(platform, weighting=weighting)   
    
    # Schedule the tasks.
    for t in priority_list:    
        
        # Compute the finish time on all processors, identify the processor which minimizes the finish time (with ties broken consistently by np.argmin).   
        finish_times = list([p.earliest_finish_time(t, dag, platform) for p in platform.workers])
        min_processor = np.argmin(finish_times)                       
        
        # Schedule the task on the chosen processor. 
        platform.workers[min_processor].schedule_task(t, dag, platform, finish_time=finish_times[min_processor])
        
        if return_schedule:
            pi[t] = min_processor
                    
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="HEFT", filepath=schedule_dest)
        
    # Makespan is the maximum AFT of all the exit tasks. 
    mkspan = dag.makespan() 
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset() 
    
    if return_schedule:
        return mkspan, pi    
    return mkspan 

def HBMCT(dag, platform, task_list=None, batch_policy="bmct", bmct_initial_allocation="met", group_sort=None, return_schedule=False, schedule_dest=None):
    """
    Hybrid Balanced Minimum Completion Time (Sakellariou and Zhao, 2004).
    """
    
    if return_schedule:
        pi = defaultdict(int) 
    
    # Get the priority list of tasks.
    if task_list:
        priority_list = task_list
    else:        
        priority_list = dag.sort_by_upward_rank(platform) 
                
    if schedule_dest:
        task_order = []  
                        
    # Find the groups and schedule them.
    G = []
    for task in priority_list:
        if any(p in G for p in dag.DAG.predecessors(task)):
            platform.schedule_batch(dag, G, policy=batch_policy, bmct_initial_allocation=bmct_initial_allocation, batch_sort=group_sort)
            if schedule_dest:
                for t in G:            
                    task_order.append(t.ID)
            G = []
        G.append(task)
    if len(G):
        platform.schedule_batch(dag, G, policy=batch_policy, bmct_initial_allocation=bmct_initial_allocation, batch_sort=group_sort)
        if schedule_dest:
            for task in G:            
                task_order.append(task.ID)
    
    # If schedule_dest, save the priority list and schedule.           
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="HBMCT", filepath=schedule_dest)
        
    if return_schedule:
        for t in priority_list:
            pi[t] = t.where_scheduled
        
    # Compute makespan.       
    mkspan = dag.makespan()
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
    
    if return_schedule:
        return mkspan, pi 
    return mkspan   
    
def PEFT(dag, platform, priority_list=None, return_schedule=False, schedule_dest=None):
    """
    Predict Earliest Finish Time (Arabnejad and Barbosa, 2014).
    Notes:
        - The suggested task prioritization phase doesn't always respect the precedence constraints.
          Rank_oct(t) = sum(OCT(t) / P). Alternatives: use minimum OFT instead, or use suggested rank_oct as node weight and
          compute upward ranks.
    """
    
    if return_schedule:
        pi = defaultdict(int) 
    
    OCT = dag.optimistic_cost_table(platform)   
    
    if not priority_list:        
        task_ranks = {}
        # Compute the task ranks.
        for task in dag.DAG:
            # Quick hack to handle the case when we have a single artificial cost zero exit task.
            if task.exit and task.CPU_time == 0:
                task_ranks[task] = -1
                continue
#            task_ranks[task] = sum(OCT[task].values()) / platform.n_workers
            task_ranks[task] = min(OCT[task].values()) # Use min values instead...             
        priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))
    
    # Schedule all tasks to the processor that minimizes the sum of the OCT and EFT.
    for task in priority_list:
        OEFT = [p.earliest_finish_time(task, dag, platform) + OCT[task][p.ID] for p in platform.workers]
        p = np.argmin(OEFT)
        platform.workers[p].schedule_task(task, dag, platform)
        if return_schedule:
            pi[task] = p
        
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="PEFT", filepath=schedule_dest)
        
    # Makespan is the maximum AFT of all the exit tasks.        
    mkspan = dag.makespan()
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()  
    
    if return_schedule:
        return mkspan, pi        
    return mkspan    

def PETS(dag, platform, selection_heuristic="eft", return_schedule=False, schedule_dest=None):
    """
    Performance Effective Task Scheduling (Ilavarasan and Thambidurai, 2007). 
    """
    
    if return_schedule:
        pi = defaultdict(int)     
    if schedule_dest:
        task_order = [] 
              
    # Sort DAG into levels. 
    top_sort = list(nx.topological_sort(dag.DAG))
    level = {}
    for task in top_sort:
        if task.entry:
            level[task] = 0
            continue
        ell = max(level[p] + 1 for p in dag.DAG.predecessors(task))
        level[task] = ell    
    levels = defaultdict(list)
    for task in dag.DAG:
        levels[level[task]].append(task)   
    n_levels = max(levels)   
    
    rank = {}
    for current_level in range(n_levels + 1):       
        # Go through the DAG level by level and compute the priorities. 
        acc = {}
        for task in levels[current_level]:
            acc[task] = task.approximate_execution_cost(platform)
            rpt = max(rank[p] for p in dag.DAG.predecessors(task)) if not task.entry else 0      
            dtc = sum(platform.approximate_comm_cost(task, c) for c in dag.DAG.successors(task))             
            # Rank is just the sum of all three factors.                     
            rank[task] = round(acc[task] + dtc + rpt)            
            
        # Sort all tasks at current level by rank (largest first) with ties broken by acc (smallest first).
        priority_list = list(sorted(acc, key=acc.get))
        priority_list = list(reversed(sorted(priority_list, key=lambda t: rank[t])))   
        
        # Schedule all tasks in current level.
        platform.schedule_batch(dag, priority_list, policy=selection_heuristic)
        if schedule_dest:
            task_order += list(t.ID for t in priority_list)  
        if return_schedule:
            for task in priority_list:
                pi[task] = task.where_scheduled    
                
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in task_order:
            print(t, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="PETS", filepath=schedule_dest)
    
    # Compute makespan.      
    mkspan = dag.makespan()
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
    
    if return_schedule:
        return mkspan, pi 
    return mkspan

def HCPT(dag, platform, return_schedule=False, schedule_dest=None):
    """
    Heterogeneous Critical Parent Trees (Hagras and Janecek, 2003).
    Notes:
        - Assumes single entry and exit tasks.
    """  
    
    if return_schedule:
        pi = defaultdict(int) 
    
    AEST = {}
    forward_traversal = list(nx.topological_sort(dag.DAG))
    for t in forward_traversal:
        AEST[t] = 0 
        try:
            AEST[t] += max(AEST[p] + p.approximate_execution_cost(platform) + platform.approximate_comm_cost(parent=p, child=t) for p in dag.DAG.predecessors(t))
        except ValueError:
            pass 
        
    ALST = {}
    backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
    for t in backward_traversal:
        if t.exit:
            ALST[t] = AEST[t]
            continue
        try:
            ALST[t] = min(ALST[s] - platform.approximate_comm_cost(parent=t, child=s) for s in dag.DAG.successors(t)) - t.approximate_execution_cost(platform)
        except ValueError:
            pass
    
    S = []
    for t in forward_traversal:
        if AEST[t] == ALST[t]:
            S.append(t) 
            
    L = []
    while len(S):
        top = S[0]
        if top.entry:
            L.append(top)
            S = S[1:]
            continue
        parent = None
        for p in dag.DAG.predecessors(top):
            if p not in L:
                parent = p
                break
        if parent is not None:
            S = [parent] + S
        else:
            L.append(top)
            S = S[1:]
    
    for t in L:
        finish_times = list([p.earliest_finish_time(t, dag, platform) for p in platform.workers])
        min_processor = np.argmin(finish_times)   
        platform.workers[min_processor].schedule_task(t, dag, platform, finish_time=finish_times[min_processor])  
        if return_schedule:
            pi[t] = min_processor
    
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in L:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="HCPT", filepath=schedule_dest)
        
    # Makespan is the maximum AFT of all the exit tasks. 
    mkspan = dag.makespan() 
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
    
    if return_schedule:
        return mkspan, pi
    return mkspan

def HEFT_L(dag, platform, weighted_average=False, child_sampling_policy=None, cpu_samples=None, gpu_samples=None, schedule_dest=None):
    """ 
    HEFT with lookahead (Bittencourt, Sakellariou and Madeira, 2010).
    If weighted_average, use a weighted average of the EFTs of the child tasks rather than the minimum.
    Notes:
        - Didn't implement the priority list change version because original authors suggest that it isn't useful.
    """
    
    # Rank the tasks.
    priority_list, task_ranks = dag.sort_by_upward_rank(platform, return_rank_values=True)   
      
    # Schedule the tasks according to their children's estimated finish times.
    for task in priority_list:
        # If exit task, just assign as in HEFT.
        if task.exit:
            min_processor = np.argmin([p.earliest_finish_time(task, dag, platform) for p in platform.workers])           
            platform.workers[min_processor].schedule_task(task, dag, platform) 
            continue       
        
        # Find children.
        children = list(c for c in dag.DAG.successors(task))
        if len(children) > 1:
            if child_sampling_policy == "1R":
                children = [np.random.choice(children)]
            elif child_sampling_policy == "1P":
                children = [max(children, key = lambda c : task_ranks[c])]
            elif child_sampling_policy == "2R":
                children = list(np.random.choice(children, 2, replace=False))
            elif child_sampling_policy == "2P":
                max_child = max(children, key = lambda c : task_ranks[c])    
                children.remove(max_child)
                second_child = max(children, key = lambda c : task_ranks[c]) 
                children = [max_child, second_child]
            else:
                children = list(reversed(sorted(children, key = lambda c : task_ranks[c])))         
        
        # Estimate the finish times of children if they are scheduled as in HEFT.
        processor_estimates = []
        for p in platform.workers:
            p.schedule_task(task, dag, platform)
            child_finish_times = platform.estimate_finish_times(dag, batch=children, cpu_samples=cpu_samples, gpu_samples=gpu_samples)            
            if weighted_average:
                child_rankings = sum(task_ranks[c] for c in children)                
                eft_w = sum(task_ranks[c] * child_finish_times[c.ID] for c in children) / child_rankings if child_rankings else child_finish_times[children[0]] 
                processor_estimates.append(eft_w)                
            else: 
                processor_estimates.append(max(child_finish_times.values()))
            p.unschedule_task(task)
        # Select the processor which minimizes either the maximum or weighted average of the child finish times.
        chosen_processor = np.argmin(processor_estimates)
        platform.workers[chosen_processor].schedule_task(task, dag, platform)
    
    # If verbose, print the schedule (i.e., the load of all the processors).
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        w_avg = " with weighted averaging" if weighted_average else ""
        platform.print_schedule(name="Lookahead HEFT{}".format(w_avg), filepath=schedule_dest)
        
    # Makespan is the maximum AFT of all the exit tasks.        
    mkspan = dag.makespan()
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
        
    return mkspan

####################################################################################################
    
"""Heterogeneous Optimistic Finish Time (HOFT).""" 
    
####################################################################################################  

def OFT_priorities(dag, platform, selection="HOFT", table=None):
    """
    Returns a priority list of the tasks for use in any listing scheduler.
    (Put in a separate function because otherwise would need to significantly rewrite Graph.sort_by_upward_rank.)
    """
    
    if table:
        OFT = table
    else:
        OFT = dag.optimistic_finish_times()      
        
    backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
    task_ranks = {}
    for t in backward_traversal:
        
        if selection == "OFT-I":   
            task_ranks[t] = t.CPU_time if OFT[t]["C"] < OFT[t]["G"] else t.GPU_time
            t_proc = "C" if OFT[t]["C"] < OFT[t]["G"] else "G"
            try:
                child_costs = []
                for s in dag.DAG.successors(t):
                    s_proc = "C" if OFT[s]["C"] < OFT[s]["G"] else "G"
                    if t_proc == "C" and s_proc == "C":
                        d = (platform.n_CPUs - 1) / platform.n_CPUs
                    elif (t_proc == "C" and s_proc == "G") or (t_proc == "G" and s_proc == "C"):
                        d = 1
                    else:
                        d = (platform.n_GPUs - 1) / platform.n_GPUs
                    child_costs.append(task_ranks[s] + d * t.comm_costs["{}".format(t_proc + s_proc)][s.ID])
                task_ranks[t] += max(child_costs)
            except ValueError:
                pass 
        else:
            if selection == "OFT-II":
                task_ranks[t] = (platform.n_CPUs * OFT[t]["C"] + platform.n_GPUs * OFT[t]["G"]) / platform.n_workers
            elif selection == "OFT-III":
                w = platform.n_CPUs + t.acceleration_ratio * platform.n_GPUs
                task_ranks[t] = (platform.n_CPUs * OFT[t]["C"] + t.acceleration_ratio * platform.n_GPUs * OFT[t]["G"]) / w
            elif selection == "OFT-IV":
                task_ranks[t] = max(list(OFT[t].values()))
            elif selection == "OFT-V":
                task_ranks[t] = min(list(OFT[t].values()))
            elif selection == "HOFT":
                task_ranks[t] = max(list(OFT[t].values())) / min(list(OFT[t].values()))
            try:
                task_ranks[t] += max(task_ranks[s] for s in dag.DAG.successors(t))
            except ValueError:
                pass        
    
    priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))      
    return priority_list 
    
    
def HOFT(dag, platform, table=None, task_list=None, policy="HOFT", schedule_dest=None):
    """
    Heterogeneous Optimistic Finish Time (HOFT) heuristic.
    """
    
    if table:
        OFT = table
    else:
        OFT = dag.optimistic_finish_times() 
    
    if task_list:
        priority_list = task_list
    else:
        priority_list = OFT_priorities(dag, platform, table=OFT)         
        
    if policy == "CP": # Compute the critical path.        
        d = {}
        d["CC"], d["GG"] = 0, 0
        d["CG"], d["GC"] = 1, 1                 
        best, final_task = 0, None
        for t in dag.DAG:
            if not t.exit:
                continue
            ft = min(OFT[t]["C"], OFT[t]["G"])
            if ft > best:
                final_task = t
                final_proc = "C" if ft == OFT[t]["C"] else "G"
                final_exec_time = final_task.CPU_time if final_proc == "C" else final_task.GPU_time
                best = ft        
        # Find all CP tasks.
        cp = {final_task.ID}
        while True:
            if final_task.entry:
                break
            for p in dag.DAG.predecessors(final_task):
                if OFT[p]["C"] + d["C{}".format(final_proc)] * p.comm_costs["C{}".format(final_proc)][final_task.ID] + final_exec_time  == OFT[final_task][final_proc]:
                    cp.add(p.ID)
                    final_task, final_proc, final_exec_time = p, "C", p.CPU_time
                    break
                elif OFT[p]["G"] + d["G{}".format(final_proc)] * p.comm_costs["G{}".format(final_proc)][final_task.ID] + final_exec_time == OFT[final_task][final_proc]:
                    cp.add(p.ID)
                    final_task, final_proc, final_exec_time = p, "G", p.GPU_time
                    break            
    
    for task in priority_list:
        
        if policy == "EFT":
            finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
            min_p = np.argmin(finish_times)       
            platform.workers[min_p].schedule_task(task, dag, platform, finish_time=finish_times[min_p])
        
        elif policy == "FIXED":       
            gpu = True if OFT[task]["G"] < OFT[task]["C"] else False 
            if gpu:
                finish_times = [p.earliest_finish_time(task, dag, platform) for p in platform.workers if p.GPU]
            else:
                finish_times = [p.earliest_finish_time(task, dag, platform) for p in platform.workers if p.CPU]
            st = platform.n_CPUs if gpu else 0
            min_p = st + np.argmin(finish_times)
            platform.workers[min_p].schedule_task(task, dag, platform, finish_time=finish_times[min_p - st])
        
        elif policy == "SUM":
            OEFT = [OFT[task]["C"]] * platform.n_CPUs + [OFT[task]["G"]] * platform.n_GPUs
            for i, p in enumerate(platform.workers):
                OEFT[i] += p.earliest_finish_time(task, dag, platform)             
            chosen_worker = np.argmin(OEFT)
            platform.workers[chosen_worker].schedule_task(task, dag, platform)   
        
        elif policy == "CP":
            if task.ID in cp:
                if task.entry:
                    finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
                    cp_p = np.argmin(finish_times)       
                    platform.workers[cp_p].schedule_task(task, dag, platform, finish_time=finish_times[cp_p]) 
                else:
                    platform.workers[cp_p].schedule_task(task, dag, platform) 
            else:
                finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
                min_p = np.argmin(finish_times)       
                platform.workers[min_p].schedule_task(task, dag, platform, finish_time=finish_times[min_p])     
                
        elif policy == "HOFT":
            finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
            min_p = np.argmin(finish_times)       
            min_type = "C" if min_p < platform.n_CPUs else "G"
            fastest_type = "C" if task.CPU_time < task.GPU_time else "G" 
            if min_type == fastest_type:
                platform.workers[min_p].schedule_task(task, dag, platform, finish_time=finish_times[min_p]) 
            else:
                fastest_p = np.argmin(finish_times[:platform.n_CPUs]) if min_type == "G" else platform.n_CPUs + np.argmin(finish_times[platform.n_CPUs:]) 
                saving = finish_times[fastest_p] - finish_times[min_p]
                # Estimate the costs we expect to incur by scheduling task on min_p and fastest_p.
                min_costs, fastest_costs = 0, 0
                for s in dag.DAG.successors(task):
                    s_p = "C" if OFT[s]["C"] < OFT[s]["G"] else "G" # Expectation of where child scheduled based on OFT.
                    exec_time = s.CPU_time if s_p == "C" else s.GPU_time
                    min_costs = max(exec_time + task.comm_costs["{}".format(min_type + s_p)][s.ID], min_costs)
                    fastest_costs = max(exec_time + task.comm_costs["{}".format(fastest_type + s_p)][s.ID], fastest_costs)                
                if saving > (min_costs - fastest_costs):
                    platform.workers[min_p].schedule_task(task, dag, platform, finish_time=finish_times[min_p])
                else:
                    platform.workers[fastest_p].schedule_task(task, dag, platform, finish_time=finish_times[fastest_p])       
                       
    # If schedule_dest, print the schedule (i.e., the load of all the processors).
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="HOFT", filepath=schedule_dest) 
        
    # Makespan is the maximum AFT of all the exit tasks.        
    mkspan = dag.makespan()
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
        
    return mkspan  

####################################################################################################
    
"""Experimental/abandoned ideas. Included here just in case I ever decide to go back to them.""" 
    
####################################################################################################     

def level_scheduler(dag, platform, level_sort="HEFT", selection_heuristic="eft", schedule_dest=None):
    """
    A generic level-based scheduling framework.
    level_sort : Heuristic for prioritizing tasks in each level.
    selection_heuristic : Processor selection heuristic.
    """
    
    if schedule_dest:
        task_order = [] 
              
    # Sort DAG into levels. This may not be the most efficient way to do it...
    top_sort = list(nx.topological_sort(dag.DAG))
    level = {}
    if selection_heuristic == "eft":    
        weight = {}
    for task in top_sort:
        if task.entry:
            level[task] = 0
            continue
        ell = max(level[p] + 1 for p in dag.DAG.predecessors(task))
        level[task] = ell    
    levels = defaultdict(list)
    for task in dag.DAG:
        levels[level[task]].append(task)
        if selection_heuristic == "eft": 
            weight[task.ID] = task.approximate_execution_cost(platform, weighting=level_sort) 
    n_levels = max(levels)     
    
    for current_level in range(n_levels + 1):
        if selection_heuristic == "eft": 
            ranked_level = list(reversed(sorted(levels[current_level], key=lambda t : weight[t.ID])))
        else:
            ranked_level = levels[current_level]
        if schedule_dest:
            task_order += list(t.ID for t in ranked_level)
        platform.schedule_batch(dag, ranked_level, policy=selection_heuristic)        
    
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in task_order:
            print(t, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="Level scheduler", filepath=schedule_dest)
    
    # Compute makespan.      
    mkspan = dag.makespan()
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
        
    return mkspan

def stochastic_list_scheduler(dag, platform, task_list=None, policy="R-I", schedule_dest=None):
    """
    Stochastic processor selection methods for any input task list.  
    Obviously expected to do badly, the idea was that it might be useful for a kind of Monte Carlo sampling lookahead
    but didn't pursue that very far.
    """     
    
    # Get a list of the tasks in nonincreasing order of upward rank.
    if task_list:
        priority_list = task_list
    else:
        priority_list = dag.sort_by_upward_rank(platform) 
    
    if policy == "R-II" or policy == "R-IV":        
        r = np.mean(list(t.acceleration_ratio for t in dag.DAG))
        
    for task in priority_list:
        if policy == "R-I":
            chosen_processor = np.random.choice(range(platform.n_workers))
    
        elif policy == "R-II" or policy == "R-III":
            if policy == "R-III":
                r = task.acceleration_ratio
            s = platform.n_CPUs + r * platform.n_GPUs 
            relative_speeds = [1 / s] * platform.n_CPUs + [r / s] * platform.n_GPUs
            chosen_processor = np.random.choice(range(platform.n_workers), p=relative_speeds)           
            
        elif policy == "R-IV" or policy == "R-V":
            cpu_finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers if p.CPU])
            gpu_finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers if p.GPU])
            if policy == "R-V":
                r = task.acceleration_ratio
            w = platform.n_CPUs + r * platform.n_GPUs
            aff = np.random.choice(["CPU", "GPU"], p=(platform.n_CPUs / w, (r * platform.n_GPUs) / w))            
            # Now choose the specific worker.
            finish_times = cpu_finish_times if aff == "CPU" else gpu_finish_times
            st = platform.n_CPUs if aff == "GPU" else 0
            chosen_processor = st + np.argmin(finish_times)         
            
        elif policy == "R-VI":            
            finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
            m = max(finish_times)
            relative_finish_times = list([m - f for f in finish_times])
            # Normalize... 
            n = sum(relative_finish_times)
            relative_finish_times = list([f/n for f in relative_finish_times])
            chosen_processor = np.random.choice(range(platform.n_workers), p=relative_finish_times)             
            
        elif policy == "R-VII":
            finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
            p1 = np.argmin(finish_times)
            eft1 = finish_times[p1]
            finish_times[p1] = float('inf')
            p2 = np.argmin(finish_times)
            eft2 = finish_times[p2]
            
            # Now use the two EFT values to select the chosen processor.
            # Want q = 0.5 if eft2 - eft1 > eft1. Want relatively big q if gap is small and small q if gap is large.
            d = eft2 - eft1
            q = 0.5 if 2 * d >= eft1 else d / eft1
            chosen_processor = np.random.choice([p1, p2], p=(0.5 + q, 0.5 - q))         
            
        # Schedule the task on the chosen worker.  
        platform.workers[chosen_processor].schedule_task(task, dag, platform)                 
        
    # If verbose, print the schedule (i.e., the load of all the processors).
    if schedule_dest: 
        platform.print_schedule(name="Stochastic list scheduler with {} selection policy".format(policy), filepath=schedule_dest)
     
    mkspan = dag.makespan() 
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
    
    return mkspan    


def HEFT_FL(dag, platform, verbose=False, schedule_dest=False):
    """ 
    Extension of lookahead HEFT from Bittencourt, Sakellariou and Madeira that looks at a task's grandchildren.
    Early results weren't promising so it isn't as customizable as HEFT_L above and I don't intend to pursue any further.
    """
    
    # Rank the tasks.
    priority_list, task_ranks = dag.sort_by_upward_rank(platform, return_rank_values=True) 
    
    for task in priority_list:
        # If exit task, just assign as in HEFT.
        if task.exit:
            min_processor = np.argmin([p.earliest_finish_time(task, dag, platform) for p in platform.workers])           
            platform.workers[min_processor].schedule_task(task, dag, platform) 
            continue 
                        
        # Select children.
        children = list(c for c in dag.DAG.successors(task))
        if len(children) > 1:
            max_child = max(children, key = lambda c : task_ranks[c])    
            children.remove(max_child)
            second_child = max(children, key = lambda c : task_ranks[c]) 
            children = [max_child, second_child]         
            
        # Select grandchildren.
        grandchildren = []
        for c in children:
            if c.exit:
                continue
            kids = list(k for k in dag.DAG.successors(c))
            if len(kids) > 1:
                max_child = max(kids, key = lambda k : task_ranks[k])    
                if max_child not in grandchildren:                    
                    grandchildren.append(max_child)
                kids.remove(max_child)
                second_child = max(kids, key = lambda k : task_ranks[k]) 
                if second_child not in grandchildren:
                    grandchildren.append(second_child) 
            else:
                if kids[0] not in grandchildren:
                    grandchildren.append(kids[0])
                    
        if not len(grandchildren):
            min_processor = np.argmin([p.earliest_finish_time(task, dag, platform) for p in platform.workers])           
            platform.workers[min_processor].schedule_task(task, dag, platform) 
            continue     
        
        processor_estimates = []
        for p in platform.workers:
            p.schedule_task(task, dag, platform)      
            # Schedule the children.                      
            _, child_destinations = platform.estimate_finish_times(dag, batch=children, where_scheduled=True)
            for c in children:
                platform.workers[child_destinations[c.ID][0]].schedule_task(c, dag, platform, finish_time=child_destinations[c.ID][1])
            # Estimate the grandchild finish times.
            grandchild_finish_times = platform.estimate_finish_times(dag, batch=grandchildren) 
            # Compute weighted average of the grandchild finish times.
            grandchild_rankings = sum(task_ranks[g] for g in grandchildren)                
            eft_w = sum(task_ranks[g] * grandchild_finish_times[g.ID] for g in grandchildren) / grandchild_rankings if grandchild_rankings else grandchild_finish_times[grandchildren[0]] 
            processor_estimates.append(eft_w)        
            # Unschedule the task and its children.
            p.unschedule_task(task)
            for c in children:
                platform.workers[child_destinations[c.ID][0]].unschedule_task(c)
            
        # Select the processor which minimizes the grandchild finish times.
        chosen_processor = np.argmin(processor_estimates)
        platform.workers[chosen_processor].schedule_task(task, dag, platform)
    
    # If verbose, print the schedule (i.e., the load of all the processors).
    if verbose: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="Sampling-based lookahead HEFT", filepath=schedule_dest)        
        
    # Makespan is the maximum AFT of all the exit tasks.        
    mkspan = dag.makespan()
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
        
    return mkspan  

    
def LB_priorities(dag, platform, selection="LB-I"):
    """
    Helper function for LB_heuristic below.
    "LB" = "Load Balancing". Priorities determined based on acceleration ratio and how many tasks "should be" assigned to CPU/GPU 
    according to an "optimal" load balance. 
    Notes:
        - Put in a separate function because otherwise would need to significantly rewrite Graph.sort_by_upward_rank.
    """
    
    if not selection == "LB-III":
        acc_ratios = list(t.acceleration_ratio for t in dag.DAG)
        r_bar = np.mean(acc_ratios)
    
    if selection == "LB-I":
        num_tasks_on_cpu = int(round(dag.num_tasks * platform.n_CPUs / (platform.n_CPUs + r_bar * platform.n_GPUs)))
        sorted(acc_ratios)
        thresh = acc_ratios[num_tasks_on_cpu] 
          
    where_scheduled = defaultdict(str)
    for t in dag.DAG:
        if selection == "LB-I":
            where_scheduled[t.ID] = "CPU" if t.acceleration_ratio < thresh else "GPU" 
        else:
            r = r_bar if selection == "LB-II" else t.acceleration_ratio
            w = platform.n_CPUs + r * platform.n_GPUs
            where_scheduled[t.ID] = np.random.choice(["CPU", "GPU"], p=[platform.n_CPUs / w, (r * platform.n_GPUs) / w])            
    
    # Compute upward rank.
    backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
    task_ranks = {}
    for t in backward_traversal:            
        task_ranks[t] = t.CPU_time if where_scheduled[t.ID] == "CPU" else t.GPU_time 
        try:
            child_comm_costs = {}
            for s in dag.DAG.successors(t):
                child_comm_costs[s.ID] = platform.comm_cost()
                if where_scheduled[t.ID] == "CPU" and where_scheduled[s.ID] == "CPU":
                    A = (platform.n_CPUs - 1) / platform.n_CPUs
                    child_comm_costs[s.ID] = A * t.comm_costs["CC"][s.ID]
                elif where_scheduled[t.ID] == "CPU" and where_scheduled[s.ID] == "GPU":
                    child_comm_costs[s.ID] = t.comm_costs["CG"][s.ID]
                elif where_scheduled[t.ID] == "GPU" and where_scheduled[s.ID] == "CPU":
                    child_comm_costs[s.ID] = t.comm_costs["GC"][s.ID]
                else: # Both GPU.
                    A = (platform.n_GPUs - 1) / platform.n_GPUs
                    child_comm_costs[s.ID] = A * t.comm_costs["GG"][s.ID]
            task_ranks[t] += max(child_comm_costs[s.ID] + task_ranks[s] for s in dag.DAG.successors(t))
        except ValueError:
            pass  
    # Create and return priority list of tasks.            
    priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))
    return priority_list

def LB_heuristic(dag, platform, schedule_dest=None):
    """
    Basic idea: try to restrict tasks to a CPU or GPU based on their acceleration ratio.
    Expected to do poorly (and does), but might try and extend this approach somehow in the future.
    """
    
    priority_list, where_scheduled = LB_priorities(dag, platform)    
    
    for task in priority_list:
        gpu = True if where_scheduled[task.ID] == "GPU" else False        
        proc, _ = platform.fastest_processor_of_type(task, dag, gpu=gpu)
        platform.workers[proc].schedule_task(task, dag, platform)
    
    # If schedule_dest, print the schedule (i.e., the load of all the processors).
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="LB heuristic", filepath=schedule_dest)         
        
    # Makespan is the maximum AFT of all the exit tasks.        
    mkspan = dag.makespan()
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
        
    return mkspan                
    
def CP_heuristic(dag, platform, priority_list=None, weighting="mean", schedule_dest=None):
    """
    Very similar to Critical Path on a Processor (Topcuoglu, Hariri and Wu, 2002) but uses any input priority list.
    Meant to pursue this further but early results weren't promising.
    """  
        
    # Compute upward and downward ranks of all tasks.
    if priority_list is None:
        priority_list, upward_ranks = dag.sort_by_upward_rank(platform, weighting, return_rank_values=True)
    else:
        _, upward_ranks = dag.sort_by_upward_rank(platform, weighting, return_rank_values=True)
    _, downward_ranks = dag.sort_by_downward_rank(platform, weighting, return_rank_values=True)
    
    # Compute priorities.
    priority = defaultdict(float)    
    for t in dag.DAG:
        priority[t.ID] = upward_ranks[t] + downward_ranks[t]
        if t.entry:
            cp = priority[t.ID] # Assumes single entry task.
            nk = t
            cpu_weight, gpu_weight = t.CPU_time, t.GPU_time
        if t.exit:
            exit_id = t.ID 
    
    # Identify the tasks on the critical path.
    cp_tasks = {nk.ID}
    while nk.ID != exit_id:
        nj = np.random.choice(list(s for s in dag.DAG.successors(nk) if abs(priority[s.ID] - cp) < 1e-6))           
        cp_tasks.add(nj.ID)
        cpu_weight += t.CPU_time
        gpu_weight += t.GPU_time
        nk = nj
    
    if cpu_weight < gpu_weight:
        cp_processor = platform.workers[0]
    else:
        cp_processor = platform.workers[-1]    
       
    for task in priority_list:
        
        if task.ID in cp_tasks:
            cp_processor.schedule_task(task, dag, platform)
        else:
            finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
            min_processor = np.argmin(finish_times)   
            platform.workers[min_processor].schedule_task(task, dag, platform, finish_time=finish_times[min_processor])     
            
    # If schedule_dest, print the schedule (i.e., the load of all the processors).
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="CP heuristic", filepath=schedule_dest)  
            
    # Makespan is the maximum AFT of all the exit tasks. 
    mkspan = dag.makespan() 
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
    
    return mkspan     
    
    