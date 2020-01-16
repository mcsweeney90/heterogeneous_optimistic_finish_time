#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:11:27 2018

This module contains implementations of several static scheduling heuristics, as well as 
Heterogeneous Optimistic Finish Time (HOFT).

@author: Tom
"""

import numpy as np
import networkx as nx
from collections import defaultdict 

####################################################################################################
    
"""Classic heuristics.""" 
    
####################################################################################################   
    
def HEFT(dag, platform, priority_list=None, weighting="HEFT", return_schedule=False, schedule_dest=None):
    """
    Heterogeneous Earliest Finish Time.
    'Performance-effective and low-complexity task scheduling for heterogeneous computing',
    Topcuoglu, Hariri and Wu, 2002.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    The DAG to which the task belongs.
          
    platform - Node object (see Environment.py module)
    The Node to which the Worker belongs. 
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled. 
    
    weighting - string
    How the tasks and edges should be weighted in dag.sort_by_upward_rank.
    Default is "HEFT" which is mean values over all processors as in the original paper. 
    See platform.approximate_comm_cost and task.approximate_execution_cost for other options.
    
    return_schedule - bool
    If True, return the schedule as well.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by HEFT. 
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {task : ID of Worker it is scheduled on}.    
    """ 
    
    if return_schedule:
        pi = defaultdict(int)  
    
    # List all tasks by upward rank unless alternative is specified.
    if priority_list is None:
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
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset() 
    
    if return_schedule:
        return mkspan, pi    
    return mkspan 

def HBMCT(dag, platform, priority_list=None, batch_policy="bmct", bmct_initial_allocation="met", group_sort=None, return_schedule=False, schedule_dest=None):
    """
    Hybrid Balanced Minimum Completion Time.
    'A hybrid heuristic for DAG scheduling on heterogeneous systems',
    Sakellariou and Zhao, 2004.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    The DAG to which the task belongs.
          
    platform - Node object (see Environment.py module)
    The Node to which the Worker belongs. 
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled. 
    
    bmct_policy - string
    bmct_initial_allocation - string
    group_sort - None/string
    The above 3 parameters are all options for platform.schedule_batch; see that method for more detail.
    
    return_schedule - bool
    If True, return the schedule as well.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by HBMCT. 
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {task : ID of Worker it is scheduled on}.    
    """ 
    
    if return_schedule:
        task_order = []
        pi = defaultdict(int) 
    elif schedule_dest:
        task_order = [] 
    
    # List all tasks by upward rank unless alternative is specified.
    if priority_list is None:       
        priority_list = dag.sort_by_upward_rank(platform)                 
                        
    # Find the groups and schedule them.
    G = []
    for task in priority_list:
        if any(p in G for p in dag.DAG.predecessors(task)):
            platform.schedule_batch(dag, G, policy=batch_policy, bmct_initial_allocation=bmct_initial_allocation, batch_sort=group_sort)
            if return_schedule or schedule_dest:
                for t in G:            
                    task_order.append(t.ID)
            G = []
        G.append(task)
    if len(G):
        platform.schedule_batch(dag, G, policy=batch_policy, bmct_initial_allocation=bmct_initial_allocation, batch_sort=group_sort)
        if return_schedule or schedule_dest:
            for task in G:            
                task_order.append(task.ID)
    
    # If schedule_dest, save the priority list and schedule.           
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in task_order:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="HBMCT", filepath=schedule_dest)
        
    if return_schedule:
        for t in task_order:
            pi[t] = t.where_scheduled
        
    # Compute makespan.       
    mkspan = dag.makespan()
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()    
    
    if return_schedule:
        return mkspan, pi 
    return mkspan   
    
def PEFT(dag, platform, priority_list=None, return_schedule=False, schedule_dest=None):
    """
    Predict Earliest Finish Time.
    'List scheduling algorithm for heterogeneous systems by an optimistic cost table',
    Arabnejad and Barbosa, 2014.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    The DAG to which the task belongs.
          
    platform - Node object (see Environment.py module)
    The Node to which the Worker belongs. 
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled. 
        
    return_schedule - bool
    If True, return the schedule as well.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by PEFT. 
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {task : ID of Worker it is scheduled on}.    
    """ 
    
    if return_schedule:
        pi = defaultdict(int) 
    if schedule_dest and priority_list is None:
        task_order = []
    
    OCT = dag.optimistic_cost_table(platform)   
    
    if priority_list is not None:   
        for task in priority_list:
            OEFT = [p.earliest_finish_time(task, dag, platform) + OCT[task][p.ID] for p in platform.workers]
            p = np.argmin(OEFT)
            platform.workers[p].schedule_task(task, dag, platform)
            if return_schedule:
                pi[task] = p
    else:            
        task_weights = {t.ID : np.mean(OCT[task].values()) for t in dag.DAG}    
        ready_tasks = list(t for t in dag.DAG if t.entry)    
        while len(ready_tasks):          
            task = max(ready_tasks, key = lambda t : task_weights[t.ID]) 
            if schedule_dest:         
                task_order.append(task.ID)
            OEFT = [p.earliest_finish_time(task, dag, platform) + OCT[task][p.ID] for p in platform.workers]
            p = np.argmin(OEFT)  
            platform.workers[p].schedule_task(task, dag, platform)
            if return_schedule:
                pi[task] = p                
            ready_tasks.remove(task)
            for c in dag.DAG.successors(task):
                if c.ready_to_schedule(dag):
                    ready_tasks.append(c) 
        
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        if priority_list is None:
            for t in task_order:
                print(t.ID, file=schedule_dest)
        else:            
            for t in priority_list:
                print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="PEFT", filepath=schedule_dest)
        
    # Compute makespan.        
    mkspan = dag.makespan()
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
    
    if return_schedule:
        return mkspan, pi        
    return mkspan    

def PETS(dag, platform, return_schedule=False, schedule_dest=None):
    """
    Performance Effective Task Scheduling.
    'Low complexity performance effective task scheduling algorithm for heterogeneous computing environments',
    Ilavarasan and Thambidurai, 2007.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    The DAG to which the task belongs.
          
    platform - Node object (see Environment.py module)
    The Node to which the Worker belongs. 
            
    return_schedule - bool
    If True, return the schedule as well.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by PETS. 
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {task : ID of Worker it is scheduled on}.    
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
            rank[task] = round(acc[task] + dtc + rpt)            
            
        # Sort all tasks at current level by rank (largest first) with ties broken by acc (smallest first).
        priority_list = list(sorted(acc, key=acc.get))
        priority_list = list(reversed(sorted(priority_list, key=lambda t: rank[t])))   
        
        # Schedule all tasks in current level.
        platform.schedule_batch(dag, priority_list, policy="eft")
        if schedule_dest:
            task_order += list(t.ID for t in priority_list)  
        if return_schedule:
            for task in priority_list:
                pi[task] = task.where_scheduled    
                
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
    Heterogeneous Critical Parent Trees.
    'A simple scheduling heuristic for heterogeneous computing environments',
    Hagras and Janecek, 2003.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    The DAG to which the task belongs.
          
    platform - Node object (see Environment.py module)
    The Node to which the Worker belongs. 
            
    return_schedule - bool
    If True, return the schedule as well.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by PETS. 
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {task : ID of Worker it is scheduled on}.    
    """ 
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
    
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in L:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="HCPT", filepath=schedule_dest)
        
    mkspan = dag.makespan() 
    
    # Reset DAG and platform.
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

    

    
    