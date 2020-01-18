#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

TODO: re-run this (forgot MCS initially so ran it separately but should re-run so all results are saved to the same place.)
Expensive so maybe do it in stages...

Investigating how useful static scheduling actually is for dynamic environments.

"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
from collections import defaultdict 
from timeit import default_timer as timer
import dill
import sys
sys.path.append('../../') 
import Environment    
from Heuristics import HEFT

# Set some parameters for plots.
# See here: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.titlepad'] = 5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 12
plt.rcParams["figure.figsize"] = (9.6,4)
#plt.ioff() # Uncomment to suppress plots.

####################################################################################################

def follow_schedule(dag, platform, schedule, schedule_dest=None):
    """
    Schedule all tasks according to the input schedule.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    Represents the task DAG to be scheduled.
          
    platform - Node object (see Environment.py module)
    Represents the target platform. 
    
    schedule - dict
    An ordered dict {task : Worker ID} which describes where all tasks are to be scheduled and 
    in what order this should be done. 
                     
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule.        
    """  
   
    for task in schedule:
        chosen_processor = schedule[task]
        platform.workers[chosen_processor].schedule_task(task, dag, platform)                     
        
    if schedule_dest: 
        platform.print_schedule(name="CUSTOM", filepath=schedule_dest)
    
    # Compute makespan.
    mkspan = dag.makespan() 
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()    
    
    return mkspan 

def expected_earliest_finish_time(worker, task, dag, platform, comm_estimates, comp_estimates, insertion=True):
    """
    Returns the expected earliest finish time for a task on a worker, using cost estimates.
    
    Parameters
    ------------------------    
    worker - Worker Object (see Environment.py module)
    Represents the processing resource to be considered. 
    
    task - Task object (see Graph.py module)
    Represents a (static) task.
    
    dag - DAG object (see Graph.py module)
    The DAG to which the task belongs.
          
    platform - Node object (see Environment.py module)
    The Node object to which the Worker belongs.
    Needed for calculating communication costs.
    
    comm_estimates - Nested dict 
    Of the form {"CC" : {task ID : {child ID : communication time, ...}, ...}.
    The (statically computed) task communication time estimates.
    
    comp_estimates - Nested dict 
    Of the form {"CPU" : {task ID : computation time, ...}, "GPU" : {task ID : computation time, ...}}.
    The (statically computed) task computation time estimates.
    
    insertion - bool
    If True, use insertion-based scheduling policy - i.e., task can be scheduled 
    between two already scheduled tasks, if permitted by dependencies.
                 
    Returns
    ------------------------
    float 
    The expected earliest finish time for the task on the worker.        
    """        
    
    if worker.idle:   # If no tasks scheduled on processor...
        if task.entry: # If an entry task...
            return 0   
        else:
            target_type = "C" if worker.CPU else "G"
            data_ready_time_estimates = []
            for p in dag.DAG.predecessors(task):
                source_type = "C" if p.where_scheduled < platform.n_CPUs else "G"
                data_ready_time_estimates.append(p.AFT + comm_estimates["{}".format(source_type + target_type)][p.ID][task.ID]) 
            processing_time = comp_estimates["CPU"][task.ID] if worker.CPU else comp_estimates["GPU"][task.ID] 
            return max(data_ready_time_estimates)+ processing_time
            
    # Find earliest time all task predecessors have finished and the task can theoretically start.     
    est = 0
    if not task.entry:                    
        predecessors = dag.DAG.predecessors(task) 
        target_type = "C" if worker.CPU else "G"
        data_ready_time_estimates = []
        for p in predecessors:
            source_type = "C" if p.where_scheduled < platform.n_CPUs else "G"
            data_ready_time_estimates.append(p.AFT + comm_estimates["{}".format(source_type + target_type)][p.ID][task.ID])
        est += max(data_ready_time_estimates)  
        
    processing_time = comp_estimates["CPU"][task.ID] if worker.CPU else comp_estimates["GPU"][task.ID]        
    # At least one task already scheduled on processor... 
    # Check if it can be scheduled before any task.
    prev_finish_time = 0
    for t in worker.load:
        if t.AST < est:
            prev_finish_time = t.AFT
            continue
        poss_start_time = max(prev_finish_time, est)
        if poss_start_time + processing_time <= t.AST:
            return poss_start_time + processing_time
        prev_finish_time = t.AFT
    
    # No valid gap found.
    return max(worker.load[-1].AFT, est) + processing_time

def priority_scheduler(dag, platform, priority_list, comm_estimates, comp_estimates, schedule_dest=None):
    """
    Schedule all sets of ready tasks to the Worker estimated to complete their execution at the earliest time (EFT)
    according to a complete task prioritization phase computed statically.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    Represents the task DAG to be scheduled.
          
    platform - Node object (see Environment.py module)
    Represents the target platform. 
    
    priority_list - list
    An ordered list of all tasks in descending order of their priorities.
    
    comm_estimates - Nested dict 
    Of the form {"CC" : {task ID : {child ID : communication time, ...}, ...}.
    The (statically computed) task communication time estimates.
    
    comp_estimates - Nested dict 
    Of the form {"CPU" : {task ID : computation time, ...}, "GPU" : {task ID : computation time, ...}}.
    The (statically computed) task computation time estimates.
                     
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule.        
    """  
    
    if schedule_dest:
        task_order = []
    
    ranks = {t.ID : i for i, t in enumerate(priority_list)} 
    ready_tasks = list(t for t in dag.DAG if t.entry)    
    while len(ready_tasks):          
        task = min(ready_tasks, key = lambda t : ranks[t.ID]) 
        if schedule_dest:         
            task_order.append(task.ID)
        finish_times = list([expected_earliest_finish_time(p, task, dag, platform, comm_estimates, comp_estimates) for p in platform.workers])
        min_processor = np.argmin(finish_times)   
        platform.workers[min_processor].schedule_task(task, dag, platform)
            
        ready_tasks.remove(task)
        for c in dag.DAG.successors(task):
            if c.ready_to_schedule(dag):
                ready_tasks.append(c)
                
    # Compute makespan.       
    mkspan = dag.makespan()
    
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in task_order:
            print(t, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="PRIORITY", filepath=schedule_dest)
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()    
        
    return mkspan 

def immediate_scheduler(dag, platform, comm_estimates, comp_estimates, schedule_dest=None):
    """
    Schedule all tasks as soon as they become ready according to the EFT heuristic.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    Represents the task DAG to be scheduled.
          
    platform - Node object (see Environment.py module)
    Represents the target platform. 
    
    comm_estimates - Nested dict 
    Of the form {"CC" : {task ID : {child ID : communication time, ...}, ...}.
    The (statically computed) task communication time estimates.
    
    comp_estimates - Nested dict 
    Of the form {"CPU" : {task ID : computation time, ...}, "GPU" : {task ID : computation time, ...}}.
    The (statically computed) task computation time estimates.
                         
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule.        
    """
    
    if schedule_dest:
        task_order = []     
    
    entry_times = defaultdict(float)
    ready_tasks = list(t for t in dag.DAG if t.entry)  
    for task in ready_tasks:
        entry_times[task.ID] = 0.0
        
    while len(ready_tasks):  
        task = min(ready_tasks, key = lambda t : entry_times[t.ID])
        if schedule_dest:         
            task_order.append(task.ID)
        finish_times = list([expected_earliest_finish_time(p, task, dag, platform, comm_estimates, comp_estimates) for p in platform.workers])
        min_processor = np.argmin(finish_times)   
        platform.workers[min_processor].schedule_task(task, dag, platform)
            
        ready_tasks.remove(task)
        for c in dag.DAG.successors(task):
            if c.ready_to_schedule(dag):
                ready_tasks.append(c)
                entry_times[c.ID] = task.AFT
            
    # Compute makespan.        
    mkspan = dag.makespan()
    
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in task_order:
            print(t, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="IMMEDIATE", filepath=schedule_dest)
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()    
        
    return mkspan 

def reset_dag_to_original_values(dag, original_comp, original_comm):
    """
    Reset all DAG costs (CPU/GPU times, communication times) to the values specified by original_comp and original_comm.
    Helper function for MCS (see below).
    """
    for task in dag.DAG:
        task.CPU_time = original_comp["CPU"][task.ID]
        task.GPU_time = original_comp["GPU"][task.ID]
        for child in dag.DAG.successors(task):
            task.comm_costs["CC"][child.ID] = original_comm["CC"][task.ID][child.ID]
            task.comm_costs["CG"][child.ID] = original_comm["CG"][task.ID][child.ID]
            task.comm_costs["GC"][child.ID] = original_comm["GC"][task.ID][child.ID]
            task.comm_costs["GG"][child.ID] = original_comm["GG"][task.ID][child.ID]

def MCS(dag, platform, comp_sample, comm_sample, production_steps=10, selection_steps=10, threshold=0.1):
    """
    Monte Carlo Scheduling (MCS) heuristic.
    'Stochastic DAG scheduling using a Monte Carlo approach',
    Zheng and Sakellariou and Madeira, 2013.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    Represents the task DAG to be scheduled.
          
    platform - Node object (see Environment.py module)
    Represents the target platform. 
    
    comp_sample - Nested dict
    Computation costs to sample from to simulate the actual computation times at runtime.
    
    comm_sample - Nested dict
    Communication costs to sample from to simulate the actual computation times at runtime.
    
    production_steps - int
    Controls the number of candidate schedules that are generated.
    
    selection_steps - int 
    Controls the number of simulated executions used to whittle down potential schedules.
    
    Threshold - float
    Used to check whether generated schedules are worth further investigation.                     
    
    Returns
    ------------------------
    dict
    Of the form {task : ID of Worker to schedule it on}, with scheduling order determined by order of keys.
    The MCS schedule.  

    Notes
    ------------------------
    1. Uses HEFT to generate schedules but can be used with almost any other static heuristic instead.
    """
    
    L = [] 
    M_std, pi = HEFT(dag, platform, return_schedule=True) 
    L.append(pi)
    
    # Save the original DAG costs.
    default_comp = defaultdict(lambda: defaultdict(float))
    default_comm = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for task in dag.DAG:
        default_comp["CPU"][task.ID] = task.CPU_time
        default_comp["GPU"][task.ID] = task.GPU_time
        for child in dag.DAG.successors(task):
            default_comm["CC"][task.ID][child.ID] = task.comm_costs["CC"][child.ID]
            default_comm["CG"][task.ID][child.ID] = task.comm_costs["CG"][child.ID]
            default_comm["GC"][task.ID][child.ID] = task.comm_costs["GC"][child.ID]
            default_comm["GG"][task.ID][child.ID] = task.comm_costs["GG"][child.ID]       
        
    for i in range(production_steps):
        print("Production step: {}".format(i))
        # Set costs to random samples from timing data.
        for task in dag.DAG:
            task.CPU_time = np.random.choice(comp_sample["CPU"][task.type])
            task.GPU_time = np.random.choice(comp_sample["GPU"][task.type])
            cc_comm_costs, cg_comm_costs, gc_comm_costs, gg_comm_costs = {}, {}, {}, {}
            for child in dag.DAG.successors(task):
                cc_comm_costs[child.ID] = np.random.choice(comm_sample["CC"][child.type])
                cg_comm_costs[child.ID] = np.random.choice(comm_sample["CG"][child.type])
                gc_comm_costs[child.ID] = np.random.choice(comm_sample["GC"][child.type])
                gg_comm_costs[child.ID] = np.random.choice(comm_sample["GG"][child.type])
            task.comm_costs["CC"] = cc_comm_costs
            task.comm_costs["CG"] = cg_comm_costs
            task.comm_costs["GC"] = gc_comm_costs
            task.comm_costs["GG"] = gg_comm_costs
        _, pi = HEFT(dag, platform, return_schedule=True)
        # Reset DAG to original costs.
        reset_dag_to_original_values(dag, default_comp, default_comm)
        M_pi = follow_schedule(dag, platform, schedule=pi) 
        M_std = min(M_std, M_pi)
        if pi in L:
            continue
        if M_pi < M_std * (1 + threshold): 
            L.append(pi)
    
#    print("Number of schedules: {}".format(len(L)))
    avg_schedule_mkspans = [0.0] * len(L)        
    for i in range(selection_steps):
        # Set costs to random samples from timing data.
        for task in dag.DAG:
            task.CPU_time = np.random.choice(comp_sample["CPU"][task.type])
            task.GPU_time = np.random.choice(comp_sample["GPU"][task.type])
            cc_comm_costs, cg_comm_costs, gc_comm_costs, gg_comm_costs = {}, {}, {}, {}
            for child in dag.DAG.successors(task):
                cc_comm_costs[child.ID] = np.random.choice(comm_sample["CC"][child.type])
                cg_comm_costs[child.ID] = np.random.choice(comm_sample["CG"][child.type])
                gc_comm_costs[child.ID] = np.random.choice(comm_sample["GC"][child.type])
                gg_comm_costs[child.ID] = np.random.choice(comm_sample["GG"][child.type])
            task.comm_costs["CC"] = cc_comm_costs
            task.comm_costs["CG"] = cg_comm_costs
            task.comm_costs["GC"] = gc_comm_costs
            task.comm_costs["GG"] = gg_comm_costs                
        for j, pi in enumerate(L):
            mkspan = follow_schedule(dag, platform, schedule=pi)
            avg_schedule_mkspans[j] += mkspan
        # Restore DAG to original cost values.
        reset_dag_to_original_values(dag, default_comp, default_comm)        
    
    avg_schedule_mkspans[:] = [m / selection_steps for m in avg_schedule_mkspans]
    
    # Find the schedule that minimizes the average makespan.
    return L[np.argmin(avg_schedule_mkspans)]

####################################################################################################

# Define environments to be considered.
single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

####################################################################################################

"""Load real timing data. Used for simulating dynamic executions."""  

min_tile = 32
max_tile = 1024
inc_factor = 2
iterations = 1001
sizes = [32, 64, 128, 256, 512, 1024]
cpu_data = defaultdict(lambda: defaultdict(list))
gpu_data = defaultdict(lambda: defaultdict(list))
comm_data = defaultdict(lambda: defaultdict(list))

# Load CPU data.
name = "skylake"
dgemm_raw = np.genfromtxt('data/{}/DGEMM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dpotrf_raw = np.genfromtxt('data/{}/DPOTRF_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dsyrk_raw = np.genfromtxt('data/{}/DSYRK_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dtrsm_raw = np.genfromtxt('data/{}/DTRSM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
x = min_tile
i = 1 # Ignore the first iteration for each tile size - more expensive than later iterations because of libraries being loaded, etc.
while x < max_tile + 1:
    cpu_data[x]["GEMM"] = [y[2] for y in dgemm_raw[i: i + iterations - 1]]
    cpu_data[x]["POTRF"] = [y[2] for y in dpotrf_raw[i: i + iterations - 1]]
    cpu_data[x]["SYRK"]= [y[2] for y in dsyrk_raw[i: i + iterations - 1]]
    cpu_data[x]["TRSM"] = [y[2] for y in dtrsm_raw[i: i + iterations - 1]]            
    i += iterations
    x *= inc_factor        

# Load GPU data.
name = "V100"
dgemm_raw = np.genfromtxt('data/{}/DGEMM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dpotrf_raw = np.genfromtxt('data/{}/DPOTRF_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dsyrk_raw = np.genfromtxt('data/{}/DSYRK_{}.csv'.format(name, name), delimiter=',', skip_header=1)
dtrsm_raw = np.genfromtxt('data/{}/DTRSM_{}.csv'.format(name, name), delimiter=',', skip_header=1)
x = min_tile
i = 1 # Ignore the first iteration for each tile size - more expensive than later iterations because of libraries being loaded, etc.
while x < max_tile + 1:
    gpu_data[x]["GEMM"] = [y[2] for y in dgemm_raw[i: i + iterations - 1]]
    gpu_data[x]["POTRF"] = [y[2] for y in dpotrf_raw[i: i + iterations - 1]]
    gpu_data[x]["SYRK"] = [y[2] for y in dsyrk_raw[i: i + iterations - 1]]
    gpu_data[x]["TRSM"] = [y[2] for y in dtrsm_raw[i: i + iterations - 1]]  
    comm_data[x]["GEMM"] = [y[3] - y[2] for y in dgemm_raw[i: i + iterations - 1]]  
    comm_data[x]["POTRF"] = [y[3] - y[2] for y in dpotrf_raw[i: i + iterations - 1]]
    comm_data[x]["SYRK"] = [y[3] - y[2] for y in dsyrk_raw[i: i + iterations - 1]]
    comm_data[x]["TRSM"] = [y[3] - y[2] for y in dtrsm_raw[i: i + iterations - 1]]   
    i += iterations
    x *= inc_factor

####################################################################################################
    
"""
Compare static, dynamic, hybrid and MCS for Cholesky DAGs, where:
    - "static" means following the HEFT schedule (computed statically with cost estimates);
    - "dynamic" means scheduling tasks as soon as they become ready to the worker estimated to complete
      their execution at the earliest time;
    - "hybrid" means computing all task priorities according to upward rank using cost estimates
      then choosing from ready task set according to these and selecting EFT worker;
    - "MCS" means following the Monte Carlo Scheduling heuristic.
"""
    
start = timer()

# Needed for MCS.
comp_sample = defaultdict(lambda: dict)
comm_sample = defaultdict(lambda: defaultdict(list))

runs = 10 # Number of simulated dynamic executions.
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]# 16215, 22100] # Slow, so comment largest DAGs.
static_mkspans = defaultdict(lambda: defaultdict(list))
dynamic_mkspans = defaultdict(lambda: defaultdict(list))
mixed_mkspans = defaultdict(lambda: defaultdict(list))
mcs_mkspans = defaultdict(lambda: defaultdict(list))
for env in [single, multiple]:
    env.print_info()
    for nb in [128, 1024]:
        comp_sample["CPU"], comp_sample["GPU"] = cpu_data[nb], gpu_data[nb]
        comm_sample["CG"], comm_sample["GC"], comm_sample["GG"] = comm_data[nb], comm_data[nb], comm_data[nb]  
        comm_sample["CC"]["GEMM"], comm_sample["CC"]["POTRF"], comm_sample["CC"]["SYRK"], comm_sample["CC"]["TRSM"] = [0], [0], [0], [0] # lazy hack
        with open("results/MCS_{}_nb{}.txt".format(env.name, nb), "w") as dest:            
            env.print_info(filepath=dest)
            for j, nt in enumerate(n_tasks):
                dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))              
                dag.print_info(filepath=dest)
                
                # Save original values.
                default_comp, default_comm = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
                for task in dag.DAG:
                    default_comp["CPU"][task.ID] = task.CPU_time
                    default_comp["GPU"][task.ID] = task.GPU_time
                    for child in dag.DAG.successors(task):
                        default_comm["CC"][task.ID][child.ID] = task.comm_costs["CC"][child.ID]
                        default_comm["CG"][task.ID][child.ID] = task.comm_costs["CG"][child.ID]
                        default_comm["GC"][task.ID][child.ID] = task.comm_costs["GC"][child.ID]
                        default_comm["GG"][task.ID][child.ID] = task.comm_costs["GG"][child.ID]
                
                # Compute the static, dynamic and mixed schedules for the original DAG.
                mkspan, pi = HEFT(dag, platform=env, return_schedule=True)
                print("Static HEFT makespan of original DAG: {}".format(mkspan), file=dest)
                
                mkspan = immediate_scheduler(dag, platform=env, comm_estimates=default_comm, comp_estimates=default_comp)
                print("Dynamic HEFT makespan of original DAG: {}".format(mkspan), file=dest)  
                
                priority_list = list(pi.keys())
                mkspan = priority_scheduler(dag, platform=env, priority_list, comm_estimates=default_comm, comp_estimates=default_comp)
                print("Mixed HEFT makespan of original DAG: {}".format(mkspan), file=dest)                
                
                mcs_schedule = MCS(dag, env, comp_sample=comp_sample, comm_sample=comm_sample, threshold=0.1)
                mcs_mkspan = follow_schedule(dag, platform=env, schedule=mcs_schedule)
                print("MCS makespan of original DAG: {}\n".format(mcs_mkspan), file=dest)                
                                
                for i in range(runs):   
                    # Change DAG weights to simulate dynamic execution.
                    for task in dag.DAG:
                        task.CPU_time = np.random.choice(cpu_data[nb][task.type])
                        task.GPU_time = np.random.choice(gpu_data[nb][task.type])
                        task.acceleration_ratio = task.CPU_time / task.GPU_time
                        cc_comm_costs, cg_comm_costs, gc_comm_costs, gg_comm_costs = {}, {}, {}, {}
                        for child in dag.DAG.successors(task):
                            cc_comm_costs[child.ID] = 0
                            cg_comm_costs[child.ID] = np.random.choice(comm_data[nb][child.type])
                            gc_comm_costs[child.ID] = np.random.choice(comm_data[nb][child.type])
                            gg_comm_costs[child.ID] = np.random.choice(comm_data[nb][child.type])
                        task.comm_costs["CC"] = cc_comm_costs
                        task.comm_costs["CG"] = cg_comm_costs
                        task.comm_costs["GC"] = gc_comm_costs
                        task.comm_costs["GG"] = gg_comm_costs                    
                    static_mkspan = follow_schedule(dag, platform=env, schedule=pi)
                    static_mkspans[env.name][nb].append(static_mkspan)
                    dynamic_mkspan = immediate_scheduler(dag, platform=env, comm_estimates=default_comm, comp_estimates=default_comp) 
                    dynamic_mkspans[env.name][nb].append(dynamic_mkspan)
                    mixed_mkspan = priority_scheduler(dag, platform=env, priority_list=priority_list, comm_estimates=default_comm, comp_estimates=default_comp)
                    mixed_mkspans[env.name][nb].append(mixed_mkspan)                    
                    mcs_mkspan = follow_schedule(dag, platform=env, schedule=mcs_schedule)
                    mcs_mkspans[env.name][nb].append(mcs_mkspan)
               
                print("Number of runs: {}".format(runs), file=dest)                
                print("Mean dynamic makespan: {}".format(np.mean(dynamic_mkspans[env.name][nb][j * 10: (j + 1) * 10])), file=dest)
                print("Mean static makespan: {}".format(np.mean(static_mkspans[env.name][nb][j * 10: (j + 1) * 10])), file=dest)
                print("Mean mixed makespan: {}".format(np.mean(mixed_mkspans[env.name][nb][j * 10: (j + 1) * 10])), file=dest)  
                print("Mean MCS makespan: {}".format(np.mean(mcs_mkspans[env.name][nb][j * 10: (j + 1) * 10])), file=dest)                   
                print("--------------------------------------------------------\n", file=dest)  
                
# Save the makespans so can use later.
with open('results/static_mkspans.dill', 'wb') as handle:
    dill.dump(static_mkspans, handle)    
with open('results/dynamic_mkspans.dill', 'wb') as handle:
    dill.dump(dynamic_mkspans, handle)    
with open('results/mixed_mkspans.dill', 'wb') as handle:
    dill.dump(mixed_mkspans, handle)    
with open('results/mcs_mkspans.dill', 'wb') as handle:
    dill.dump(mcs_mkspans, handle)    
elapsed = timer() - start
print("This took {} minutes".format(elapsed / 60))
    

"""Plot the mean makespans."""  
try:
    static_mkspans
except NameError:
    with open('results/static_mkspans.dill', 'rb') as file:
        static_mkspans = dill.load(file)  
try:
    dynamic_mkspans
except NameError:
    with open('results/dynamic_mkspans.dill', 'rb') as file:
        dynamic_mkspans = dill.load(file)
try:
    mixed_mkspans
except NameError:
    with open('results/mixed_mkspans.dill', 'rb') as file:
        mixed_mkspans = dill.load(file)
try:
    mcs_mkspans
except NameError:
    with open('results/mcs_mkspans.dill', 'rb') as file:
        mcs_mkspans = dill.load(file)    

n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

# Choose nb and platform to plot for...
nb = 128
env = multiple

dynamic_makespans, static_makespans, mixed_makespans, mcs_makespans = [], [], [], [] 

for i in range(0, 80, 10):
    dynamic_makespans.append(np.mean(dynamic_mkspans[env.name][nb][i:i+10]))
    static_makespans.append(np.mean(static_mkspans[env.name][nb][i:i+10]))
    mixed_makespans.append(np.mean(mixed_mkspans[env.name][nb][i:i+10]))
    mcs_makespans.append(np.mean(mcs_mkspans[env.name][nb][i:i+10]))
    
static_reductions, mixed_reductions, mcs_reductions = [], [], []
for i, m in enumerate(dynamic_makespans):
    static_reductions.append(100 - (static_makespans[i] / m) * 100)
    mixed_reductions.append(100 - (mixed_makespans[i] / m) * 100)
    mcs_reductions.append(100 - (mcs_makespans[i] / m) * 100)

preferences = {"DYNAMIC" : ["-", "o"], "HYBRID" : ["-", "s"], "MCS": ["-", "D"]}             
fig1 = plt.figure(dpi=400) 
ax1 = fig1.add_subplot(111)
ax1.set_xlabel("NUMBER OF TASKS", labelpad=10) 
ax1.set_ylabel("MEAN REDUCTION VS DYNAMIC (%)", labelpad=10)  
plt.xscale('log')
#plt.yscale('log')
ax1.plot(n_tasks, static_reductions, linestyle=preferences["STATIC"][0], marker=preferences["STATIC"][1], label="STATIC")
ax1.plot(n_tasks, mixed_reductions, linestyle=preferences["HYBRID"][0], marker=preferences["HYBRID"][1], label="HYBRID")
ax1.plot(n_tasks, mcs_reductions, linestyle=preferences["MCS"][0], marker=preferences["MCS"][1], label="MCS")
ax1.set_ylim(bottom=0)
plt.yticks(np.arange(0, 50, 10.0)) # Make it look nicer.
ax1.legend(handlelength=1.8, handletextpad=0.4, loc='upper left', fancybox=True)  
plt.savefig('plots/{}_cholesky_nb{}'.format(env.name, nb), bbox_inches='tight') 