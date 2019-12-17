#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:42:26 2019

TODO: re-run this (forgot MCS initially so ran it separately but should re-run so all results are saved to the same place.)

Investigating how useful static scheduling actually is for dynamic environments.

@author: Tom
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
from collections import defaultdict 
from timeit import default_timer as timer
import dill
import sys
sys.path.append('../../') # Quick fix to let us import modules from main directory. 
import Environment    # Node classes and functions.
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

####################################################################################################

def dynamic_allocation_tester(dag, platform, schedule, schedule_dest=None):
    """
    For testing static schedules in dynamic environments. 
    Notes: 
        1. Schedule is assumed be an (ordered) dict {task : processor to schedule it on}.    
    """    
   
    for task in schedule:
        chosen_processor = schedule[task]
        platform.workers[chosen_processor].schedule_task(task, dag, platform)                     
        
    # If verbose, print the schedule (i.e., the load of all the processors).
    if schedule_dest: 
        platform.print_schedule(name="Dynamic allocation tester", filepath=schedule_dest)
     
    mkspan = dag.makespan() 
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
    
    return mkspan 

def priority_scheduler(dag, platform, priority_list, allocation_policy="eft", comm_estimates=None, comp_estimates=None, schedule_dest=None):
    """ Schedules sets of ready tasks according to some input priority list (of all tasks in DAG)."""
    
    if schedule_dest:
        task_order = []
    
    ranks = {t.ID : i for i, t in enumerate(priority_list)}
        
    if allocation_policy == "random":
        mean_acc_ratio = np.mean(t.acceleration_ratio for t in dag.DAG)
        s = platform.n_CPUs + platform.n_GPUs * mean_acc_ratio
        relative_speeds = [1 / s] * platform.n_CPUs + [mean_acc_ratio / s] * platform.n_GPUs
    
    ready_tasks = list(t for t in dag.DAG if t.entry)    
    while len(ready_tasks):          
        task = min(ready_tasks, key = lambda t : ranks[t.ID]) 
        if schedule_dest:         
            task_order.append(task.ID)
        if allocation_policy == "eft":
            finish_times = list([p.earliest_finish_time(task, dag, platform, comm_estimates=comm_estimates, comp_estimates=comp_estimates) for p in platform.workers])
            min_processor = np.argmin(finish_times)   
            platform.workers[min_processor].schedule_task(task, dag, platform)
        elif allocation_policy == "random":
            chosen_processor = np.random.choice(range(platform.workers), p=relative_speeds)
            platform.workers[chosen_processor].schedule_task(task, dag, platform)
            
        ready_tasks.remove(task)
        for c in dag.DAG.successors(task):
            if c.ready_to_schedule(dag):
                ready_tasks.append(c)
                
    # Makespan is the maximum AFT of all the exit tasks.        
    mkspan = dag.makespan()
    
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in task_order:
            print(t, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="Priority scheduler", filepath=schedule_dest)
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
        
    return mkspan 

def immediate_scheduler(dag, platform, allocation_policy="eft", comm_estimates=None, comp_estimates=None, schedule_dest=None):
    """ Schedules all tasks as soon as they become ready according to some task selection policy.""" 
    
    if schedule_dest:
        task_order = []

    if allocation_policy == "random":
        mean_acc_ratio = np.mean(t.acceleration_ratio for t in dag.DAG)
        s = platform.n_CPUs + platform.n_GPUs * mean_acc_ratio
        relative_speeds = [1 / s] * platform.n_CPUs + [mean_acc_ratio / s] * platform.n_GPUs      
    
    entry_times = defaultdict(float)
    ready_tasks = list(t for t in dag.DAG if t.entry)  
    for task in ready_tasks:
        entry_times[task.ID] = 0.0
        
    while len(ready_tasks):  
        task = min(ready_tasks, key = lambda t : entry_times[t.ID])
        if schedule_dest:         
            task_order.append(task.ID)
        if allocation_policy == "eft": 
            finish_times = list([p.earliest_finish_time(task, dag, platform, comm_estimates=comm_estimates, comp_estimates=comp_estimates) for p in platform.workers])
            min_processor = np.argmin(finish_times)   
            platform.workers[min_processor].schedule_task(task, dag, platform)
        elif allocation_policy == "random":
            chosen_processor = np.random.choice(range(platform.workers), p=relative_speeds)
            platform.workers[chosen_processor].schedule_task(task, dag, platform)
            
        ready_tasks.remove(task)
        for c in dag.DAG.successors(task):
            if c.ready_to_schedule(dag):
                ready_tasks.append(c)
                entry_times[c.ID] = task.AFT
            
    # Makespan is the maximum AFT of all the exit tasks.        
    mkspan = dag.makespan()
    
    if schedule_dest: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in task_order:
            print(t, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(name="Immediate scheduler", filepath=schedule_dest)
    
    # Clean up DAG and platform.
    dag.reset()
    platform.reset()    
        
    return mkspan 

def reset_dag_to_original_values(dag, original_comp, original_comm):
    """Helper function for MCS (see below)."""
    for task in dag.DAG:
        task.CPU_time = original_comp["CPU"][task.ID]
        task.GPU_time = original_comp["GPU"][task.ID]
        for child in dag.DAG.successors(task):
            task.comm_costs["CC"][child.ID] = original_comm["CC"][task.ID][child.ID]
            task.comm_costs["CG"][child.ID] = original_comm["CG"][task.ID][child.ID]
            task.comm_costs["GC"][child.ID] = original_comm["GC"][task.ID][child.ID]
            task.comm_costs["GG"][child.ID] = original_comm["GG"][task.ID][child.ID]

def MCS(dag, platform, comp_sample, comm_sample, production_steps=10, selection_steps=10, heuristic="heft", threshold=0.1):
    """ 
    Monte Carlo Scheduling heuristic (Zheng and Sakellariou, 2013).
    Input DAG is assumed to be set with mean task execution time estimates.
    """
    
    L = [] # Empty schedule list.
    M_std, pi = HEFT(dag, platform, return_schedule=True)
    L.append(pi)
    
    default_comp, default_comm = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
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
        # Perturb dag.
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
        # Reset DAG to mean values.
        reset_dag_to_original_values(dag, default_comp, default_comm)
        M_pi = dynamic_allocation_tester(dag, platform, schedule=pi) 
        M_std = min(M_std, M_pi)
        if pi in L:
            continue
        if M_pi < M_std * (1 + threshold): 
            L.append(pi)
    
    print("Number of schedules: {}".format(len(L)))
    avg_schedule_mkspans = [0.0] * len(L)
        
    for i in range(selection_steps):
        # Perturb DAG.
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
            mkspan = dynamic_allocation_tester(dag, platform, schedule=pi)
            avg_schedule_mkspans[j] += mkspan
        
        reset_dag_to_original_values(dag, default_comp, default_comm)        
    
    avg_schedule_mkspans[:] = [m / selection_steps for m in avg_schedule_mkspans]
    
    # Find the schedule that minimizes the average makespan.
    return L[np.argmin(avg_schedule_mkspans)]

####################################################################################################

# Environments.

single = Environment.Node(7, 1, name="Single_GPU")
multiple = Environment.Node(28, 4, name="Multiple_GPU")

####################################################################################################

"""Load real timing data. used for simulating dynamic executions."""  

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
    
"""Compare static, dynamic, hybrid and MCS for Cholesky DAGs."""
    
start = timer()

# Needed for MCS.
comp_sample = defaultdict(lambda: dict)
comm_sample = defaultdict(lambda: defaultdict(list))

runs = 10 # Number of dynamic executions.
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
        comm_sample["CC"]["GEMM"], comm_sample["CC"]["POTRF"], comm_sample["CC"]["SYRK"], comm_sample["CC"]["TRSM"] = [0], [0], [0], [0] # lazy hack, fix...
        with open("results/cholesky/MCS_{}_nb{}.txt".format(env.name, nb), "w") as dest:            
            env.print_info(filepath=dest)
            for j, nt in enumerate(n_tasks):
                dag = nx.read_gpickle('../../graphs/cholesky/nb{}/{}tasks.gpickle'.format(nb, nt))              
                dag.print_info(filepath=dest)
                
                # Compute the static, dynamic and mixed schedules for the original (mean valued) DAG.
                mkspan, pi = HEFT(dag, platform=env, return_schedule=True)
                print("Static HEFT makespan of original DAG: {}".format(mkspan), file=dest)
                
                mkspan = immediate_scheduler(dag, platform=env)
                print("Dynamic HEFT makespan of original DAG: {}".format(mkspan), file=dest)  
                
                priority_list = list(pi.keys())
                mkspan = priority_scheduler(dag, platform=env, priority_list=priority_list)
                print("Mixed HEFT makespan of original DAG: {}".format(mkspan), file=dest)                
                
                mcs_schedule = MCS(dag, env, comp_sample=comp_sample, comm_sample=comm_sample, threshold=0.1)
                mcs_mkspan = dynamic_allocation_tester(dag, platform=env, schedule=mcs_schedule)
                print("MCS makespan of original DAG: {}\n".format(mcs_mkspan), file=dest)
                
                # Save original (mean) values.
                default_comp, default_comm = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
                for task in dag.DAG:
                    default_comp["CPU"][task.ID] = task.CPU_time
                    default_comp["GPU"][task.ID] = task.GPU_time
                    for child in dag.DAG.successors(task):
                        default_comm["CC"][task.ID][child.ID] = task.comm_costs["CC"][child.ID]
                        default_comm["CG"][task.ID][child.ID] = task.comm_costs["CG"][child.ID]
                        default_comm["GC"][task.ID][child.ID] = task.comm_costs["GC"][child.ID]
                        default_comm["GG"][task.ID][child.ID] = task.comm_costs["GG"][child.ID]
                                
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
                    static_mkspan = dynamic_allocation_tester(dag, platform=env, schedule=pi)
                    static_mkspans[env.name][nb].append(static_mkspan)
                    dynamic_mkspan = immediate_scheduler(dag, platform=env, comm_estimates=default_comm, comp_estimates=default_comp) 
                    dynamic_mkspans[env.name][nb].append(dynamic_mkspan)
                    mixed_mkspan = priority_scheduler(dag, platform=env, priority_list=priority_list, comm_estimates=default_comm, comp_estimates=default_comp)
                    mixed_mkspans[env.name][nb].append(mixed_mkspan)
                    
                    mcs_mkspan = dynamic_allocation_tester(dag, platform=env, schedule=mcs_schedule)
                    mcs_mkspans[env.name][nb].append(mcs_mkspan)
               
                print("Number of runs: {}".format(runs), file=dest)                
                print("Mean dynamic makespan: {}".format(np.mean(dynamic_mkspans[env.name][nb][j * 10: (j + 1) * 10])), file=dest)
                print("Mean static makespan: {}".format(np.mean(static_mkspans[env.name][nb][j * 10: (j + 1) * 10])), file=dest)
                print("Mean mixed makespan: {}".format(np.mean(mixed_mkspans[env.name][nb][j * 10: (j + 1) * 10])), file=dest)  
                print("Mean MCS makespan: {}".format(np.mean(mcs_mkspans[env.name][nb][j * 10: (j + 1) * 10])), file=dest)                   
                print("--------------------------------------------------------\n", file=dest)  
                
# Save the makespans so can use again later if I want...
with open('data/static_mkspans.dill', 'wb') as handle:
    dill.dump(static_mkspans, handle)    
with open('data/dynamic_mkspans.dill', 'wb') as handle:
    dill.dump(dynamic_mkspans, handle)    
with open('data/mixed_mkspans.dill', 'wb') as handle:
    dill.dump(mixed_mkspans, handle)    
with open('data/mcs_mkspans.dill', 'wb') as handle:
    dill.dump(mcs_mkspans, handle)    
elapsed = timer() - start
print("This took {} minutes".format(elapsed / 60))
    

"""Plot the mean makespans."""  

with open('data/static_mkspans.dill', 'rb') as file:
    static_mkspans = dill.load(file)    
with open('data/dynamic_mkspans.dill', 'rb') as file:
    dynamic_mkspans = dill.load(file)
with open('data/mixed_mkspans.dill', 'rb') as file:
    mixed_mkspans = dill.load(file)
with open('data/mcs_mkspans.dill', 'rb') as file:
    mcs_mkspans = dill.load(file)    

n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
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