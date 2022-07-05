import numpy as np
import scipy.optimize as op
import pandas as pd
import os
import sys

import models
from utils.twostep_support import *  
from MLE import eval_LL_AI, eval_LL_RL, MLE_procedure

# Input is an integer from sbatch, serving as reference to 1 subject, as well as setting the random seed.
sub = 1
# If you are batching this code with SKRUM, uncomment the following line:
#s = int(sys.argv[1]) - 1 # batch is submitted 1-to-n_subs 

model = "RL" # RL or AI
learning = "RL" # RL, PSM, ...
task = "shock" 

mtype = 0
model_ID = "M" +  str(mtype) # custom note


pfdir = "/.../hierarchicalFittingAndModelComparison/data/"
trial_type = 0 # Self=0, Other=1
T = 136

if model == "RL":
    p_names = ["lr1", "lr2", "lam", "b1", "b2", "p", "w"]
    lower_bounds = np.array([0, 0, 0, 0, 0, -1, 0])
    upper_bounds = np.array([1, 1, 1, 20, 20, 1, 1])
elif model == "AI":
    if mtype == 0:
        p_names = ["lr","vunsamp", "vsamp", "vps", "gamma1", "gamma2", "lam", "kappa_a", "prior_r"]
        lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.2])
        upper_bounds = np.array([4, 0.9, 0.9, 0.9, 30, 30, 10, 5, 0.8])
    elif mtype == 1:
        p_names = ["lr","vunsamp", "vps", "gamma1", "gamma2", "lam", "kappa_a", "prior_r"]
        lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0.2])
        upper_bounds = np.array([4, 0.9, 0.9, 30, 30, 10, 5, 0.8])
    elif mtype == 2:
        p_names = ["lr", "vsamp", "vps", "gamma1", "gamma2", "lam", "kappa_a", "prior_r"]
        lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0.2])
        upper_bounds = np.array([4, 0.9, 0.9, 30, 30, 10, 5, 0.8])
    elif mtype == 3:
        p_names = ["lr","vunsamp", "vsamp", "gamma1", "gamma2", "lam", "kappa_a", "prior_r"]
        lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0.2])
        upper_bounds = np.array([4, 0.9, 0.9, 30, 30, 10, 5, 0.8])       


nump = len(p_names)

n_starts = 25

mc_files = os.listdir(pfdir)
mc_files.sort()

subs = []

for f in mc_files:
    if task == "shock":
        if f.endswith('MFMB.mat'):
            subs.append(f)

n_subs = len(subs)
T_adj = np.zeros(n_subs)

# Load in results file
pf = load_obj(pfdir + subs[sub])
lg = pf["lg"]

transitions = []

badtrials = np.argwhere(np.isnan(lg[:,3]))
lg = np.delete(lg, badtrials,axis=0) # delete bad trials
lg = np.delete(lg, np.argwhere(lg[:,4]==((1-trial_type) + 1)), axis=0) # delete irrelevant condition

actions_i = lg[:,0] - 1
actions_f = lg[:,3] - 1
rewards = lg[:,1]
for t in range(lg.shape[0]):
    if lg[t,3] > 2:
        actions_f[t] -= 2
                
    if lg[t,0] == 1 and lg[t,2] == 0: # rare
        transitions.append(1)
    elif lg[t,0] == 1 and lg[t,2] == 1: # common
        transitions.append(0)
    elif lg[t,0] == 2 and lg[t,2] == 0: # rare
        transitions.append(0)
    elif lg[t,0] == 2 and lg[t,2] == 1: # common
        transitions.append(1)
                    
actions = np.zeros((lg.shape[0],2))
observations = np.zeros((lg.shape[0],2))
T_adj[sub] = int(lg.shape[0])

actions[:,0] = actions_i      
actions[:,1] = actions_f
observations[:,0] = transitions
observations[:,1] = rewards
actions = actions.astype(int)
observations = observations.astype(int)

max_p, max_LL, LLs = MLE_procedure(p_names, 
                                        observations.astype(int), 
                                        actions.astype(int), 
                                        learning,
                                        lower_bounds, 
                                        upper_bounds,
                                        n_starts,
                                        model=model,
                                        mtype=mtype,
                                        seed=sub)

results_formatted = {"max_p": max_p,
                     "max_LL": max_LL,
                     "LLs": LLs}

save_obj(results_formatted, 
"/.../mfit/" + task + "/ultra/" + subs[s] + "_type_"  + str(trial_type) + "_" + learning + "_" + model_ID)

