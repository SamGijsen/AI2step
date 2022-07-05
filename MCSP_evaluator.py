import numpy as np
import scipy.optimize as op
import pandas as pd
import os
import sys

import models
from utils.twostep_support import *    
from MLE import eval_LL_AI, eval_LL_RL, MLE_procedure

# Input is an integer from sbatch, serving as reference to 1 subject, as well as setting the random seed.
s = 1
# If you are batching this code with SKRUM, uncomment the following line:
#s = int(sys.argv[1]) - 1 # batch is submitted 1-to-n_subs 

# spaceship or magic_carpet
model = "RL" # RL or AI
learning = "RL" # RL, PSM
task = "spaceship" # spaceship or magic_carpet

mtype = 0

model_ID = "M" +  str(mtype) # custom note

if task == "spaceship": # 21 subjects
    T = 250
elif task == "magic_carpet": # 24 subjects
    T = 201

# participant choice data 

pfdir = "/.../muddled_models/results/" + task + "/choices/"

# Different model setups
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
    
    if task == "spaceship":
        if f.endswith('csv') and not f.endswith('practice.csv'):
            subs.append(f)
            
    elif task == "magic_carpet":
        if f.endswith('_game.csv'):
            subs.append(f)
            
n_subs = len(subs)

# Load in results file
pf = pd.read_csv(pfdir + subs[s])

# Identify very fast and slow trials
badtrials = np.concatenate((np.argwhere(pf["rt1"].values < 0)[:, 0], np.argwhere(pf["rt2"].values < 0)[:, 0]))
badtrials = np.sort(np.unique(badtrials))

if task == "spaceship":
    actions_i = []

    actions_f = pf["choice2"].values + 1
    transitions = pf["final_state"].values + 1
    
    for t in range(len(pf["choice1"].values)):
        if pf["common"].values[t]:
            actions_i.append(pf["final_state"].values[t] + 1)
        else:
            actions_i.append(2 - pf["final_state"].values[t])

elif task == "magic_carpet":
    
    actions_i = pf["choice1"].values
    actions_f = pf["choice2"].values
    transitions = pf["final_state"].values

rewards = pf["reward"].values

actions = np.zeros((T,2))
observations = np.zeros((T,2))

actions[:,0] = np.delete(actions_i, badtrials) - 1
actions[:,1] = np.delete(actions_f, badtrials) - 1
observations[:,0] = np.delete(transitions, badtrials) - 1
observations[:,1] = np.delete(rewards, badtrials)

max_p, max_LL, LLs = MLE_procedure(p_names, 
                                        observations.astype(int), 
                                        actions.astype(int), 
                                        learning,
                                        lower_bounds, 
                                        upper_bounds,
                                        n_starts,
                                        model=model,
                                        mtype=mtype,
                                        seed=s)

results_formatted = {"max_p": max_p,
                     "max_LL": max_LL,
                     "LLs": LLs}

save_obj(results_formatted, 
"/.../mfit/" + task + "/models/" + subs[s] + "_" + learning + "_" + model_ID)
