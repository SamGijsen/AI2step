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

# spaceship or magic_carpet
model = "RL" # RL or AI
learning = "RL" # RL, PSM
task = "online" 

mtype = 0

model_ID = "M" +  str(mtype) # custom note

T = 150
n_subs = 206

# participant choice data
pfdir = "/.../tradeoffs/data/daw paradigm/data.mat"

full = load_obj(pfdir)
data = full['data']

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

# 206 subjects, 150 trials each
badtrials = np.zeros((n_subs))
repeat_i = np.zeros((n_subs))
repeat_f = np.zeros((n_subs))
s_last = 999
for i in range(30900):  
    s = int(np.floor(i/150))
    
    if data[i][3] < 0 or data[i][3] > 2000: # check RTs
        badtrials[s] += 1
    elif data[i][7] < 0 or data[i][7] > 2000:
        badtrials[s] += 1
        
    if s == s_last: # Count action-repeats to exclude people who pressed the same button on every trial
        if data[i][4] == data[i-1][4]:
            repeat_i[s] += 1
        if (data[i][8] == data[i-1][8]):
            repeat_f[s] += 1
            
    s_last = np.copy(s)
    
badsubs = np.argwhere(badtrials>25)[:,0]
goodsubs = np.argwhere(badtrials<=25)[:,0]
n_subs_good = len(goodsubs)

sub = goodsubs[sub]

badtrials, actions_i, actions_f, transitions, rewards = [], [], [], [], []

ts = sub*T # start trial index
te = (sub+1)*T # end trial index
for t in range(T):
    i = sub*T + t
    
    # Identify missed trials
    if data[i][3] < 0 or data[i][3] > 2000:
        badtrials.append(t)
    elif data[i][7] < 0 or data[i][7] > 2000:
        badtrials.append(t)
        
    # Store actions, transitions, rewards
    actions_i.append(data[i][4])
    actions_f.append(data[i][8])
    transitions.append(data[i][10])
    rewards.append(data[i][9])
    
T_adj = T - len(badtrials)
        
actions = np.zeros((T - len(badtrials),2))
observations = np.zeros((T - len(badtrials),2))

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
                                        seed=sub)

results_formatted = {"max_p": max_p,
                     "max_LL": max_LL,
                     "LLs": LLs}

save_obj(results_formatted, 
"/.../mfit/" + task + "/models/" + subs[s] + "_" + learning + "_" + model_ID)
