import numpy as np
import scipy.optimize as op
import pandas as pd
import os
import sys

import models
from utils.twostep_support import *    

def eval_LL_AI(params, observations, actions, learning, mtype):
    """
    Evaluate likelihood for a sequence of actions given a sequence of observations (performed trial-wise).

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    observations: sequence of transition and outcome observations
    actions: sequence of taken actions
    learning: learning algorithm used (Default is "PSM": Predictive-surprise modulated learning)
    mtype: integer specifying submodel
    """

    lr = params[0]
    vunsamp = params[1]
    vsamp = params[2]
    vps = params[3]
    gam1 = params[4]
    gam2 = params[5]
    lam = params[6]
    kappa_a = params[7]
    prior_nu = 2
    prior_r = params[8]

    # As is, we have the full model. Reductions: 
    if mtype > 0: # We set parameters 0, 1, and 2 later, for now the identical parameters:
        lr = params[0]
        gam1 = params[3]
        gam2 = params[4]
        lam = params[5]
        kappa_a = params[6]
        prior_nu = 2
        prior_r = params[7]

    if mtype == 1: # No Decay for Sampled Actions
        vunsamp = params[1]
        vsamp = 0
        vps = params[2]

    if mtype == 2: # No Surprise Learning
        vunsamp = params[1]
        vsamp = params[2]
        vps = 0

    if mtype == 3: # No Decay for UNsampled Actions
        vunsamp = 0
        vsamp = params[1]
        vps = params[2]


    #### ----------
    # Specify task and generate (potential) observations
    T = observations.shape[0]
    task = {  
        "type": "drift",
        "T": T,
        "x": False,
        "r": True,
        "delta": 0.025
    }
    
    model = { # Model specification
        "act": "AI",
        "learn": "PSM",
        "learn_transitions": False,
        "lr": lr,
        "vunsamp": vunsamp,
        "vsamp": vsamp,
        "vps": vps, 
        "gamma1": gam1,
        "gamma2": gam2,
        "lam": lam,
        "kappa_a": kappa_a,
        "prior_r": prior_r
        }

    temp = models.learn_and_act(task, model, seed=1)
    La = np.ones((T,2))

    po = np.zeros(2)
    pa = np.zeros(2)
    
    # Check which actions were taken and which outcomes observed
    for t in range(T):
        po = observations[t,:].astype(int)
        pa = actions[t,:].astype(int)

        a, o, pi, p_trans, p_r, GQ = temp.perform_trial(t, pa, po)

        # softmax for both stages
        GQ[t,0,:] = np.exp(GQ[t,0,:]) / np.sum(np.exp(GQ[t,0,:]))
        GQ[t,po[0]+1,:] = np.exp(GQ[t,po[0]+1,:]) / np.sum(np.exp(GQ[t,po[0]+1,:]))

        La[t,0] = GQ[t, 0, pa[0]]
        La[t,1] = GQ[t, po[0]+1, pa[1]]

        with np.errstate(divide='raise'):
            try:
                -np.sum(np.log(La))
            except :
                print(t, La[:t+1,:])
                print("GQ",GQ[:t+1,:,:])

    return -np.sum(np.log(La))

def eval_LL_RL(params, observations, actions):
    """
    Evaluate likelihood for a sequence of actions given a sequence of observations (performed trial-wise).

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    observations: sequence of transition and outcome observations
    actions: sequence of taken actions
    """

    a1 = params[0]
    a2 = params[1]
    lam = params[2]
    b1 = params[3]
    b2 = params[4]
    p = params[5]
    w = params[6]

    T = observations.shape[0] 

    Steps = 2 
    Qb = np.zeros((3,2))
    Qf = np.zeros((3,2))

    counts = np.zeros((2,2)) # 2 actions by 2 final states 
    tm = np.array([[0.5, 0.5]])

    prev_a = 999

    La = np.ones((T,Steps))
    for t in range(T):
        for step in range(Steps):
            if step == 0:
                state = 0
            else:
                state = int(o) + 1
                counts[actions[t,0], int(observations[t,0])] += 1

            # Action selection --------------------------------------
            a_t, gq = action_selection_RL(state, b1, b2, w, p, Qb, Qf, prev_a)
            if step == 0: 
                prev_a = int(actions[t,step])

            # Check participant action 
            La[t,step] = gq[int(actions[t,step])]
            o = observations[t,step]

            # Update Q-values
            if step == 1:
                Qf = update_SARSA(state, a1, a2, lam, Qf, prev_a, actions[t,step], observations[t,step])
                Qb[1:,:] = np.copy(Qf[1:,:]) # MB equals MF for the final stage
                Qb = update_MB(Qb, tm)

            # Determine most likely transition matrix
            if (counts[0,0] + counts[1,1]) > (counts[0,1] + counts[1,0]):
                tm = np.array([0.3, 0.7])
            if (counts[0,0] + counts[1,1]) < (counts[0,1] + counts[1,0]):
                tm = np.array([0.7, 0.3])
            if (counts[0,0] + counts[1,1]) == (counts[0,1] + counts[1,0]):
                tm = np.array([0.5, 0.5])

    return -np.sum(np.log(La)) # minimize logs

def MLE_magiccarpet(params, obs, actions, learning, lower_bounds, upper_bounds, n_starts, model, mtype, seed=1):
    """
    This function calls scipy.op.minimize() repeatedly to perform maximum likelihood estimation.

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    obs: sequence of transition and outcome observations
    actions: sequence of taken actions
    learning: learning algorithm
    lower_bounds, upper_bounds: each parameter needs a min and max bound between which the minimizer functions
    n_starts: amount of iterations. be careful of local minima in case n_starts < 10
    model: active inference (AI) or reinforcement learning (RL)
    mtype: submodel type for active inference
    """

    np.random.seed(seed)

    nump = len(params)
    LL = np.zeros(n_starts)
    init_params = np.zeros((nump, n_starts))
    params = np.zeros((nump, n_starts))

    # Create bounds and parameter initializations
    bounds = []
    for i in range(nump):
        bounds.append((lower_bounds[i], upper_bounds[i]))
        for j in range(n_starts):
            init_params[i,j] = np.random.uniform(low=lower_bounds[i], high=upper_bounds[i])

    options = {"disp":False}

    for j in range(n_starts):

        if model == "RL":
            res = op.minimize(
            eval_LL_RL,
            [init_params[:,j]],
            args=(observations,actions),
            method="L-BFGS-B",
            bounds=bounds,
            options=options)
        elif model == "AI":
            res = op.minimize(
            eval_LL_AI,
            [init_params[:,j]],
            args=(observations,actions,learning,mtype),
            method="L-BFGS-B",
            bounds=bounds,
            options=options)           

        for i in range(nump):
            params[i,j] = res.x[i]
        LL[j] = res.fun

        # print(res.nit, res.success, res.status, res.message)

    best_iter = np.nanargmin(LL)
    print("allparams=",params,"LLs",LL,"chosen",best_iter)
    return params[:,best_iter], LL[best_iter], LL

# Input is an integer from sbatch, serving as reference to 1 subject, as well as setting the random seed.
s = 1
# If you are batching this code with SKRUM, uncomment the following line:
#s = int(sys.argv[1]) - 1 # batch is submitted 1-to-n_subs 

# spaceship or magic_carpet
model = "AI" # RL or AI
learning = "PSM" # Q, PSM
task = "online" 

mtype = 0

model_ID = "M" +  str(mtype) # custom note

T = 150
n_subs = 206

# participant choice data
pfdir = "/.../tradeoffs/data/daw paradigm/data.mat"

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

max_p, max_LL, LLs = MLE_magiccarpet(p_names, 
                                        observations.astype(int), 
                                        actions.astype(int), 
                                        learning,
                                        lower_bounds, 
                                        upper_bounds,
                                        n_starts,
                                        curious,
                                        model=model,
                                        seed=s)

results_formatted = {"max_p": max_p,
                     "max_LL": max_LL,
                     "LLs": LLs}

save_obj(results_formatted, 
"/.../mfit/" + task + "/models/" + subs[s] + "_" + learning + "_" + model_ID)
