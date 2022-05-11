import numpy as np
import scipy.optimize as op
import pandas as pd
import os
import sys

from twostep_learning_acting import *
from twostep_support import *    

def eval_LL_AI(params, observations, actions, learning, mtype):
    """
    Evaluate likelihood for a sequence of actions given a sequence of observations (performed trial-wise).
    v  = environmental variability
    lam = precision on expected free energy
    obs = sequence of observations
    actions = sequence of taken actions
    K = amount of arms
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


    T = observations.shape[0]
    Steps = 2

    pi = np.ones((T+1,Steps,2,6))
    # Construct prior distribution
    if learning == "PSM":
        prior = np.array([(1-prior_r)*prior_nu, prior_r*prior_nu])        
        for step in range(Steps):
            for a in range(2,6):
                pi[:,step,:,a] = prior

    # Initialize transition counts, transition matrix beliefs, and previously selected action
    counts = np.zeros((2,2)) # 2 actions by 2 final states
    tm = np.array([0.5, 0.5])
    prev_a = 999

    La = np.ones((T,Steps))

    for t in range(T):
        for step in range(Steps):
            if step == 0:
                state = 0
            else:
                state = int(o) + 1 
                counts[actions[t,0], int(observations[t,0])] += 1

            # Action selection
            if step == 0:
                gamma = gam1
            else:
                gamma = gam2
            a_t, g = action_selection_AI(t, state, pi, tm, lr, vunsamp, vsamp, vps, lam, kappa_a, prev_a, learning, gamma, prior_nu, prior_r, False)
            gq = np.exp(g) / np.sum(np.exp(g))

            if step==0:
                prev_a = np.copy(int(actions[t,step]))

            # Check participant action
            La[t,step] = gq[int(actions[t,step])]
            o = observations[t,step]

            # Update model
            if step == 1:
                if state == 1:
                    a = actions[t,step] + 2
                elif state == 2:
                    a = actions[t,step] + 4
                else:
                    raise("!")

                # Learning
                pi = PSM_learning(t, step, int(a), int(observations[t,step]), pi, tm, lr, vunsamp, vsamp, vps, prior_nu, prior_r, False)

            # Determine most likely transition matrix
            if (counts[0,0] + counts[1,1]) > (counts[0,1] + counts[1,0]):
                tm = np.array([0.3, 0.7])
            if (counts[0,0] + counts[1,1]) < (counts[0,1] + counts[1,0]):
                tm = np.array([0.7, 0.3])
            if (counts[0,0] + counts[1,1]) == (counts[0,1] + counts[1,0]):
                tm = np.array([0.5, 0.5])

    return -np.sum(np.log(La))

def eval_LL_RL(params, observations, actions):
    """
    Evaluate likelihood for a sequence of actions given a sequence of observations (performed trial-wise).
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
s = int(sys.argv[1]) - 1 # batch is submitted 1-to-n_subs

# spaceship or magic_carpet
model = "AI" # RL or AI
learning = "PSM" # Q, PSM
task = "spaceship" # spaceship or magic_carpet

mtype = 0

model_ID = "M" +  str(mtype) # custom note

if task == "spaceship": # 21 subjects
    T = 250
elif task == "magic_carpet": # 24 subjects
    T = 201

# participant choice data
pfdir = "/.../muddled_models/results/" + task + "/choices/"

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

max_p, max_LL, LLs = MLE_magiccarpet(p_names, 
                                        observations.astype(int), 
                                        actions.astype(int), 
                                        learning,
                                        lower_bounds, 
                                        upper_bounds,
                                        n_starts,
                                        model=model,
                                        seed=s)

results_formatted = {"max_p": max_p,
                     "max_LL": max_LL,
                     "LLs": LLs}

save_obj(results_formatted, 
"/.../mfit/" + task + "/models/" + subs[s] + "_" + learning + "_" + model_ID)
