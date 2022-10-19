import numpy as np
import scipy.optimize as op
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
        "delta": 0.025,
        "bounds": [0.25 ,0.75]
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

    temp = models.learn_and_act(task, model)
    La = np.ones((T,2))

    po = np.zeros(2)
    pa = np.zeros(2)
    
    # Check which actions were taken and which outcomes observed
    for t in range(T):
        po = observations[t,:].astype(int)
        pa = actions[t,:].astype(int)

        a, o, pi, p_trans, p_r, GQ = temp.perform_trial(t, pa, po)

        La[t,0] = GQ[t, 0, pa[0]]
        La[t,1] = GQ[t, po[0]+1, pa[1]]

    return -np.sum(np.log(La)), GQ

def eval_LL_RL(params, observations, actions):
    """
    Evaluate likelihood for a sequence of actions given a sequence of observations (performed trial-wise).

    ~~~~~~
    INPUTS
    ~~~~~~
    params: model parameters
    observations: sequence of transition and outcome observations
    actions: sequence of taken actions
    learning: learning algorithm used (Default is "RL": Currently no other options for RL-based modeling)
    """

    # Specify task and generate (potential) observations
    T = observations.shape[0]
    task = {  
        "type": "drift",
        "T": T,
        "x": False,
        "r": True,
        "delta": 0.025,
        "bounds": [0.25, 0.75]
    }
    
    model = { # Model specification
        "act": "RL",
        "learn": "RL",
        "learn_transitions": False,
        "lr1": params[0],
        "lr2": params[1],
        "lam": params[2],
        "b1": params[3], 
        "b2": params[4],
        "p": params[5],
        "w": params[6],
        }

    temp = models.learn_and_act(task, model)
    La = np.ones((T,2))

    po = np.zeros(2)
    pa = np.zeros(2)
    
    # Check which actions were taken and which outcomes observed
    for t in range(T):
        po = observations[t,:].astype(int)
        pa = actions[t,:].astype(int)

        a, o, pi, p_trans, p_r, GQ = temp.perform_trial(t, pa, po)

        La[t,0] = GQ[t, 0, pa[0]]
        La[t,1] = GQ[t, po[0]+1, pa[1]]

    return -np.sum(np.log(La)),GQ

def MLE_procedure(params, observations, actions, learning, lower_bounds, upper_bounds, n_starts, model, mtype, seed=1):
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

    options = {"disp":True}

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