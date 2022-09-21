import numpy as np
from utils.twostep_support import *

def generate_observations_twostep(type="drift",T=50,delta=[99999],bounds=[0.25,0.75],change_transitions=True,change_rewards=True,seed=0):
    """
    Generates two-step task environment and simulates potential state transitions and outcomes.

    ~~~~~~
    INPUTS
    ~~~~~~
    type: task type determining the evolution of outcome probabilities; drift (as per original two-step task) or changepoint
    T: amount of trials
    change_transitions: determines whether transition probabilities change (True) or stay fixed (False)
    change_rewards: determines whether reward/outcome probabilities change or stay fixed
    """
    
    np.random.seed(seed)
    obs = np.zeros((3,2,T)).astype(int) # Two transition-buttons and four reward-buttons
    p_trans = np.zeros((2,T))
    p_r = np.zeros((2,2,T))

    # Initialize
    for k in range(2):
        p_r[k,0,0] = np.random.uniform(bounds[0],0.5)
        p_r[k,1,0] = np.random.uniform(0.5, bounds[1])
    p_trans[0,0] = 0.3 #np.random.uniform(0, 0.5)
    p_trans[1,0] = 0.7 #np.random.uniform(0.5, 1)

    # Fill rest of sequence
    if type == "changepoint":
        for t in range(1,T):
            if t in delta or t==0:
                for k in range(2):
                    if change_transitions:
                        new = np.random.uniform(0,1)
                        odds_old = p_trans[k,t-1]/(1-p_trans[k,t-1])
                        if t>0:
                            while (new/(1-new)) / odds_old < 4 and odds_old / (new/(1-new)) < 4: 
                                new = np.random.uniform(0,1)
                        p_trans[k,t] = np.copy(new)
                    else:
                        p_trans[k,t] = p_trans[k,t-1]

                    if change_rewards:
                        for l in range(2):
                            new = np.random.uniform(0,1)
                            odds_old = p_r[k,l,t-1]/(1-p_r[k,l,t-1])
                            if t>0:
                                while (new/(1-new)) / odds_old < 4 and odds_old / (new/(1-new)) < 4: 
                                    new = np.random.uniform(0,1)
                            p_r[k,l,t] = np.copy(new)
                    else:
                        p_r[k,:,t] = p_r[k,:,t-1]

            else: # Maintain probabilities
                p_trans[:,t] = p_trans[:,t-1]
                p_r[:,:,t] = p_r[:,:,t-1]

            for k in range(2): # Sample transitions
                obs[0,k,t] = np.random.binomial(1,p_trans[k,t])
                for l in range(2): # Sample rewards
                    obs[k+1,l,t] = np.random.binomial(1,p_r[k,l,t])

    elif type == "drift":
        for t in range(1,T):
            for k in range(2):
                if change_transitions:
                    p_trans[k,t] = sample_trunc_norm(
                                    mu=p_trans[k,t-1],
                                    var=delta**2,
                                    min=0,
                                    max=1)
                    if p_trans[k,t] < 0.25:
                        p_trans[k,t] = 0.25 + (0.25 - p_trans[k,t])
                    elif p_trans[k,t] > 0.75:
                        p_trans[k,t] = 0.75 - (p_trans[k,t] - 0.75)

                else:
                    p_trans[k,t] = p_trans[k,t-1]

                if change_rewards:
                    for l in range(2):
                        p_r[k,l,t] = sample_trunc_norm(
                                    mu=p_r[k,l,t-1],
                                    var=delta**2,
                                    min=0,
                                    max=1)
                        if p_r[k,l,t] < bounds[0]:
                            p_r[k,l,t] = bounds[0] + (bounds[0] - p_r[k,l,t])
                        elif p_r[k,l,t] > bounds[1]:
                            p_r[k,l,t] = bounds[1] - (p_r[k,l,t] - bounds[1])
                else:
                    p_r[k,:,t] = p_r[k,:,t-1]


            for k in range(2): # Sample transitions
                obs[0,k,t] = np.random.binomial(1,p_trans[k,t])
                for l in range(2): # Sample rewards
                    obs[k+1,l,t] = np.random.binomial(1,p_r[k,l,t])
    return obs, p_trans, p_r