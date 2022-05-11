import numpy as np
from scipy.special import gamma, digamma, gammaln, betaincinv
import scipy.io as sio

from twostep_environment import *
from twostep_support import *   

def learn_and_act(task, model, seed=1):
    
    T = task["T"]
    if model["learn"] == "MPN":
        N = model["N"]
    else:
        N = 1
        e = np.zeros((3,2))
    Steps = 2

    # intialize
    theta = np.ones((T+1,2,2,N,6)) # T, 2 steps, Alpha/Beta, N Particles, 6 rv
    w = np.ones((T+1,2,N))
    pi = np.ones((T+1,2,2,6)) # T, 2 steps, Alpha/Beta, 6 rv
    if model["learn_transitions"]==False: # Set to correct TPs
        pi[:,:,0,0] *= 7
        pi[:,:,1,0] *= 3
        pi[:,:,0,1] *= 3
        pi[:,:,1,1] *= 7
        theta[:,:,0,:,0] *= 7
        theta[:,:,1,:,0] *= 3
        theta[:,:,0,:,1] *= 3
        theta[:,:,1,:,1] *= 7

    # Integrate prior parameters
    prior = np.array([(1-model["prior_r"])*model["prior_nu"], model["prior_r"]*model["prior_nu"]])
    for step in range(Steps):
        for a in range(2,6):
            pi[:,step,:,a] = prior

    # Recordings
    actions = np.zeros((T,Steps)).astype(int)
    observations = np.zeros((T,Steps)).astype(int)
    GQ = np.zeros((T,3,2))
    prev_a = 999
    o=999

    Qb = np.zeros((3,2))
    Qf = np.zeros((3,2))

    counts = np.zeros((2,2)) # 2 actions by 2 final states 
    tm = np.array([0.5, 0.5])

    np.random.seed(seed)

    obs, p_trans, p_r = generate_observations_twostep(type=task["type"], T=T, delta=task["delta"],change_transitions=task["x"],seed=seed)

    for t in range(T):
        for step in range(2):
            if step == 0:
                state = 0
            else:
                state = o + 1
                counts[a_t, o] += 1

            # Action selection --------------------------------------
            if model["act"] == "RL":
                a_t, gq = action_selection_RL(state, model["b1"], model["b2"], model["w"], model["p"], Qb, Qf, prev_a)

            elif model["act"] == "AI":
                if step == 0:
                    gamma = model["gamma1"]
                else:
                    gamma = model["gamma2"]
                a_t, GQ[t,state,:] = action_selection_AI(
                    t, state, pi, tm, model["v"], model["lam"], model["kappa_a"], prev_a,
                    model["learn"], gamma, model["prior_nu"], model["prior_r"], model["learn_transitions"])

            if step == 0: 
                prev_a = np.copy(a_t)

            # Interact ----------------------------------------------
            o = obs[state,a_t,t] 

            # Update ------------------------------------------------
            if model["learn"] == "RL": 
                # Update Q-values
                if step == 1:
                    Qf = update_SARSA(state, model["lr1"], model["lr2"], model["lam"], Qf, prev_a, a_t, o)
                    Qb[1:,:] = np.copy(Qf[1:,:]) # MB equals MF for the final stage
                    Qb = update_MB(Qb, tm)

            elif model["act"] == "AI" and step == 1:
                if state == 1:
                    ao = 2
                if state == 2:
                    ao = 4
                pi = PSM_learning(t, step, a_t+ao, o, pi, tm, model["v"], model["prior_nu"], model["prior_r"], model["learn_transitions"])

            # Determine most likely transition matrix
            if (counts[0,0] + counts[1,1]) > (counts[0,1] + counts[1,0]):
                tm = np.array([0.3, 0.7])
            if (counts[0,0] + counts[1,1]) < (counts[0,1] + counts[1,0]):
                tm = np.array([0.7, 0.3])
            if (counts[0,0] + counts[1,1]) == (counts[0,1] + counts[1,0]):
                tm = np.array([0.5, 0.5])

            actions[t,step] = a_t
            observations[t,step] = o #+ state*2

        if model["learn"] == "RL":
            GQ[t,:,:] = model["w"]*Qb + (1-model["w"])*Qf

    return actions, observations, theta, w, pi, p_trans, p_r, GQ


def PSM_learning(t, step, a, o, pi, tm, lr, vunsamp, vsamp, vps, prior_nu=2, prior_r=0.5, learn_transitions=False):
    # Predictive-Surprise Modulated learning

    prior = np.array([(1-prior_r)*prior_nu, prior_r*prior_nu])

    copy = np.array([0,1,2,3,4,5])
    decay = np.array([2,3,4,5])

    PS = -np.log(pi[t,1,o,a]/np.sum(pi[t,1,:,a]))
    m = vps/(1-vps) # uses vps for PS modulation
    gamma = (m*PS)/(1+m*PS)

    # Decay unsampled arms by vunsamp
    pi[t+1,1,:,copy] = np.copy(pi[t,1,:,copy])
    pi[t+1,1,:,decay] = (1-vunsamp)*pi[t,1,:,decay] + vunsamp*prior

    # Sampled arm
    # first, decay by vasmp
    pi[t+1,1,:,a] = (1-vsamp)*pi[t,1,:,a] + vsamp*prior
    # second, decay by gamma=f(vps)
    pi[t+1,1,:,a] = (1-gamma)*pi[t+1,1,:,a]
    # third, increment sampled action by lr
    pi[t+1,1,o,a] += lr

    return pi

def smile_learning(t, step, a, o, pi, tm, v, prior_r=0.5, learn_transitions=False):

    prior = np.array([(1-prior_r)*2, prior_r*2])

    SBF = prior_r / (pi[t,1,o,a]/np.sum(pi[t,1,:,a]))
    m = v/(1-v)
    gamma = (m*SBF) / (1+m*SBF)

    pi[t+1,1,:,:] = np.copy(pi[t,1,:,:])
    #pi[t+1,1,:,[2,3,4,5]] = (1-v)*pi[t,1,:,[2,3,4,5]] + v*prior
    pi[t+1,1,:,a] = (1-gamma)*pi[t,1,:,a] + gamma*prior # We can do this across arms (2: instead of a if beliefs generalize across arms)
    pi[t+1,1,o,a] += 1

    return pi

def leaky_learning(t, step, a, o, pi, tm, v, learn_transitions=False):

    pi[t+1,1,:,:] = np.copy(pi[t,1,:,:])
    pi[t+1,1,:,a] = (1-v)*pi[t,1,:,a]
    pi[t+1,1,o,a] += 1

    return pi

def update_SARSA(state, lr1, lr2, lam, Q, a1, a2, o):
    # SARSA(\lambda): temporal difference learning
    # Q contains our Q-values: Q_TD(s,a)

    PE_i = Q[state,a2] - Q[0,a1]
    PE_f = o - Q[state,a2]
    Q[0,a1] = Q[0,a1] + lr1*PE_i + lr1*lam*PE_f

    Q[state,a2] = Q[state,a2] + lr2*PE_f

    return Q 

def update_MB(Q, tm):

    Q[0,:] = (1-tm) * np.max(Q[1,:]) + tm*np.max(Q[2,:])

    return Q

def compute_drift_EFE(t, step, state, pi, tm, lr, vunsamp, vsamp, vps, ao, lam, prior_nu, prior_r=0.5, learn_transitions=False):
    # Empirically compute EFE for a state

    G = np.zeros(2)
    for a in range(2):

        Gi = np.zeros(2)
        for o in range(2):
            pi_temp = np.copy(pi)
            Q_pi = PSM_learning(t, step, a+ao, o, pi_temp, tm, lr, vunsamp, vsamp, vps, prior_nu, prior_r, learn_transitions)

            G[a] -= KL_dir(pi[t,step,:,a+ao], Q_pi[t+1,1,:,a+ao]) * (pi[t,step,o,a+ao]/np.sum(pi[t,step,:,a+ao])) # Intrinsic term


        G[a] -= 2*lam*np.log(pi[t,step,1,a+ao]/np.sum(pi[t,step,:,a+ao])) # Extrinsic term

    return G


def action_selection_AI(t, state, pi, tm, lr, vunsamp, vsamp, vps, lam, kappa_a, prev_a, learning, gamma=5, prior_nu=2, prior_r=0.5, learn_transitions=False):
    """
    If learn_transitions==True, we want to add an info-gain term over the first-to-second
    stage transitions. 
    """
    if state == 0:
        step = 0
        deep = 1 # Flag deep-policy
    else:   
        step = 1
        deep = 0

    G_s0, G_s1, G_s2 = np.zeros(2), np.zeros(2), np.zeros(2)

    if state == 1 or deep:
        ao = 2
        G_s1 = compute_drift_EFE(t, 1, 1, pi, tm, lr, vunsamp, vsamp, vps, ao, lam, prior_nu, prior_r, learn_transitions)

    if state == 2 or deep:
        ao = 4
        G_s2 = compute_drift_EFE(t, 1, 2, pi, tm, lr, vunsamp, vsamp, vps, ao, lam, prior_nu, prior_r, learn_transitions)

    if state == 0:
        G = np.zeros(2)

        # Habits
        E = np.zeros(2)
        if t > 0:
            E[prev_a] += -np.exp(kappa_a)
            E[1-prev_a] += -np.exp(-kappa_a)

        G_s0 = np.concatenate((G_s1, G_s2))
        G[0] = np.dot(G_s0, np.array([ # Action 0
        1-tm[0], 1-tm[0], tm[0], tm[0]]))
        G[1] = np.dot(G_s0, np.array([ # Action 1
        1-tm[1], 1-tm[1], tm[1], tm[1]]))

        G = G + E

    elif state == 1:
        G = G_s1
    elif state == 2:
        G = G_s2

    Gg = np.clip(-G * gamma,-500,500)

    return np.random.choice(np.arange(2),p=np.exp(Gg)/np.sum(np.exp(Gg))), Gg        


def action_selection_RL(state, b1, b2, w, p, Qb, Qf, prev_a):
    # Softmax with step-dependent Beta (inverse temperature) parameters

    rep = np.zeros(2)
    if prev_a<2:
        rep[prev_a] = 1

    probs = np.zeros(2)

    if state == 0:
        for a in range(2):
            probs[a] = np.exp(b1 * (w*Qb[state,a] + (1-w)*Qf[state,a] + p*rep[a])) / np.sum(np.exp(b1* (w*Qb[state,:] + (1-w)*Qf[state,:] + p*rep[:])))
    else:
        for a in range(2):
            probs[a] = np.exp(b2*Qf[state,a]) / np.sum(np.exp(b2*(Qf[state,:])))

    return int(np.random.choice(np.arange(2), p=probs)), probs

def update_transitions(t, pi, a, o, learn_transitions=False):
    if learn_transitions:
        if t>0:
            pi[t,0,:,:] = np.copy(pi[t-1,0,:,:])
        pi[t,0,o,a] += 1
    else:
        pi[t,0,0,0:2] = [7,3]
        pi[t,0,1,0:2] = [3,7]

    return pi
