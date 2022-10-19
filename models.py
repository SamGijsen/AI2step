import numpy as np
from scipy.special import gamma, digamma, gammaln, betaincinv
import scipy.io as sio

from utils.twostep_environment import *
from utils.twostep_support import *   

class learn_and_act():

    def __init__(self, task, model, seed=1):
        """
        DESCRIPTION: RL and Active inference agent 
            * Learns from two-step task observations
            * Acts on each stage to produce behaviour
        INPUT:  Task:
                    * type: str; drift, changepoint
                    * T: int; number of trials
                    * x: Boolean; Whether transition probabilities are resampled
                    * r: Boolean; Whether outcome probabilities are resampled
                    * delta: float; The volatility of task statistics (variance of Gaussian for drift-version)
                    * bounds: list of 2 floats; lower and upper bounds of (final-stage) outcome probabilities
                Model:
                    * act: RL or AI
                    if RL, then required arguments:
                        * learn: "RL"
                        * learn_transitions: False
                        * lr1: learning rate for first stage
                        * lr2: learning rate for second stage
                        * lam: lambda model parameter
                        * b1: temperature parameter for first stage softmax
                        * b2: temperature parameter for second stage softmax
                        * p: response stickiness parameter
                        * w: model-based weight
                    if AI, then:
                        * learn: "PSM"
                        * learn_transitions: False
                        * lr: learning rate
                        * vunsamp: volatility/decay rate for beliefs of unsampled actions
                        * vsamp: volatility/decay rate for beliefs of sampled actions
                        * vps: rate of predictive surprise influence on beliefs
                        * gamma1, gamma2: temperature parameter for first and second stage softmax, respectively
                        * kappa_a: precision of action-repetition habit
                        * prior_r: prior outcome probability

        OUTCOME:
                * A sequence of agent actions
                * A sequence of agent observations
                * A sequence of agent beliefs
        """

        self.task = task
        self.model = model
        self.seed = seed

        self.T = task["T"]
        self.Steps = 2

        # intialize
        self.pi = np.ones((self.T+1,2,2,6)) # T, 2 steps, Alpha/Beta, 6 rv

        if model["learn_transitions"]==False: # Set to correct TPs
            self.pi[:,:,0,0] *= 7
            self.pi[:,:,1,0] *= 3
            self.pi[:,:,0,1] *= 3
            self.pi[:,:,1,1] *= 7

        # Integrate prior parameters
        if model["act"] == "AI":
            prior_nu = 2
            self.prior = np.array([(1-self.model["prior_r"])*prior_nu, self.model["prior_r"]*prior_nu])
            for step in range(self.Steps):
                for a in range(2,6):
                    self.pi[:,step,:,a] = self.prior

        # Recordings
        self.actions = np.zeros((self.T,self.Steps)).astype(int)
        self.observations = np.zeros((self.T,self.Steps)).astype(int)
        self.GQ = np.zeros((self.T,3,2))
        self.prev_a = 999
        self.o = 999

        self.Qb = np.zeros((3,2))
        self.Qf = np.zeros((3,2))

        self.counts = np.zeros((2,2)) # 2 actions by 2 final states 
        self.tm = np.array([0.5, 0.5])

        np.random.seed(seed)

        self.obs, self.p_trans, self.p_r = generate_observations_twostep(
             type=task["type"], T=self.T, delta=task["delta"],bounds=task["bounds"],change_transitions=task["x"],seed=seed)


    def perform_task(self):

        for t in range(self.T):
            for step in range(self.Steps):
                if step == 0:
                    state = 0
                else:
                    state = o + 1
                    self.counts[a_t, o] += 1

                # Action selection --------------------------------------
                if self.model["act"] == "RL":
                    a_t, self.GQ[t,state,:] = self.action_selection_RL(state)

                elif self.model["act"] == "AI":
                    if step == 0:
                        gamma = self.model["gamma1"]
                    else:
                        gamma = self.model["gamma2"]

                    a_t, self.GQ[t,state,:] = self.action_selection_AI(t, state, gamma, self.model["learn"])

                if step == 0: 
                    self.prev_a = np.copy(a_t)

                # Interact ----------------------------------------------
                o = self.obs[state,a_t,t] 

                # Update ------------------------------------------------
                if self.model["learn"] == "RL": 
                    # Update Q-values
                    if step == 1:
                        self.Qf = self.update_SARSA(a_t, state, o)
                        self.Qb[1:,:] = np.copy(self.Qf[1:,:]) # MB equals MF for the final stage
                        self.Qb = self.update_MB()

                elif self.model["act"] == "AI" and step == 1:

                    ao = state*2
                    self.pi = self.PSM_learning(t, step, a_t+ao, o, self.pi, self.model["lr"], self.model["vunsamp"], self.model["vsamp"], self.model["vps"], 
                                            self.model["prior_r"], self.model["learn_transitions"])

                # Determine most likely transition matrix
                if (self.counts[0,0] + self.counts[1,1]) > (self.counts[0,1] + self.counts[1,0]):
                    self.tm = np.array([0.3, 0.7])
                if (self.counts[0,0] + self.counts[1,1]) < (self.counts[0,1] + self.counts[1,0]):
                    self.tm = np.array([0.7, 0.3])
                if (self.counts[0,0] + self.counts[1,1]) == (self.counts[0,1] + self.counts[1,0]):
                    self.tm = np.array([0.5, 0.5])

                self.actions[t,step] = a_t
                self.observations[t,step] = o #+ state*2


        return self.actions, self.observations, self.pi, self.p_trans, self.p_r, self.GQ


    def perform_trial(self, t, pa, po):
        """
        Advances task by one trial by advancing through by steps.
        Differences to running a full task:
        - actions are provided (pa: [1x2])
        - observations are provided (po: [1x2])
        - particularly interesting are the distributions over actions/policies, rather than actions themselves
        """

        for step in range(self.Steps):
            if step == 0:
                state = 0
            else:
                state = o + 1
                self.counts[a_t, o] += 1

            # Action selection --------------------------------------
            if self.model["act"] == "RL":
                a_t, self.GQ[t,state,:] = self.action_selection_RL(state)

            elif self.model["act"] == "AI":
                if step == 0:
                    gamma = self.model["gamma1"]
                else:
                    gamma = self.model["gamma2"]

                a_t, self.GQ[t,state,:] = self.action_selection_AI(t, state, gamma, self.model["learn"])

            a_t = pa[step]

            if step == 0: 
                self.prev_a = np.copy(a_t)

            # Interact (Fixed) --------------------------------------
            o = po[step]

            # Update ------------------------------------------------
            if self.model["learn"] == "RL": 
                # Update Q-values
                if step == 1:
                    self.Qf = self.update_SARSA(a_t, state, o)
                    self.Qb[1:,:] = np.copy(self.Qf[1:,:]) # MB equals MF for the final stage
                    self.Qb = self.update_MB()

            elif self.model["act"] == "AI" and step == 1:

                ao = state*2
                self.pi = self.PSM_learning(t, step, a_t+ao, o, self.pi, self.model["lr"], self.model["vunsamp"], self.model["vsamp"], self.model["vps"], 
                                        self.model["prior_r"], self.model["learn_transitions"])

            # Determine most likely transition matrix
            if (self.counts[0,0] + self.counts[1,1]) > (self.counts[0,1] + self.counts[1,0]):
                self.tm = np.array([0.3, 0.7])
            if (self.counts[0,0] + self.counts[1,1]) < (self.counts[0,1] + self.counts[1,0]):
                self.tm = np.array([0.7, 0.3])
            if (self.counts[0,0] + self.counts[1,1]) == (self.counts[0,1] + self.counts[1,0]):
                self.tm = np.array([0.5, 0.5])

            self.actions[t,step] = a_t
            self.observations[t,step] = o #+ state*2


        return self.actions, self.observations, self.pi, self.p_trans, self.p_r, self.GQ

    def PSM_learning(self, t, step, a, o, pi, lr, vunsamp, vsamp, vps, prior_r=0.5, learn_transitions=False):

        # Predictive-Surprise Modulated learning
        prior_nu = 2

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


    def update_SARSA(self, a2, state, o):
        # SARSA(\lambda): temporal difference learning
        # Q contains our Q-values: Q_TD(s,a)

        lr1 = self.model["lr1"]
        lr2 = self.model["lr2"]
        lam = self.model["lam"]

        PE_i = self.Qf[state,a2] - self.Qf[0,self.prev_a]
        PE_f = o - self.Qf[state,a2]
        self.Qf[0,self.prev_a] = self.Qf[0,self.prev_a] + lr1*PE_i + lr1*lam*PE_f

        self.Qf[state,a2] = self.Qf[state, a2] + lr2*PE_f

        return self.Qf


    def update_MB(self):

        self.Qb[0,:] = (1-self.tm) * np.max(self.Qb[1,:]) + self.tm*np.max(self.Qb[2,:])

        return self.Qb


    def compute_drift_EFE(self, t, step, state, lr, vunsamp, vsamp, vps, ao, lam, prior_r=0.5, learn_transitions=False):
        # Empirically compute EFE for a state

        G = np.zeros(2)
        for a in range(2):

            Gi = np.zeros(2)
            for o in range(2):
                pi_temp = np.copy(self.pi)
                Q_pi = self.PSM_learning(t, step, a+ao, o, pi_temp, lr, vunsamp, vsamp, vps, prior_r, learn_transitions)

                G[a] -= KL_dir(self.pi[t,step,:,a+ao], Q_pi[t+1,1,:,a+ao]) * (self.pi[t,step,o,a+ao]/np.sum(self.pi[t,step,:,a+ao])) # Intrinsic term


            G[a] -= 2*lam*np.log(self.pi[t,step,1,a+ao]/np.sum(self.pi[t,step,:,a+ao])) # Extrinsic term

        return G


    def action_selection_AI(self, t, state, gamma, learning, learn_transitions=False):
        """
        ~~~~~~
        INPUTS
        ~~~~~~
        t: current timepoint
        state: current state
        pi: belief distributions
        lr: learning rate (model parameter)
        vunsamp: rate of decay for beliefs on unsampled actions (model parameter)
        vsamp: rate of decay for beliefs of sampled actions (model parameter)
        vps: rate of influence of predictive surprise on beliefs of sampled actions (model parameter)
        lam: precision of prior preferences (model parameter)
        kappa_a: precision of 'action-stickiness' habit (model parameter)
        prev_a: previous first-stage action taken by the agent
        learning": type of learning algorithm
        gamma: softmax inverse temperature parameter controlling for decision noise (model parameter)
        prior_r: \alpha / (\alpha + \beta) of prior Beta-distribution, i.e. the prior reward probability(model parameter)
        learn_transitions: whether state-transition probabilities are known to be 0.3 and 0.7 
        """

        lr = self.model["lr"]
        vunsamp = self.model["vunsamp"]
        vsamp = self.model["vsamp"]
        vps = self.model["vps"]
        lam = self.model["lam"]
        kappa_a = self.model["kappa_a"]
        prior_r = self.model["prior_r"]

        if state == 0:
            step = 0
            deep = 1 # Flag deep-policy
        else:   
            step = 1
            deep = 0

        G_s0, G_s1, G_s2 = np.zeros(2), np.zeros(2), np.zeros(2)

        if state == 1 or deep:
            ao = 2
            G_s1 = self.compute_drift_EFE(t, 1, 1, lr, vunsamp, vsamp, vps, ao, lam, prior_r, learn_transitions)

        if state == 2 or deep:
            ao = 4
            G_s2 = self.compute_drift_EFE(t, 1, 2, lr, vunsamp, vsamp, vps, ao, lam, prior_r, learn_transitions)

        if state == 0:
            G = np.zeros(2)

            # Habits
            E = np.zeros(2)
            if t > 0:
                E[self.prev_a] += -np.exp(kappa_a)
                E[1-self.prev_a] += -np.exp(-kappa_a)

            G_s0 = np.concatenate((G_s1, G_s2))
            G[0] = np.dot(G_s0, np.array([ # Action 0
            1-self.tm[0], 1-self.tm[0], self.tm[0], self.tm[0]]))
            G[1] = np.dot(G_s0, np.array([ # Action 1
            1-self.tm[1], 1-self.tm[1], self.tm[1], self.tm[1]]))

            G = G + E

        elif state == 1:
            G = G_s1
        elif state == 2:
            G = G_s2

        Gg = np.clip(-G * gamma,-500,500)
        probs = np.exp(Gg)/np.sum(np.exp(Gg))

        return np.random.choice(np.arange(2),p=probs), probs       


    def action_selection_RL(self, state):
        # Softmax with step-dependent Beta (inverse temperature) parameters

        b1 = self.model["b1"]
        b2 = self.model["b2"]
        w = self.model["w"]
        p = self.model["p"]

        rep = np.zeros(2)
        if self.prev_a<2:
            rep[self.prev_a] = 1

        probs = np.zeros(2)

        if state == 0:
            for a in range(2):
                probs[a] = np.exp(b1 * (w*self.Qb[state,a] + (1-w)*self.Qf[state,a] + p*rep[a])) \
                / np.sum(np.exp(b1* (w*self.Qb[state,:] + (1-w)*self.Qf[state,:] + p*rep[:])))
        else:
            for a in range(2):
                probs[a] = np.exp(b2*self.Qf[state,a]) / np.sum(np.exp(b2*(self.Qf[state,:])))

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

