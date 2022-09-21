import numpy as np
import os
from scipy.special import gammaln, psi
import matplotlib.pyplot as plt
import scipy.io as sio

def sample_trunc_norm(mu, var, min, max, size=1):
    # Sample from truncated normal. Simple resampling as long as sample lies outside of provided interval.
    samples = np.zeros(size)
    for i in range(size):
        s = np.random.normal(mu,np.sqrt(var))
        while (s < min) or (s > max):
            s = np.random.normal(mu,np.sqrt(var))
        samples[i] = s
    return samples

def KL_dir(p, q):
    """
    KL(p||q)
    Returns the Kullback-Leibler divergence of two Dirichlet distribution vectors p and q.
    Uses scipy.special.psi and .gammaln
    """
    p0 = np.sum(p)
    q0 = np.sum(q)
    
    return gammaln(p0) - gammaln(q0) - np.sum(gammaln(p)) + np.sum(gammaln(q)) + (p-q)@(psi(p) - psi(p0))

# Plotting functions -------------------------------------------------

def plot_results(T, model, p_trans, p_r, actions, observations, pi, Q):
     
    state2_choices = np.argwhere(observations[:,0]==0)
    state3_choices = np.argwhere(observations[:,0]==1)

    # Split actions up for states and buttons
    s2_choices = state2_choices.T[0]
    s2_actions = actions[state2_choices.T[0],:][:,1]
    s2_a0 = s2_choices[np.argwhere(s2_actions==0)]
    s2_a1 = s2_choices[np.argwhere(s2_actions==1)]

    s3_choices = state3_choices.T[0]
    s3_actions = actions[state3_choices.T[0],:][:,1]
    s3_a0 = s3_choices[np.argwhere(s3_actions==0)]
    s3_a1 = s3_choices[np.argwhere(s3_actions==1)]

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,11))
    
    # first level actions
    ax1.scatter(np.argwhere(actions[:,0]==0), actions[np.argwhere(actions[:,0]==0),0],color="r")
    ax1.scatter(np.argwhere(actions[:,0]==1), actions[np.argwhere(actions[:,0]==1),0],color="b")
    ax1.plot(p_trans[0,:],color="r",label="$p(s_3|a_1)$",alpha=0.7) # transitions
    ax1.plot(p_trans[1,:],color="b",label="$p(s_3|a_2)$",alpha=0.7)
    if model["learn"]=="MB":
        ax1.plot(pi[:,0,1,0] / np.sum(pi[:,0,:,0],1),linestyle="--",color="r")
        ax1.plot(pi[:,0,1,1] / np.sum(pi[:,0,:,1],1),linestyle="--",color="b")
    ax1.plot(Q[:,0,0],color="r",label="Q(s_1,a_1)",linewidth=2.5)
    ax1.plot(Q[:,0,1],color="b",label="Q(s_1,a_2)",linewidth=2.5)
    ax1.set_xlim([-0.5,T])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title("State 1")
    ax1.legend(loc="upper right",framealpha=0.25)

    ax2.scatter(s2_a0.T[0], np.zeros(s2_a0.T[0].shape[0]),color="green")
    ax2.scatter(s2_a1.T[0], np.ones(s2_a1.T[0].shape[0]),color="purple")
    ax2.plot(p_r[0,0,:],color="green",alpha=0.7)
    ax2.plot(p_r[0,1,:],color="purple",alpha=0.7)
    ax2.plot(Q[:,1,0],color="green",label="Q(s_2,a_3)",linewidth=2.5)
    ax2.plot(Q[:,1,1],color="purple",label="Q(s_2,a_4)",linewidth=2.5)
    ax2.set_xlim([-0.5,T])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_title("State 2")
    ax2.legend(loc="upper right",framealpha=0.25)

    ax3.scatter(s3_a0.T[0], np.zeros(s3_a0.T[0].shape[0]),color="black")
    ax3.scatter(s3_a1.T[0], np.ones(s3_a1.T[0].shape[0]),color="orange")
    ax3.plot(p_r[1,0,:],color="black",alpha=0.7)
    ax3.plot(p_r[1,1,:],color="orange",alpha=0.7)
    ax3.plot(Q[:,2,0],color="black",label="Q(s_3,a_5)",linewidth=2.5)
    ax3.plot(Q[:,2,1],color="orange",label="Q(s_3,a_6)",linewidth=2.5)
    ax3.set_xlim([-0.5,T])
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_title("State 3")
    ax3.legend(loc="upper right",framealpha=0.25)
    print("rewards:", np.sum(observations[state2_choices,1]), " + ", np.sum(observations[state3_choices,1]), " = ", np.sum(observations[state2_choices,1])+np.sum(observations[state3_choices,1]))

    plt.show()

def plot_results_AI(T, model, p_trans, p_r, actions, observations, pi, Q):
     
    state2_choices = np.argwhere(observations[:,0]==0)
    state3_choices = np.argwhere(observations[:,0]==1)

    # Split actions up for states and buttons
    s2_choices = state2_choices.T[0]
    s2_actions = actions[state2_choices.T[0],:][:,1] 
    s2_a0 = s2_choices[np.argwhere(s2_actions==0)]
    s2_a1 = s2_choices[np.argwhere(s2_actions==1)]

    s3_choices = state3_choices.T[0]
    s3_actions = actions[state3_choices.T[0],:][:,1] 
    s3_a0 = s3_choices[np.argwhere(s3_actions==0)]
    s3_a1 = s3_choices[np.argwhere(s3_actions==1)]

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,11))
    
    # first level actions
    ax1.scatter(np.argwhere(actions[:,0]==0), actions[np.argwhere(actions[:,0]==0),0],color="r")
    ax1.scatter(np.argwhere(actions[:,0]==1), actions[np.argwhere(actions[:,0]==1),0],color="b")
    ax1.plot(p_trans[0,:],color="r",label="$p(s_3|a_1)$",alpha=0.7) # transitions
    ax1.plot(p_trans[1,:],color="b",label="$p(s_3|a_2)$",alpha=0.7)
    ax1.plot(np.exp(Q[:,0,0]) / np.sum(np.exp(Q[:,0,:]),1),color="r",linewidth=2.5)
    ax1.plot(np.exp(Q[:,0,1]) / np.sum(np.exp(Q[:,0,:]),1),color="b",linewidth=2.5)
    ax1.set_xlim([-0.5,T])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title("State 1")
    ax1.legend(loc="upper right",framealpha=0.25)

    ax2.scatter(s2_a0.T[0], np.zeros(s2_a0.T[0].shape[0]),color="green")
    ax2.scatter(s2_a1.T[0], np.ones(s2_a1.T[0].shape[0]),color="purple")
    ax2.plot(p_r[0,0,:],color="green",alpha=0.7)
    ax2.plot(p_r[0,1,:],color="purple",alpha=0.7)
    # ax2.plot(np.exp(Q[:,1,0]) / np.sum(np.exp(Q[:,1,:]),1),color="green",linewidth=2.5)
    # ax2.plot(np.exp(Q[:,1,1]) / np.sum(np.exp(Q[:,1,:]),1),color="purple",linewidth=2.5)
    ax2.plot(pi[:,1,1,2] / np.sum(pi[:,1,:,2],1),label="$p(r|a_3)$",color="green",linewidth=2.5)
    ax2.plot(pi[:,1,1,3] / np.sum(pi[:,1,:,3],1),label="$p(r|a_4)$",color="purple",linewidth=2.5)
    ax2.set_xlim([-0.5,T])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_title("State 2")
    ax2.legend(loc="upper right",framealpha=0.25)

    ax3.scatter(s3_a0.T[0], np.zeros(s3_a0.T[0].shape[0]),color="black")
    ax3.scatter(s3_a1.T[0], np.ones(s3_a1.T[0].shape[0]),color="orange")
    ax3.plot(p_r[1,0,:],color="black",alpha=0.7)
    ax3.plot(p_r[1,1,:],color="orange",alpha=0.7)
    # ax3.plot(np.exp(Q[:,2,0]) / np.sum(np.exp(Q[:,2,:]),1),color="black",linewidth=2.5)
    # ax3.plot(np.exp(Q[:,2,1]) / np.sum(np.exp(Q[:,2,:]),1),color="orange",linewidth=2.5)
    ax3.plot(pi[:,1,1,4] / np.sum(pi[:,1,:,4],1),label="$p(r|a_5)$",color="black",linewidth=2.5)
    ax3.plot(pi[:,1,1,5] / np.sum(pi[:,1,:,5],1),label="$p(r|a_6)$",color="orange",linewidth=2.5)
    ax3.set_xlim([-0.5,T])
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_title("State 3")
    ax3.legend(loc="upper right",framealpha=0.25)
    print("rewards:", np.sum(observations[state2_choices,1]), " + ", np.sum(observations[state3_choices,1]), " = ", np.sum(observations[state2_choices,1])+np.sum(observations[state3_choices,1]))

    plt.show()

def plot_results_MPN(T, model, p_trans, p_r, actions, observations, pi, Q):

    state2_choices = np.argwhere(observations[:,0]==0)
    state3_choices = np.argwhere(observations[:,0]==1)

    # Split actions up for states and buttons
    s2_choices = state2_choices.T[0]
    s2_actions = actions[state2_choices.T[0],:][:,1]-2
    s2_a0 = s2_choices[np.argwhere(s2_actions==0)]
    s2_a1 = s2_choices[np.argwhere(s2_actions==1)]

    s3_choices = state3_choices.T[0]
    s3_actions = actions[state3_choices.T[0],:][:,1]-4
    s3_a0 = s3_choices[np.argwhere(s3_actions==0)]
    s3_a1 = s3_choices[np.argwhere(s3_actions==1)]

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,11))
    
    ax1.scatter(np.argwhere(actions[:,0]==0), actions[np.argwhere(actions[:,0]==0),0],color="r")
    ax1.scatter(np.argwhere(actions[:,0]==1), actions[np.argwhere(actions[:,0]==1),0],color="b")
    ax1.plot(np.exp(Q[:,0,0]) / np.sum(np.exp(Q[:,0,:]),1),color="r",linewidth=2.5)
    ax1.plot(np.exp(Q[:,0,1]) / np.sum(np.exp(Q[:,0,:]),1),color="b",linewidth=2.5)
    ax1.plot(p_trans[0,:-1],color="r",label="$p(s_3|a_1)$",alpha=0.7) # transitions
    ax1.plot(p_trans[1,:-1],color="b",label="$p(s_3|a_2)$",alpha=0.7)
    ax1.plot(pi[:,0,1,0] / np.sum(pi[:,0,:,0],1),linestyle="--",color="r")
    ax1.plot(pi[:,0,1,1] / np.sum(pi[:,0,:,1],1),linestyle="--",color="b")
    ax1.set_xlim([-0.5,T])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title("State 1 - p(S3|a) Red->S2, Blue->S3")
    ax1.legend(loc="upper right",framealpha=0.25)

    ax2.scatter(s2_a0.T[0], np.zeros(s2_a0.T[0].shape[0]),color="green")
    ax2.scatter(s2_a1.T[0], np.ones(s2_a1.T[0].shape[0]),color="purple")
    ax2.plot(p_r[0,0,:],color="green",label="$p(r|a_3)$",alpha=0.7) # Rewards
    ax2.plot(p_r[0,1,:],color="purple",label="$p(r|a_4)$",alpha=0.7)
    ax2.plot(pi[:,0,1,2] / np.sum(pi[:,0,:,2],1),color="green",linewidth=2.5)
    ax2.plot(pi[:,0,1,3] / np.sum(pi[:,0,:,3],1),color="purple",linewidth=2.5)
    ax2.set_xlim([-0.5,T])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_title("State 2")
    ax2.legend(loc="upper right",framealpha=0.25)

    ax3.scatter(s3_a0.T[0], np.zeros(s3_a0.T[0].shape[0]),color="black")
    ax3.scatter(s3_a1.T[0], np.ones(s3_a1.T[0].shape[0]),color="orange")
    ax3.plot(p_r[1,0,:],color="black",label="$p(r|a_5)$",alpha=0.7) # Rewards
    ax3.plot(p_r[1,1,:],color="orange",label="$p(r|a_6)$",alpha=0.7)
    ax3.plot(pi[:,0,1,4] / np.sum(pi[:,0,:,4],1),color="black",linewidth=2.5)
    ax3.plot(pi[:,0,1,5] / np.sum(pi[:,0,:,5],1),color="orange",linewidth=2.5)
    ax3.set_xlim([-0.5,T])
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_title("State 3")
    ax3.legend(loc="upper right",framealpha=0.25)
    print("rewards:", np.sum(observations[state2_choices,1]), " + ", np.sum(observations[state3_choices,1]), " = ", np.sum(observations[state2_choices,1])+np.sum(observations[state3_choices,1]))

    plt.show()

def compute_Daw_histograms(T, observations, actions, p_trans):

    reward = []
    repeat_stage1 = []
    transition = [] 
    
    for t in range(1,T):
        # Did common transition happen? no=0, yes=1. Depends on action
        transition.append(int(p_trans[actions[t-1,0],t-1] > 0.5) == observations[t-1,0])

        # Did we get a reward?
        if observations[t-1,0] == 0: # Stage 2
            reward.append(observations[t-1,1])
        else: # Stage 3
            reward.append(observations[t-1,1])
        # Did we repeat our stage1 action?
        repeat_stage1.append(actions[t,0] == actions[t-1,0])

    occurences = np.zeros((2,2,2))+1e-4 # common transition x rewarded 

    for t in range(T-1):
        if transition[t] and reward[t]:
            occurences[0,1,1] += 1
            if repeat_stage1[t]:
                occurences[1,1,1] += 1
        elif transition[t] and not reward[t]:
            occurences[0,1,0] += 1
            if repeat_stage1[t]:
                occurences[1,1,0] += 1
        elif not transition[t] and not reward[t]:
            occurences[0,0,0] += 1
            if repeat_stage1[t]:
                occurences[1,0,0] += 1
        elif not transition[t] and reward[t]:
            occurences[0,0,1] += 1
            if repeat_stage1[t]:
                occurences[1,0,1] += 1

    probs = np.zeros(4)
    probs[0] = occurences[1,1,1] / occurences[0,1,1] # common rewarded
    probs[1] = occurences[1,0,1] / occurences[0,0,1] # uncommon rewarded
    probs[2] = occurences[1,1,0] / occurences[0,1,0] # common unrewarded
    probs[3] = occurences[1,0,0] / occurences[0,0,0] # uncommon unrewarded

    return probs, np.array(reward), np.array(transition).astype(int), np.array(repeat_stage1).astype(int)


def load_obj(title, surprise=False):
    """
    Load an object that is either .mat or .pkl file
    """
    filename, file_extension = os.path.splitext(title)
    if file_extension == ".mat" and not surprise:
        out = sio.loadmat(title)
        # sample = out["sample_output"]
        # meta = {}
        #meta["prob_obs_init"] = out["C"][0][0][5][0][0][1]
        #meta["prob_regime_init"] = out["C"][0][0][5][0][0][2]
        #meta["prob_obs_change"] = out["C"][0][0][5][0][0][3]
        #%meta["prob_regime_change"] = out["C"][0][0][5][0][0][4]
        return out
    elif file_extension == ".mat" and surprise:
        out = sio.loadmat(title)
        meta = {}
        meta['time'] = out['SP_CD'][0][0][0][0]
        meta['sequence'] = out['SP_CD'][0][0][1][0]
        meta['hidden'] = out['SP_CD'][0][0][2][0]
        meta['predictive_surprise'] = out['SP_CD'][0][0][3][0]
        meta['bayesian_surprise'] = out['SP_CD'][0][0][4][0]
        meta['confidence_corrected_surprise'] = out['SP_CD'][0][0][5][0]
        return meta
    else:
        with open(title, 'rb') as f:
            return pickle.load(f)

def save_obj(obj, title):
    """
    Save an object as a pickle file
    """
    #with open(title + '.pkl', 'wb') as f:
    #    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    with open(title + '.mat', 'wb') as f:
        sio.savemat(f, obj)

def Many_Trials_Back(T, nback, observations, actions, p_trans):
    choice = np.zeros(T-nback)
    #rt = np.zeros(T-nback)

    choice_back = np.zeros((T-nback, nback))
    reward = np.zeros((T-nback, nback))
    transition = np.zeros((T-nback, nback))
    
    for t in range(nback,T): # Loop across trials

        choice[t-nback-1] = actions[t,0]
        #rt[t-nback-1] = rts[t]

        for back in range(nback): # Loop over nback trials

            # Check whether a reward was obtained
            reward[t-nback-1,back] = observations[t-back-1,1] 
            # Check whether a common transition occured
            transition[t-nback-1,back] = int(p_trans[actions[t-back-1,0]] > 0.5) == observations[t-back-1,0]
            # Check which initial-stage action was taken
            choice_back[t-nback-1,back] = actions[t-back-1,0]

    return choice, reward, transition.astype(int), choice_back