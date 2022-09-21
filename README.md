### Active inference and the two-step task
Code accompanying the paper 'Active inference and the two-step task'

To be completed in the upcoming weeks.

<br/>

##### STRUCTURE
```
AI2Step

    +- `models.py`: contains RL and active inference learning and action selection algorithms. 
    See below for example usage. Uses the following two files for support:

    Code for fitting and maximum likelihood estimation. These may be easily integrated into one file if preferred. 
    (Requires the data from the publications below.)
    +- `MCSP_evaluator.py`(Magic Carpet and Spaceship datasets)
    +- `Online_evaluator.py`
    +- `Shock_evaluator.py`
    
  ├── utils: helper functions    
      +- `twostep_environment.py`: contains the code for the two-step task 
      +- `twostep_support.py`: contains misc. support functions for the other code 

```


##### Package requirements:

numpy, scipy, pandas, os, sys, matplotlib

<br/>

#### DATA

Thankfully uses data from the following publications:

Kool, W., Cushman, F. A. & Gershman, S. J. When does model-based control pay off? PLoS computational biology 12,
e1005090 (2016).

Lockwood, P. L., Klein-Flügge, M. C., Abdurahman, A. & Crockett, M. J. Model-free decision making is prioritized when
learning to avoid harming others. Proc. Natl. Acad. Sci. 117, 27719–27730 (2020).

da Silva, C. F. & Hare, T. A. Humans primarily use model-based inference in the two-stage task. Nat. Hum. Behav. 4,
1053–1066 (2020).


#### Task and model example

```
    import models
    
    # Specify task and generate (potential) observations
    task = {  
        "type": "drift",
        "T": 200,
        "x": False,
        "r": True,
        "delta": 0.025,
        "bounds": [0.25, 0.75]
    }
    
    model = { # Model specification
        "act": "AI",
        "learn": "PSM",
        "learn_transitions": False,
        "lr": 1,
        "vunsamp": 0.2,
        "vsamp": 0.2,
        "vps": 0.2, 
        "gamma1": 5,
        "gamma2": 5,
        "lam": 0.1,
        "kappa_a": 0.1,
        "prior_r": 0.5
        }
        
    AI = models.learn_and_act(task, model, seed=1)
    actions, observations, beliefs, p_trans, p_r, Q = AI.perform_task()
        
```
