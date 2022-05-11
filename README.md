### Active inference and the two-step task
Code accompanying the paper 'Active inference and the two-step task'

To be completed in the upcoming weeks.

<br/>

##### STRUCTURE

##### Task and Models

`twostep_environment.py`: contains the code for the two-step task 

`twostep_support.py`: contains misc. support functions for the other code 

`twostep_learning_acting.py`: contains RL and active inference learning and action selection algorithms

<br/>

##### Fitting via Maximum Likelihood Estimation

These may be easily integrated into one file if preferred. Require the data from the publications below.

`MagicCarpet_Spaceship_evaluator.py`

`Online_evaluator.py`

`Shock_evaluator.py`

<br/>

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
