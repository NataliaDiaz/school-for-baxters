#!/usr/bin/python
#coding: utf-8
"""
Module to deal with constant, much more easier and every sub module can access it
Easy to change model, parameters, learning_rate, cuda, experiments etc ...

Also path for loading, saving etc ...
"""
from os.path import expanduser
import torch

class DrunkProgrammer(Exception):pass

HOME = expanduser("~")+'/'

MODEL ='repr'
MEMORY = 'prioritized' #choice between 'uniform' and 'prioritized'

STORE_STATE_MEAN = True
NUM_RBF = 4
RBF = False
#rbf is just a way to represent the state,
#instead of being a single value, it is values taken from a mixture of gaussian distribution

TASK = 3
#1st task => look at your left, don't look at the right, there is a monster
#2nd task => don't look at your left or right, look in front of you
#3rd task => 3D controlling hand with button. Action available are easy (only descending and lateral)
#4th task => 3D controlling hand with button action available are in all directions NOT AVAILABLE

MAX_PAN = 1.2
MIDDLE_PAN = 0.1

LOADING = False
#Do you want to load the model (DQN) or learn from scratch ?

USE_CUDA = True
torch.cuda.set_device(0)
device = torch.cuda.current_device()+1
print "Current device is the {}{}".format(device,'nd' if device==2 else 'st')

# To GPU or not GPU

DISPLAY = False  #display image and representation associated at every timestep
NO_BRAIN = False
#NO_BRAIN = 3
#False : you want baxter to act normally
#int (in action range): baxter does only this action

SHOW_REWARD = False #show rewardbatch at every timestep

LEARNING_RATE = 0.01
GAMMA = 0.90
POWER = 0.5 #For prioritized memory, higher value => higher probability to replay 'surprising' reward. 0 => uniform random

NUM_EP = 100
BATCH_SIZE = 80
PRINT_INFO = 20

if RBF:
    NUM_INPUT = NUM_RBF

N = 50 #number of hidden neuron

SIZE_MEMORY = 10000

EXPLO = 'boltzman' #explo can be 'boltzman' or 'eps'

#For epsilon-greedy
EPS_START = 0.9
EPS_END = 0.10
EPS_DECAY = 300

MAIN_PATH = './'

RL_PATH = MAIN_PATH+'rl/'
MODEL_PATH = RL_PATH+'model/'
LOG_RL= RL_PATH+'Log/'
TIM_PATH = MAIN_PATH+'baxter_representation_learning_1D/Log/'

MODEL_3D = MAIN_PATH+'models_3D/'

#Those values are the one used in resNet, since we use this model to calculate representation, we need to use the same values
MEAN_MODEL = [0.485, 0.456, 0.406]
STD_MODEL = [0.229, 0.224, 0.225]

NUM_EXPE = 20
#1 : Do only one Rl for testing, the model is saved
#>1 : Do multiple experiences to get stats and plots, only last model is saved

if TASK>2:
    RESET_TIME = 2 #Estimation of the time to reset robot
    ACTION_TIME = 0.1 #Estimation of the time to execute one action
    NUM_INPUT = 3

    if TASK==4:
        raise NotImplemented("Try Task 3 at the moment")
        NUM_ACTION = 26
        LIMIT = 399 #Timeout, the game is over, new game incomming

    elif TASK==3:

        DEFAULT_BUTTON_POS = [0.605240, -0.0128, 0.77]

        NEUTRAL_POSITION = [(0.50,0.60),(0.07,0.17),0]
        GOOD_Z_POS = -0.09 # to avoid the hand slowly sinking into the table
        
        RESET_TIME = 2 #Estimation of the time to reset robot
        ACTION_TIME = 2.5 #Estimation of the time to execute one a
        NUM_ACTION = 5
        LIMIT = 299 #Timeout, the game is over, new game incomming

        LIM_INF_X = 0.42
        LIM_SUP_X = 0.80

        LIM_INF_Y = -0.2
        LIM_SUP_Y = 0.50

    else:
        raise DrunkProgrammer("Only task 3 and 4 exists")
else:
    RESET_TIME = 1 #Estimation of the time to reset robot
    ACTION_TIME = 0.40 #Estimation of the time to execute one action
    LIMIT = 199
    NUM_ACTION = 2
    NUM_INPUT = 1

    
