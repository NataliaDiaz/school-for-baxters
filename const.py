#!/usr/bin/python
#coding: utf-8
from os.path import expanduser
HOME = expanduser("~")+'/'

"""Module to deal with constant, much more easier and every sub module can access it"""
USE_CUDA = True
DISPLAY = False

LEARNING_RATE = 0.001

GAMMA = 0.85

NUM_EP = 12
BATCH_SIZE = 40
PRINT_INFO = 50

NUM_INPUT = 1
NUM_ACTION = 2
NUM_OBS = 1
N = 20 #number of hidden neuron

SIZE_MEMORY = 1000

EXPLO = 'boltzman'
EPS_START = 0.9
EPS_END = 0.10
EPS_DECAY = 300

MAIN_PATH = HOME+'Documents/enstage/'

RL_PATH = MAIN_PATH+'rl/'
MODEL_PATH = RL_PATH+'model/'
LOG_RL = RL_PATH+'Log/'
TIM_PATH = MAIN_PATH+'Baxter_Learning/Log/'

NUM_EXPE = 20
#1 : Do only one Rl for testing, the model is saved
#>1 : Do multiple experiences to get stats and plots, only last model is saved


RESET_TIME = 1 #Estimation of the time to reset robot
ACTION_TIME = 0.25 #Estimation of the time to execute one action
