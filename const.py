#!/usr/bin/python
#coding: utf-8
from os.path import expanduser
HOME = expanduser("~")+'/'

"""Module to deal with constant, much more easier and every sub module can access it"""


USE_CUDA = True
DISPLAY = False

LEARNING_RATE = 1e-2

EPS_START = 0.9
EPS_END = 0.10

EPS_DECAY = 200
GAMMA = 0.85

NUM_EP = 100
BATCH_SIZE = 128
PRINT_INFO = 5

SIZE_MEMORY = 10000
EXPLO = 'eps-greedy'

MAIN_PATH = HOME+'Documents/enstage/'

RL_PATH = MAIN_PATH+'rl/'
MODEL_PATH = RL_PATH+'model/'
LOG_RL = RL_PATH+'Log/'
TIM_PATH = MAIN_PATH+'Baxter_Learning/Log/'

CV_EXPE = True
#False : Do only one Rl for testing
#True : Do multiple experiences to get stat

RESET_TIME = 1 #Estimation of the time to reset robot
ACTION_TIME = 0.25 #Estimation of the time to execute one action
