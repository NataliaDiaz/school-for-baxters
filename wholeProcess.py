#!/usr/bin/python
#coding: utf-8
from __future__ import division

import const # all constants needed are there, the format to use constanst is : const.CONSTANT_NAME
#EX : const.USE_CUDA

from interfaceBax.env import LearnEnv, TrueNet, DummyTimNet
from rl.learningAlg import DQN, DQN_prioritized, DQN_endToEnd, RlContainer
from Baxter_Learning.loadingLuaModel import loadModel,LuaModel
from plot import plotMeanScores, plotOneExp, saveTempLog, loadTempLog

from os.path import isfile
import subprocess
import rospy

import time
import random

from torch import optim

def doExpe(timNet, reset=True):

    #=============== CREATING MODEL HERE ================
    #====================================================
    if const.MODEL in ['auto1','auto2','repr','true','superv']:
        if const.MEMORY == 'uniform':
            rl = DQN(const.NUM_INPUT,const.NUM_ACTION,const.N)
            modelString = "{}Dqn{}.state".format(const.MODEL_PATH,const.N)
        elif const.MEMORY == 'prioritized':
            rl = DQN_prioritized(const.NUM_INPUT,const.NUM_ACTION,const.N)
            modelString = "{}Dqn_prioritized{}.state".format(const.MODEL_PATH,const.N)
        else:
            raise const.DrunkProgrammer("Wrong memory : {} doesn't exist".format(const.MEMORY))

    elif const.MODEL == 'end':
        rl = DQN_endToEnd(const.NUM_INPUT,const.NUM_ACTION, const.N)
        modelString = "{}Dqn_endToEnd.state".format(const.MODEL_PATH,const.N)
    else:
        raise const.DrunkProgrammer("Wrong model : {} doesn't exist".format(const.MODEL))
        
    if isfile(modelString) and const.LOADING:
        print "Model exists : LOADING MODEL"
        rl.load_state_dict(torch.load(modelString))
    else:
        print "Model doesn't exist (or const.LOADING is False) : LEARNING FROM SCRATCH "

    optimizer = optim.RMSprop(rl.parameters(),lr=const.LEARNING_RATE)

    if const.USE_CUDA:
        timNet.cuda()
        rl.cuda()

    rlObj = RlContainer(rl,timNet,const.EXPLO)
    
    #Creating env
    env = LearnEnv(rlObj, optimizer)
    print("Running. Ctrl-c to quit")

    print "Begin Learning"
    logScores = env.run()

    if reset:
        subprocess.call(["rm",modelString])

    return logScores

#======================================
#======================================

np.random.seed(1337)

rospy.init_node('Learning')

print "Whole Process : Model = ",const.MODEL 

if const.MODEL == 'auto1':
    timNet = loadModel("auto1d.t7")
elif const.MODEL == 'auto2':
    timNet = loadModel("auto1dAugm.t7")
elif const.MODEL == 'repr':
    timNet = loadModel("reprLearner1d.t7")
elif const.MODEL == 'superv':
    timNet = loadModel('HeadSupervised.t7')
elif const.MODEL == 'true':
    timNet = TrueNet() #True position of the head, for testing
elif const.MODEL == 'end':
    timNet = DummyTimNet() # Does nothing
else:
    raise const.DrunkProgrammer("Wrong model : {} doesn't exist".format(const.MODEL))

    
if const.NUM_EXPE>1:
    reset=True
    logMean, expeDone = loadTempLog()
    
    for i in range(expeDone, const.NUM_EXPE):
        if i==const.NUM_EXPE-1:
            reset=False
        print "Experience n°{}, begin".format(i+1)
        try:
            logMean[i,:] = doExpe(timNet,reset=reset)
        except:
            saveTempLog(logMean)
            raise
            
        
        print "Experience n°{}, over".format(i+1)
        print "Scores", logMean[i,:] 
        saveTempLog(logMean)
        print "================================="
        print "================================="
        
    plotMeanScores(logMean)

else:
    meanScores = doExpe(timNet,reset=False)
    plotOneExp(meanScores)
