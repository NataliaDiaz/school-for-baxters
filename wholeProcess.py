#!/usr/bin/python
#coding: utf-8
from __future__ import division

import const # all constants needed are there, the format to use constanst is : const.CONSTANT_NAME
#EX : const.USE_CUDA

from interfaceBax.env import LearnEnv, TrueNet, DummyTimNet
from rl.learningAlg import DQN, DQN_prioritized, DQN_endToEnd, RlContainer
from Baxter_Learning.loadingLuaModel import loadModel,LuaModel
from plot import plotMeanScores, plotOneExp

from os.path import isfile
import subprocess
import rospy

import time
import random

from torch import optim

def doExpe(timNet, reset=True):
    # rl = DQN(const.NUM_INPUT,const.NUM_ACTION,const.N)
    # modelString = "{}Dqn{}.state".format(const.MODEL_PATH,const.N)

    rl = DQN_prioritized(const.NUM_INPUT,const.NUM_ACTION,const.N)
    modelString = "{}Dqn_prioritized{}.state".format(const.MODEL_PATH,const.N)

    # rl = DQN_endToEnd(const.NUM_INPUT,const.NUM_ACTION, const.N)
    # modelString = "{}Dqn_endToEnd.state".format(const.MODEL_PATH,const.N)

    if isfile(modelString):
        print "Model exists : LOADING MODEL"
        rl.load_state_dict(torch.load(modelString))
    else:
        print "Model doesn't exist : LEARNING FROM SCRATCH"

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

# np.random.seed(42)

rospy.init_node('Learning')

#timNet = loadModel("auto1d.t7")
timNet = loadModel("reprLearner1d.t7")
# timNet = loadModel('HeadSupervised.t7')
# timNet = TrueNet() #True position of the head, for testing

#timNet = DummyTimNet()

if const.NUM_EXPE>1:
    numMeasure = const.NUM_EP
    reset=True
    logMean = np.empty((const.NUM_EXPE,numMeasure))
    
    for i in range(const.NUM_EXPE):
        if i==const.NUM_EXPE-1:
            reset=False
        print "Experience n°{}, begin".format(i+1)
        try:
            logMean[i,:] = doExpe(timNet,reset=reset)
        except RuntimeError:
            print i+" experience"
            raise RuntimeError("Cuda failed")
        
        print "Experience n°{}, over".format(i+1)
        print "Scores", logMean[i,:] 
        print "================================="
        print "=================================" 
    plotMeanScores(logMean)

else:
    meanScores = doExpe(timNet,reset=False)
    plotOneExp(meanScores)
