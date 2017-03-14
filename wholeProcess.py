#!/usr/bin/python
#coding: utf-8
from __future__ import division

import const # all constants needed are there, the format to use constanst is : const.CONSTANT_NAME
#EX : const.USE_CUDA

from interfaceBax.env import LearnEnv, trueNet
from rl.learningAlg import DQN 
from Baxter_Learning.loadingLuaModel import loadModel
from plot import plotMeanScores, plotOneExp

from os.path import isfile
import subprocess
import rospy

import time
import random

from torch import optim

def doExpe(timNet, reset=True):
    rl = DQN(const.NUM_INPUT,const.NUM_ACTION,const.N,
             timNet,exploration=const.EXPLO)

    modelString = "{}Dqn{}.state".format(const.MODEL_PATH,const.N)
    
    if isfile(modelString):
        print "Model exists : LOADING MODEL"
        rl.load_state_dict(torch.load(modelString))
    else:
        print "Model doesn't exist : LEARNING FROM SCRATCH"

    optimizer = optim.RMSprop(rl.parameters(),lr=const.LEARNING_RATE)

    if const.USE_CUDA:
        timNet.cuda()
        rl.cuda()

    #Creating env
    env = LearnEnv(rl, optimizer)
    print("Running. Ctrl-c to quit")

    print "Begin Learning"
    logScores = env.run()

    if reset:
        subprocess.call(["rm",modelString])

    return logScores


rospy.init_node('Learning')

#timNet = loadModel("reprLearner1d.t7")
# timNet = loadModel('HeadSupervised.t7')
timNet = trueNet() #True position of the head, for testing

if const.NUM_EXPE>1:
    numMeasure = const.NUM_EP
    reset=True
    logMean = np.empty((const.NUM_EXPE,numMeasure))
    
    for i in range(const.NUM_EXPE):
        if i==const.NUM_EXPE-1:
            reset=False
        print "Experience n°{}, begin".format(i+1)
        logMean[i,:] = doExpe(timNet,reset=reset)
        print "Experience n°{}, over".format(i+1)
        print "Scores", logMean[i,:] 
        print "================================="
        print "=================================" 
    plotMeanScores(logMean)

else:
    meanScores = doExpe(timNet,reset=False)
    plotOneExp(meanScores)
