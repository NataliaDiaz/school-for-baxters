#!/usr/bin/python
#coding: utf-8
from __future__ import division

import const # const.USE_CUDA for exemple

from interfaceBax.env import * #LearnerEnv
from rl.learningAlg import * #DQN 
from Baxter_Learning.loadingLuaModel import loadModel

from os.path import isfile

import time
import random

random.seed(1337)
rospy.init_node('Learning')

timNet = loadModel("reprLearner1d.t7")
# timNet = loadModel('HeadSupervised.t7')
# timNet = trueNet() #True position of the head, for testing


numInput = 1
numAction = 2
numObs = 1
N = 100 #number of hidden neuron

rl = DQN(numInput,numAction,N,timNet,exploration=const.EXPLO)

modelString = "{}Dqn{}.state".format(const.MODEL_PATH,N)

if isfile(modelString):
    print "Model exists : LOADING MODEL"
    rl.load_state_dict(torch.load(modelString))
else:
    print "Model doesn't exist : LEARNING FROM SCRATCH"

optimizer = optim.RMSprop(rl.parameters(),lr=const.LEARNING_RATE)

if const.USE_CUDA:
    timNet.cuda()
    rl.cuda()

logScores = []
meanScores = []
maxScores = []
step = 0

#Creating env

env = LearnEnv(rl, optimizer)
print("Running. Ctrl-c to quit")

print "Begin Learning" 
meanScores,maxScores = env.run()


# if const.CV_EXPE:
#     numExpe = 10
# else:
#     numExpe = 1
#     meanScores,maxScores = env.run()
        
# for i in range(numExpe):
#     pass


