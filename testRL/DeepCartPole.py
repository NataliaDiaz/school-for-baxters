import numpy as np
from numpy import random

import matplotlib.pyplot as plt


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import gym 

from collections import namedtuple
from itertools import count

from os.path import isfile


# Allow for model and data to be used by GPU
class Variable(torch.autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if isinstance(data,np.ndarray) :
            data = torch.Tensor(data)

        #torch works only with the first dimension being a batch
        #if the array is only 1-D, means there is no batch dim
        #but can't check if data.dim > 1
        if data.dim()==1:
            data = data.unsqueeze(0)

        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

#======================
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None) 
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return np.array(self.memory)[random.choice(len(self.memory), batch_size)]
    
    def __len__(self):
        return len(self.memory)

class DeepPolicyGradient(nn.Module):
    def __init__(self, numInput, numOutput, N):
        super(DeepPolicyGradient,self).__init__()

        #Defining the network, simple one hidden layer to test 
        self.N = N #Number of neurons in hidden layer
        self.numOutput = numOutput
        
        self.fc1 = nn.Linear(numInput,N)
        self.fc2 = nn.Linear(N,numOutput)

        self.norm1 = nn.BatchNorm1d(N)

        #Different activation unit, just trying
        self.activation = nn.ReLU()

        #stuff for action selection
        self.stepsDone = 0
        
    def forward(self,x):

        #x = self.norm1(self.activation(self.fc1(x)))
        x = self.activation(self.fc1(x))

        x = self.fc2(x)
        
        return x

    def getAction(self,state):

        sample = random.random()
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.stepsDone / EPS_DECAY)
        self.stepsDone += 1

        if sample > self.eps_threshold:
            Qs = self.forward(Variable(state, volatile=True)).data
            return self.forward(Variable(state, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.LongTensor([[random.randint(0,self.numOutput)]]) 
            #Need to be the same shape as model.forward, that's why there are [[]]

    def optimize(self, memory, optimizer):
        if len(memory)<BATCH_SIZE:
            return

        allTrans = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*allTrans))

        nonFinalMask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.nextState)))

        if USE_CUDA:
            nonFinalMask = nonFinalMask.cuda()

        #Don't propagate into backward, this is only a "value" function
        nonFinalNextStates = Variable(torch.cat(tuple(s for s in batch.nextState if s is not None)), volatile=True)
        stateBatch = Variable(torch.cat(batch.state))
        actionBatch = Variable(torch.cat(batch.action))
        rewardBatch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t) and get the value associated with the action taken at this step
        # So those lines compute Q(s_t,a)
        Q_s_t = self.forward(stateBatch)
        stateActionValue = Q_s_t.gather(1, actionBatch)
        
        nextStateValue = Variable(torch.zeros(BATCH_SIZE))
        nextStateValue[nonFinalMask] = self.forward(nonFinalNextStates).max(1)[0]

        
        #now we need to retropropagate the gradient
        nextStateValue.volatile = False
        expectedStateActionValue = (nextStateValue * GAMMA) + rewardBatch

        loss = F.smooth_l1_loss(stateActionValue,expectedStateActionValue)

        self.currentLoss = loss.data


        # if ep > 3:
        #     print("self.currentLoss",self.currentLoss)
        #     print("stateActionValue",stateActionValue[:5])
        #     print("expectedStateActionValue",expectedStateActionValue.permute(1,0)[:5])
        #     raw_input()
        
        
        optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        

MODEL_PATH = 'model/'
USE_CUDA = False


env = gym.make('CartPole-v0')
numAction = env.action_space.n
numObs = env.observation_space.shape[0]

Transition = namedtuple('Transition', ('state', 'action', 'nextState', 'reward'))

EPS_START = 0.9
EPS_END = 0.05

EPS_DECAY = 1000
GAMMA = 0.82

RENDER=True

NUM_EP = 100
NUM_ITE = 200
BATCH_SIZE = 128
PLOT_ITE = 20
N = 101 #number of hidden neuron

SIZE_MEMORY = 10000

if isfile("{}DeepCartPole{}.model".format(MODEL_PATH,N)):
    print "Model exists : LOADING MODEL"
    dpg = torch.load("{}DeepCartPole{}.model".format(MODEL_PATH,N))
else:
    print "Model doesn't exist : CREATING"
    dpg = DeepPolicyGradient(numObs,numAction,N)

memory = ReplayMemory(SIZE_MEMORY)

optimizer = optim.RMSprop(dpg.parameters(),lr=1e-2)

if USE_CUDA:
    dpg.cuda()

logScores = []
meanScores = []
maxScores = []
step = 0

for ep in range(1,NUM_EP):

    currentScreen = env.reset()
    state = torch.Tensor(currentScreen).unsqueeze(0)

    for t in count():
        step += 1
        if RENDER:
            env.render()
        action = dpg.getAction(state)

        
        
        currentScreen, reward, done, info = env.step(action[0,0]) #[0,0] because action is a 2d tensor after forward
        if done:
            nextState = None
        else:
            nextState = torch.Tensor(currentScreen).unsqueeze(0)

        reward = torch.Tensor([reward])
        memory.push(state,action,nextState,reward)

        state = nextState

        dpg.optimize(memory, optimizer)
        
        if done:
            logScores.append(t)
            #print "Score de : {}".format(t)
            break

           
    if ep%PLOT_ITE==0 and step > BATCH_SIZE:
        logTemp = logScores[ep-PLOT_ITE:ep]
        print("ep",ep)
        print("logScores",logTemp)
        print("eps",dpg.eps_threshold)
        print "loss",dpg.currentLoss[0]

        y = np.mean(logTemp)
        meanScores.append(y)
        maxScore = np.max(logTemp)
        maxScores.append(maxScore)

        print "Mean Scores", meanScores
        print "Max Scores", maxScores
        
torch.save(dpg,"{}DeepCartPole{}.model".format(MODEL_PATH,N))
