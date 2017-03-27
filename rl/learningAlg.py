#!/usr/bin/python
#coding: utf-8
from __future__ import division
import numpy as np
import torch

from numpy import random

from torch import nn
from torch.nn.init import xavier_normal

import torch.nn.functional as F
from torch.utils.serialization import load_lua
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from PIL import Image

from collections import namedtuple

import matplotlib.pyplot as plt

import const # const.USE_CUDA for exemple

class BaxterNotReady(Exception): pass

Transition = namedtuple('Transition', ('state', 'action', 'nextState', 'reward'))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

dtype = torch.cuda.FloatTensor if const.USE_CUDA else torch.FloatTensor

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priority = []
        self.position = 0
        self.tempIndex = None

    def push(self, currentState, action, nextState, reward, delta):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priority.append(None)

        self.memory[self.position] = Transition(currentState,action,nextState,reward)
        self.priority[self.position] = 1.0

        self.position = (self.position + 1) % self.capacity
        
        
    def sample(self, batchSize):
        p = softmax(np.array(self.priority))
        self.tempIndex = random.choice(len(self.memory), batchSize, p=p,replace=True)
        return np.array(self.memory)[self.tempIndex]
    
    def __len__(self):
        return len(self.memory)


class ReplayMemoryPrioritized(ReplayMemory):

    def __init__(self, capacity,epsilon=1e-5):
        super(ReplayMemoryPrioritized,self).__init__(capacity)
        self.eps = epsilon
        #smallest value, to avoid the probability to be 0.

    def push(self, currentState, action, nextState, reward, delta):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priority.append(None)

        self.memory[self.position] = Transition(currentState,action,nextState,reward)
        self.priority[self.position] = delta+self.eps

        self.position = (self.position + 1) % self.capacity
        
class RlContainer(object):

    def __init__(self, rlObj, timNet,exploration):
        self.rlObj = rlObj
        self.timNet = timNet
        self.mean, self.std = self.getMeanStdImages()
        self.exploration=exploration
        self.currentState = None #Need to remember state to save
        self.logState = []
        self.logAction = []

        
    def save(self, action, nextState, reward, estimation=None):
        #next state is an image, need the state corresponding
        nextState = self.timNet.forward(self.reformat(nextState))
        # print "self.currentState",self.currentState 
        # print "nextState",nextState
        # print "action", action
        # raw_input()
        self.rlObj.memory.push(self.currentState,action,nextState,reward,estimation)

    def getAction(self,img):
        self.stepsDone += 1
        img = self.reformat(img)
        state = self.timNet.forward(img)
        self.currentState = state


        if const.DISPLAY:
            print "state",state[0,0]

        if self.exploration=='boltzman':
            action,delta =  self.boltzman(state)
        elif self.exploration=='eps':
            action,delta =  self.epsGreedy(state)
        else:
            raise const.DrunkProgrammer("Cannot use {} exploration",format(self.exploration))

        if state.dim() == 2:
            assert state[0,0] == self.currentState[0,0]
        else:
            assert state[0,0,0,0] == self.currentState[0,0,0,0]

        self.logState.append(np.trunc(state[0,0]*100))
        self.logAction.append(action[0,0])
        
        return action,delta
    def boltzman(self,state):

        self.eps_threshold = 0
        x = Variable(state.type(dtype), volatile=True)
        res = self.rlObj.forward(x)
        Qs = F.softmax(res)

        act = np.random.choice(Qs.size(1), p=Qs[0,:].data.cpu().numpy())
        if self.stepsDone%const.PRINT_INFO==0:
            print "state",state,Qs,act,res.data 

        return torch.LongTensor([[act]]), res.data[0,act]

    def epsGreedy(self,state):
        
        sample = random.random()
        self.eps_threshold = const.EPS_END + (const.EPS_START - const.EPS_END) * np.exp(-1. * self.stepsDone / const.EPS_DECAY)
            
        Qs = self.rlObj.forward(Variable(state.type(dtype), volatile=True)).data
        if sample > self.eps_threshold:
            if self.stepsDone%const.PRINT_INFO==0:
                print "eps",self.eps_threshold 
                print "state",state[0,0]
                print "score", Qs

            action = Qs.max(1)[1].cpu()
            return action, Qs[0,action[0,0]]
        else:
            action = random.randint(0,self.rlObj.numOutput)
            return torch.LongTensor([[action]]), Qs[0,action]
            #Need to be the same shape as model.forward output, that's why there are [[]]

    def reformat(self,img):

        if img is None: raise BaxterNotReady("Relaunch Programm, camera might not be ready")

        reformatPipe = transforms.Compose([
            transforms.Scale(200),
            transforms.CenterCrop((200,200)),
            transforms.ToTensor()])

        img = Image.fromarray(img)
        img = reformatPipe(img)

        x = img.cpu().numpy()
        x = np.swapaxes(x,0,2)
        x = np.swapaxes(x,0,1)

        if const.DISPLAY:
            plt.imshow(x, interpolation='nearest')
            plt.show()
        
        img = (img - self.mean) / self.std

        if const.USE_CUDA:
            img = img.cuda()
        
        return img.unsqueeze(0)

    def getMeanStdImages(self):
        #We need mean and std calculated during timNet training
        meanTemp, stdTemp = load_lua(const.TIM_PATH+'meanStdImages.t7')
        l,w = meanTemp[1].size()

        mean = torch.zeros(3,l,w)
        std = torch.zeros(3,l,w)

        for i in range(3):
            mean[i] = meanTemp[i]
            std[i] = stdTemp[i]

        #range was 0,255, need to be 0,1
        mean /= 255
        std /= 255

        return mean,std

    def saveModel(self):
        self.rlObj.saveModel()

    def optimize(self, optimizer):
        self.rlObj.optimize(optimizer)

    @property
    def stepsDone(self):
        return self.rlObj.stepsDone

    @stepsDone.setter
    def stepsDone(self, value):
        self.rlObj.stepsDone = value

        
class DQN(nn.Module):
    def __init__(self, numInput, numOutput, N,memory='uniform'):
        super(DQN,self).__init__()

        if memory=='prioritized':
            assert isinstance(self,DQN_prioritized), "Can't use prioritize memory with {}, use DQN_prioritized".format(type(self))
            self.memory = ReplayMemoryPrioritized(const.SIZE_MEMORY)
        else:
            self.memory = ReplayMemory(const.SIZE_MEMORY)
            
        #Counter for action selection (more steps => less random action)
        self.stepsDone = 0
        self.currentLoss = [float('inf')]

        self.N = N #Number of neurons in hidden layer
        self.numOutput = numOutput
        self.numInput = numInput

        #NETWORK PARAMS
        #===============
        self.fc1 = nn.Linear(self.numInput,self.N)
        self.fc2 = nn.Linear(self.N,self.N)
        self.activation = nn.ReLU() #Different activation unit, just trying
        self.fc3 = nn.Linear(self.N,self.numOutput)
        #self.normIn = nn.BatchNorm1d(numInput)

        self.name='Dqn{}{}.state'.format(self.N, const.MODEL)

    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def optimize(self, optimizer):
        if len(self.memory)<const.BATCH_SIZE:
            return

        allTrans = self.memory.sample(const.BATCH_SIZE)
        batch = Transition(*zip(*allTrans))

        nonFinalMask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.nextState)))

        #Don't propagate into backward, this is only a "value" function
        nonFinalNextStates = Variable(torch.cat(tuple(s for s in batch.nextState if s is not None)).type(dtype), volatile=True)
        stateBatch = Variable(torch.cat(batch.state))
        actionBatch = Variable(torch.cat(batch.action))
        rewardBatch = Variable(torch.cat(batch.reward))

        if const.USE_CUDA:
            stateBatch = stateBatch.cuda()
            actionBatch = actionBatch.cuda()
        # Compute Q(s_t) and get the value associated with the action taken at this step
        # So those lines compute Q(s_t,a)
        Q_s_t = self.forward(stateBatch)
        stateActionValue = Q_s_t.gather(1, actionBatch).cpu()

        if const.REWARD:
            print "rewardBatch",rewardBatch 
        
        nextStateValue = Variable(torch.zeros(const.BATCH_SIZE))
        nextStateValue[nonFinalMask] = self.forward(nonFinalNextStates).max(1)[0].cpu()

        nextStateValue.volatile = False
        expectedStateActionValue = (nextStateValue * const.GAMMA) + rewardBatch

        # print "stateActionValue",stateActionValue.data[:5,0] 
        # print "expectedStateActionValue",expectedStateActionValue.data[0,:5]
        # raw_input()

        self.updateMemoryValue(stateActionValue,expectedStateActionValue) #Needed for prioritize memory
        
        loss = F.smooth_l1_loss(stateActionValue,expectedStateActionValue)
        self.currentLoss = loss.data
        
        optimizer.zero_grad()
        loss.backward()

        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        

    def updateMemoryValue(self, stateValue,expectedValue):
        pass

    def saveModel(self):
        torch.save(self.state_dict(), '{}{}'.format(const.MODEL_PATH,self.name))

        
class DQN_prioritized(DQN):
    def __init__(self, numInput, numOutput, N, memory='prioritized'):
        super(DQN_prioritized,self).__init__(numInput, numOutput, N,memory='prioritized')
        self.name='Dqn_prioritized{}{}.state'.format(self.N, const.MODEL)

    def updateMemoryValue(self, stateValue,expectedValue):
        delta = torch.abs(stateValue.data - expectedValue.data).pow(const.POWER)+self.memory.eps

        assert delta.size(0)==const.BATCH_SIZE, "Batch size must be unchanged\nDelta : {}\nBatch :{}".format(delta.size(0), const.BATCH_SIZE)
        
        for i in range(delta.size(0)):
            self.memory.priority[self.memory.tempIndex[i]] = delta[i,0]


class DQN_endToEnd(DQN_prioritized):
    def __init__(self, numInput, numOutput, N,memory='prioritized'):
        super(DQN_prioritized,self).__init__(numInput, numOutput, N,memory='prioritized')
        self.name='Dqn_endToEnd.state'
        self.fc3 = None

        numFilter = 32
        self.conv1 = nn.Conv2d(3, numFilter, 3)
        self.norm1 = nn.BatchNorm2d(numFilter)
        
        self.conv2 = nn.Conv2d(numFilter, numFilter*2, 3)
        self.norm2 = nn.BatchNorm2d(2*numFilter)

        self.conv3 = nn.Conv2d(numFilter*2, numFilter*4, 3)
        self.norm3 = nn.BatchNorm2d(numFilter*4)

        self.conv4 = nn.Conv2d(numFilter*4, numFilter*8, 3)
        self.norm4 = nn.BatchNorm2d(numFilter*8)

        self.conv5 = nn.Conv2d(numFilter*8,1,1)
        self.norm5 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(100,500)
        self.fc2 = nn.Linear(500,numOutput)


        #Xavier init
        self.apply(self.weights_init)

    def forward(self,x):

        batchSize = x.size(0)
        assert x.size(1)!=1, "TimNet shouldn't give representation, this is end to end q-learning"
        
        x = F.max_pool2d(F.relu(self.norm1(self.conv1(x))), (2,2))
        x = F.max_pool2d(F.relu(self.norm2(self.conv2(x))), (2,2))
        x = F.max_pool2d(F.relu(self.norm3(self.conv3(x))), (2,2))
        x = F.max_pool2d(F.relu(self.norm4(self.conv4(x))), (2,2))
        x = F.relu(self.norm5(self.conv5(x))).view(batchSize,100)
        x = self.fc2(F.relu(self.fc1(x)))

        batchSize = None
        return x

    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal(m.weight)
