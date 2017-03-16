#!/usr/bin/python
#coding: utf-8
from __future__ import division

import numpy as np
import torch

from numpy import random

from torch import nn
import torch.nn.functional as F
from torch.utils.serialization import load_lua

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



class Variable(torch.autograd.Variable):
    
    def __init__(self, data, *args, **kwargs):

        if isinstance(data,np.ndarray) :
            data = torch.Tensor(data)

        #torch works only with the first dimension being a batch
        #if the array is only 1-D, means there is no batch dim
        #but can't check if data.dim > 1
        if data.dim()==1:
            data = data.unsqueeze(0)

        if const.USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

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
        self.tempIndex = random.choice(len(self.memory), batchSize, p=p)
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
        
    
# ???????????????????
# class RlModule():
#     def __init__():
#         pass
# ???????????????????

class DQN(nn.Module):
    def __init__(self, numInput, numOutput, N, timNet,exploration,memory='uniform'):

        self.mean, self.std = self.getMeanStdImages()
        super(DQN,self).__init__()
        self.timNet = timNet

        self.exploration=exploration

        if memory=='prioritized':
            assert isinstance(self,DQN_prioritized), "Can't use prioritize memory with DQN, use DQN_prioritized"
            self.memory = ReplayMemoryPrioritized(const.SIZE_MEMORY)
        else:
            self.memory = ReplayMemory(const.SIZE_MEMORY)
            
        self.currentState = None #Need to remember state to save

        #Counter for action selection (more steps => less random action)
        self.stepsDone = 0
        self.currentLoss = [float('inf')]

        #NETWORK PARAMS
        #===============
        self.N = N #Number of neurons in hidden layer
        self.numOutput = numOutput

        self.fc1 = nn.Linear(numInput,N)
        self.activation = nn.ReLU() #Different activation unit, just trying
        self.fc2 = nn.Linear(N,numOutput)
        #self.normIn = nn.BatchNorm1d(numInput)

        self.name='Dqn{}.state'.format(self.N)

    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, action, nextState, reward, estimation=None):
        #next state is an image, need the state corresponding
        nextState = self.timNet.forward(self.reformat(nextState))
        # print "self.currentState",self.currentState 
        # print "nextState",nextState
        # print "action", action
        # raw_input()
        self.memory.push(self.currentState,action,nextState,reward,estimation)

    def getAction(self,img):
        self.stepsDone += 1
        img = self.reformat(img)
        state = self.timNet.forward(img)
        self.currentState = state

        if self.exploration=='boltzman':
            return self.boltzman(state)
        else:
            return self.epsGreedy(state)

    def boltzman(self,state):

        self.eps_threshold = 0
        res = self.forward(Variable(state, volatile=True))
        Qs = F.softmax(res)

        act = np.random.choice(Qs.size(1), p=Qs[0,:].data.cpu().numpy())
        if self.stepsDone%100==0:
            print "state",state,Qs,act,res.data 

        return torch.LongTensor([[act]]), res.data[0,act]

    def epsGreedy(self,state):
        
        sample = random.random()
        self.eps_threshold = const.EPS_END + (const.EPS_START - const.EPS_END) * np.exp(-1. * self.stepsDone / const.EPS_DECAY)
        if const.DISPLAY:
            print "state",state[0,0]
            
        Qs = self.forward(Variable(state, volatile=True)).data
        if sample > self.eps_threshold:
            if self.stepsDone%100==0:
                print "eps",self.eps_threshold 
                print "state",state[0,0]
                print "score", Qs

            action = Qs.max(1)[1].cpu()
            return action, Qs[0,action[0,0]]
        else:
            action = random.randint(0,self.numOutput)
            return torch.LongTensor([[action]]), Qs[0,action]
            #Need to be the same shape as model.forward output, that's why there are [[]]

    def optimize(self, optimizer):
        if len(self.memory)<const.BATCH_SIZE:
            return

        allTrans = self.memory.sample(const.BATCH_SIZE)
        batch = Transition(*zip(*allTrans))

        nonFinalMask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.nextState)))

        if const.USE_CUDA:
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
        
        nextStateValue = Variable(torch.zeros(const.BATCH_SIZE))
        nextStateValue[nonFinalMask] = self.forward(nonFinalNextStates).max(1)[0]

        #now we need to retropropagate the gradient
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
        torch.save(self.state_dict(), '{}{}'.format(const.MODEL_PATH,self.name))

        
class DQN_prioritized(DQN):
    def __init__(self, numInput, numOutput, N, timNet,exploration,memory='prioritized'):
        super(DQN_prioritized,self).__init__(numInput, numOutput, N, timNet,exploration,memory='prioritized')
        self.name='Dqn_prioritized{}.state'.format(self.N)

    def updateMemoryValue(self, stateValue,expectedValue):
        delta = torch.abs(stateValue.data - expectedValue.data)+self.memory.eps

        assert delta.size(0)==const.BATCH_SIZE, "Batch size must be unchanged\nDelta : {}\nBatch :{}".format(delta.size(0), const.BATCH_SIZE)
        
        for i in range(delta.size(0)):
            self.memory.priority[self.memory.tempIndex[i]] = delta[i,0]

