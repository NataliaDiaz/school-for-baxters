#!/usr/bin/python
#coding: utf-8
import torch
import numpy as np

import time
import rospy
from baxter_interface import Head, Limb, Gripper, CameraController, RobotEnable

from sensor_msgs.msg import Image
from std_msgs.msg import Empty

from cv_bridge import CvBridge

import subprocess
from itertools import count

import const # const.USE_CUDA for exemple

from scipy.stats import norm

class BaxterProblem(Exception): pass

class DummyTimNet(object):
    def __init__(self):
        pass

    def forward(self,x):
        return x

    def cuda(self):
        pass

    def setMinMaxRbf(self, img1,img2):
        assert const.DrunkProgrammer("Cannot use Dummy timnet and rbf")

    def calcRBFvalue(self, x):
        numRbf = const.NUM_RBF
        means = np.linspace(self.minRepr,self.maxRepr,numRbf+2)
        means = means[1:-1] #Getting rid of first and last elements, because you don't want gaussians on the extremum
        std = np.sqrt(np.abs(means[0]-means[1])/2)
        values = [norm.pdf(x,means[i],std) for i in range(numRbf)]

        return torch.Tensor(np.array(values)).unsqueeze(0)

        
    def __call__(self,x):
        return self.forward(x)

class TrueNet(DummyTimNet):
    def __init__(self):
        super(TrueNet,self).__init__()
        ready = False
        while not ready:
            try:
                self.head = Head()
            except OSError:
                print "Waiting for Baxter to be ready, come"
                time.sleep(1)
                continue
            else:
                ready=True

    def forward(self,*args):
        x = self.head.pan()
        if const.RBF:
            x = self.calcRBFvalue(x)
            return x
        
        return torch.Tensor(np.array([x])).unsqueeze(0)

    def setMinMaxRbf(self, img1,img2):
        self.minRepr = -1.3
        self.maxRepr = 1.3


    def __call__(self,x):
        return self.forward(x)



class LearnEnv(object):
    def __init__(self, rl, optimizer):
        #rospy.Subscriber('/environment/reset', Empty, self.reset_callback, queue_size = 1)
        subprocess.call(["rosrun", "baxter_tools","enable_robot.py","-e"])

        ready = False
        while not ready:
            try:
                head = Head()
            except OSError:
                print "Waiting for Baxter to be ready, come on"
                time.sleep(1)
                continue
            else:
                ready=True

        head.set_pan(1)
        head.set_pan(-1)

        # limb = Limb('left')
        # limb.move_to_joint_positions({'left_s0': 0.6})
        # limb.move_to_joint_positions({'left_s1': -0.2})
        # limb.move_to_joint_positions({'left_s0': -0.1})
        # limb.move_to_joint_positions({'left_e0': -1.4})
        # limb.move_to_joint_positions({'left_e1': 1})

        self.currentImage = None

        self.rl = rl
        self.head = Head()

        self.bridge = CvBridge()
        self.imageSub = rospy.Subscriber("cameras/head_camera_2/image",Image,self.imageFromCamera)

        #Constants
        self.optimizer = optimizer

        if const.RBF:
            time.sleep(1)
            self.head.set_pan(-1.3)
            img1 = self.currentImage
            self.head.set_pan(1.3)
            img2 = self.currentImage
            assert not(img1 is None and img2 is None), "Need to wait more before retrieving images"

            self.rl.setMinMaxRbf(img1,img2)

            
    def reset(self):

        if const.TASK == 1:
            self.head.set_pan(1.28)
        else:
            self.head.set_pan(0.75*np.random.choice([-1,1]))
        time.sleep(const.RESET_TIME)

    def imageFromCamera(self,data):
        self.currentImage = self.bridge.imgmsg_to_cv2(data, "rgb8")
    def step(self, action):

        currentPan = self.head.pan()

        if const.NO_BRAIN:
            action = 1
        # print "============================" 
        # print "currentPan, Before :",currentPan 
        # print "action",action
        # raw_input()
        if action==0:
            self.head.set_pan(currentPan+0.1)
        elif action==1:
            self.head.set_pan(currentPan-0.1)
        else:
            raise BaxterProblem("I can't do that.")

        time.sleep(const.ACTION_TIME)
        # print "currentPan, After :",self.head.pan()
        if np.abs(self.head.pan()-currentPan) < 0.08 :
            print "Small Lag, waiting for baxter" 
            time.sleep(const.ACTION_TIME)

        if np.abs(self.head.pan()-currentPan) < 0.08 :
            raise BaxterProblem("Head Desynchronize, you might want to wait a little more between action and reset.\nIf you pressed Ctrl-c, this is normal.")

        currentPan = self.head.pan()

        if const.TASK==1: #first task => look at your left, don't look at the right, there is a monster
            if currentPan >= const.MAX_PAN:
                reward = 20
                done = True
            elif currentPan <= -const.MAX_PAN:
                reward = -20
                done = True
            else:
                reward = 0
                done = False
            return reward, done

        elif const.TASK==2: #second task => don't look at your left or right, look in front of you
            if currentPan >= const.MAX_PAN or currentPan <= -const.MAX_PAN:
                reward = -20
                done = True
            elif currentPan <= const.MIDDLE_PAN and currentPan >= -const.MIDDLE_PAN:
                reward = 20
                done = True
            else:
                reward = 0
                done = False
            return reward, done

        else:
            raise const.DrunkProgrammer("Task cannot be {}".format(const.TASK))
                
        
    def run(self):

        logScores = []
        meanScores = []
        countEp = 0

        while ~rospy.is_shutdown() and countEp<const.NUM_EP:
            self.reset()
            totalReward = 0

            for t in count():
                self.rl.pan = self.head.pan()
                action, estimation = self.rl.getAction(self.currentImage)
                reward, done = self.step(action[0,0])
                totalReward += reward
 
                reward = torch.Tensor([reward])
                self.rl.save(action, self.currentImage, reward, estimation)
                self.rl.optimize(self.optimizer)

                if done or t>199:
                    countEp += 1
                    logScores.append(totalReward*3-t)
                    print "Trial {} Over, score : {}".format(countEp, logScores[-1])

                    if type(self.rl.logState[-1]) is float:
                        print "logState 5 last state",self.rl.logState[-5:]
                    print "logAction 5 last action",self.rl.logAction[-5:]
                    self.rl.logState = []
                    break

            self.rl.saveModel()

            if countEp>=const.PRINT_INFO:
                logTemp = logScores[countEp-const.PRINT_INFO:countEp]

            if countEp%const.PRINT_INFO==0:
                print "countEp",countEp
                print "logScores",logTemp
                #print "loss",self.rl.currentLoss[0]

                #print "Mean Scores", meanScores

        return logScores

    

