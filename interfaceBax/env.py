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

class BaxterProblem(Exception): pass

class trueNet(object):
    def __init__(self):
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
        return torch.Tensor(np.array([self.head.pan()])).unsqueeze(0)
    def cuda(self):
        pass

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

        self.rl = rl
        self.head = Head()

        self.bridge = CvBridge()
        self.imageSub = rospy.Subscriber("cameras/head_camera_2/image",Image,self.imageFromCamera)
        self.currentImage = None

        #Constants
        self.optimizer = optimizer

    def reset(self):
        self.head.set_pan(0)
        time.sleep(const.RESET_TIME)

    def imageFromCamera(self,data):
        self.currentImage = self.bridge.imgmsg_to_cv2(data, "rgb8")
    def step(self, action):

        MAX_PAN = 1.2
        currentPan = self.head.pan()

        # action = 0
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

        if self.head.pan() >= MAX_PAN:
            reward = 100
            done = True
        elif self.head.pan() <= -MAX_PAN:
            reward = -100
            done = True
            
        else:
            reward = 0
            done = False

        return reward, done
        
        
    def run(self):

        logScores = []
        meanScores = []
        countEp = 0

        while ~rospy.is_shutdown() and countEp<const.NUM_EP:
            self.reset()
            totalReward = 0

            for t in count():
                self.rl.pan = self.head.pan()
                action = self.rl.getAction(self.currentImage)
                reward, done = self.step(action[0,0])
                totalReward += reward

                reward = torch.Tensor([reward])
                self.rl.save(action, self.currentImage, reward)
                self.rl.optimize(self.optimizer)

                if done:
                    countEp += 1
                    logScores.append(totalReward-t)
                    print "Over, score : ", logScores[-1]
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

    

