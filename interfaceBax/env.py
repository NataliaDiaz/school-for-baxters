#!/usr/bin/python
#coding: utf-8
import torch
import numpy as np

import time
import rospy
from baxter_interface import Head, Limb, Gripper, CameraController, RobotEnable

from arm_scenario_experiments import baxter_utils

from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from geometry_msgs.msg import Point
from baxter_core_msgs.msg import EndpointState
from objects import SuperButton

from cv_bridge import CvBridge

import subprocess
from itertools import count
import curses

import const # const.USE_CUDA for exemple

from scipy.stats import norm
from itertools import product

class BaxterProblem(Exception): pass

class LearnEnv1D(object):
    def __init__(self, rl, optimizer):
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

        self.currentImage = None
        self.limit = 199 #Limit of step before reseting

        self.rl = rl
        self.head = Head()

        self.bridge = CvBridge()
        self.imageSub = rospy.Subscriber("cameras/head_camera_2/image",Image,self.imageFromCamera)

        #Constants
        self.optimizer = optimizer

        if const.RBF:
            self.setRbfParams()
            

    def setRbfParams():
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
        elif const.TASK ==2 :
            self.head.set_pan(0.75*np.random.choice([-1,1]))
        else:
            raise const.DrunkProgrammer("Other task doesn't exist for environnement 1D, try 3D ?")

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

        else:
            raise const.DrunkProgrammer("Task cannot be {}".format(const.TASK))
                
        return reward, done

    def run(self):

        logScores = []
        meanScores = []
        countEp = 0

        while ~rospy.is_shutdown() and countEp<const.NUM_EP:
            self.reset()
            totalReward = 0

            for t in count():
                    
                action, estimation = self.rl.getAction(self.currentImage)
                reward, done = self.step(action[0,0])
                totalReward += reward
 
                reward = torch.Tensor([reward])
                self.rl.save(action, self.currentImage, reward, estimation)
                self.rl.optimize(self.optimizer)

                if done or t>self.limit:
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


class LearnEnv3D(LearnEnv1D):
    def __init__(self, rlObj, optimizer):
        super(LearnEnv3D, self).__init__(rlObj, optimizer)
        subprocess.call(['rosrun', 'baxter_tools', 'tuck_arms.py', '-u'])

        self.imageSub = rospy.Subscriber("cameras/head_camera_2/image",Image,self.imageFromCamera)
        self.objects = {}

        self.buttonPosDefault = [0.65,0.20,0.85]

        self.leftArm = Limb('left')
        self.posSub = rospy.Subscriber("/robot/limb/left/endpoint_state",EndpointState, self.gripperPosFromSub)
        
        self.neutralPosition = np.array([0.64,0.20,0.31])
        self.currentPosition = self.neutralPosition
        
        self.beginningPosition = self.leftArm.joint_angles()
        self.ee_orientation = baxter_utils.get_ee_orientation(self.leftArm)
        
        self.limit = const.LIMIT #Limit of step before reseting

        self.actions = [np.array(i) for i in product( (0.05,-0.05,0), repeat=3) ]
        self.actions.pop()
        print "actions",self.actions 

    def spawnButton(self):
        buttonPoint = Point(*self.buttonPosDefault)
        #if button already exists, it's from an other session, need to delete.
        self.delButton()
        self.add_object( SuperButton('button1').spawn(buttonPoint))

    def delButton(self):
        subprocess.call(["rosservice","call","gazebo/delete_model","model_name: 'button1'"])
        time.sleep(const.ACTION_TIME)
            
    def gripperPosFromSub(self, message):
        self.currentPositionFromSub = (message.pose.position.x,message.pose.position.y,message.pose.position.z)
        
    def add_object(self, obj):
        self.objects[obj.gazebo_name] = obj

    def del_objects(self):
        for obj in self.objects.keys(): self.objects[obj].delete()

    def reset(self):
        subprocess.call(['rosrun', 'baxter_tools', 'tuck_arms.py', '-u'])
        time.sleep(const.ACTION_TIME)

        self.spawnButton()

        joints = None
        while not joints:
            try:
                joints = baxter_utils.IK(self.leftArm, self.neutralPosition, self.ee_orientation)
            except rospy.ServiceException:
                continue
            
        self.leftArm.move_to_joint_positions(joints)
        self.currentPosition = self.neutralPosition
        self.rl.timNet.state = self.neutralPosition
        return joints
        
    def move(self, action):

        self.currentPosition = self.currentPosition+action
        joints = None
        trial = 0
        while not joints:
            if trial == 2:
                print "waiting a little bit more for IK service" 
                time.sleep(const.RESET_TIME)
            if trial > 3:
                return False
            try:
                joints = baxter_utils.IK(self.leftArm, self.currentPosition, self.ee_orientation)
            except rospy.ServiceException :
                trial += 1

        self.leftArm.move_to_joint_positions(joints)
        self.rl.timNet.state = self.currentPosition #Needed for dummy timNet, other timNet doesn't use this value
        return True
        
    def step(self,action):

        #Apply action 
        if const.NO_BRAIN:
            #action = int(raw_input("Todo (25 down) : "))
            action = const.NO_BRAIN

        # Check if action could be done (Sometimes it fails because the point the robot need to reach is too far)
        if not self.move(self.actions[action]): #smarter way to select action, instead of doing if elif elif etc ...
            print "Hand cannot reach, if you see this many times, you should restart ros, something is wrong with IK" 
            reward = -5
            done = False
            return reward, done

        print self.objects['button1'].positionFromSub, self.buttonPosDefault[:2]
        button_moved = not np.allclose(self.objects['button1'].positionFromSub, self.buttonPosDefault[:2], rtol=0, atol=2e-2) #Check if button moved during step or not, just to be sure, you don't take z into account, because since the button is spawn above the table, the spawn z IS different from the actual position
            
        #Get reward or not
        if self.objects['button1'].is_pressed():
            reward = 20
            done = True
        elif button_moved :
            print "baxter moved button" 
            reward = -20
            done = True
        else:

            if np.allclose(self.currentPosition, self.currentPositionFromSub,rtol=0, atol=1e-2):
                self.currentPosition = np.array(self.currentPositionFromSub)
                reward = 0
                done = False

            else:
                print "boing, wall" 
                self.move(-1*self.actions[action]) #smarter way to select action, instead of doing if elif elif etc ...
                reward = -5
                done = False

        return reward,done
