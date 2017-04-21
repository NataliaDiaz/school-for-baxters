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

import const # const.USE_CUDA for exemple

from scipy.stats import norm
from itertools import product

class BaxterProblem(Exception): pass

class LearnEnv1D(object):
    def __init__(self, rl, optimizer):

        try:
            _ = subprocess.check_output(["rosrun", "baxter_tools","enable_robot.py","-e"])
        except subprocess.CalledProcessError:
            time.sleep(3)
            _ = subprocess.check_output(["rosrun", "baxter_tools","enable_robot.py","-e"])

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

            modelWeightSave, memorySave = self.rl.tempSave() #In case something fail, saving network state and memory
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

                    if done==2 : #This is a special case, means : redo this trial, something went wrong
                        print "Exit code 2 : Re-doing this trial" 
                        #reloading the network and memory, saved before this experiments
                        self.rl.loadWeightAndMemory(modelWeightSave, memorySave)
                    elif done==1 : #exit code is 1 everything is fine, the trial is done
                        countEp += 1
                        logScores.append(totalReward*3-t)
                        print "======= Trial {} Over, score : {} =========".format(countEp, logScores[-1])

                        if type(self.rl.logState[-1]) is float:
                            print "logState 5 last state",self.rl.logState[-5:]
                        print "logAction 5 last action",self.rl.logAction[-5:]
                        self.rl.logState = []
                    else:
                        raise const.DrunkProgrammer("Exit code 'done' can be 0,1,2 not {}".format(done))

                    break

            if done==2:
                #The trial went wrong, don't save or anything, redo it 
                continue
            else:
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
        self.imageSub = rospy.Subscriber("cameras/head_camera_2/image",Image,self.imageFromCamera)
        self.objects = {}

        self.buttonPosDefault = const.DEFAULT_BUTTON_POS

        self.leftArm = Limb('left')
        self.posSub = rospy.Subscriber("/robot/limb/left/endpoint_state",EndpointState, self.gripperPosFromSub)

        if const.TASK==3:
            self.neutralPosition = np.array([0.50,0.20,0])
        else:
            self.neutralPosition = np.array([0.64,0.20,0.10])

        self.beginningPosition = self.leftArm.joint_angles()
        self.ee_orientation = baxter_utils.get_ee_orientation(self.leftArm)
        
        self.limit = const.LIMIT #Limit of step before reseting

        if const.TASK==4:
            self.actions = [np.array(i) for i in product( (0.05,-0.05,0), repeat=3) ]
            self.actions.pop()
        elif const.TASK==3:
            self.actions = [
                np.array([0.05,0,0]),
                np.array([-0.05,0,0]),
                np.array([0,0.05,0]),
                np.array([0,-0.05,0]),
                np.array([0,0,-0.10]),
                np.array([0,0,0.10])
                ]

    def positionIsValid(self):
        validPosition = 0.42 < self.currentPositionFromSub[0] < 0.80
        validPosition = validPosition and -0.2 < self.currentPositionFromSub[1] < 0.76
        return validPosition

    def spawnButton(self):
        buttonPoint = Point(*self.buttonPosDefault)
        #if button already exists, it's from an other session, need to delete.
        self.delButton()
        self.add_object( SuperButton('button1').spawn(buttonPoint))

    def tuckArms(self):

        done = False
        fail = 0

        while not done and fail<5:
            try:
                _ = subprocess.check_output(['rosrun', 'baxter_tools', 'tuck_arms.py', '-u'])
                done = True
            except subprocess.CalledProcessError:
                time.sleep(3)
                fail += 1
                print "Warning tucking arm failed, catching error and continue, if you see this too often, check ros"
                
            
    def delButton(self):
        _ = subprocess.check_output(["rosservice","call","gazebo/delete_model","model_name: 'button1'"])
        time.sleep(const.ACTION_TIME)
            
    def gripperPosFromSub(self, message):
        self.currentPositionFromSub = (message.pose.position.x,message.pose.position.y,message.pose.position.z)
        
    def add_object(self, obj):
        self.objects[obj.gazebo_name] = obj

    def del_objects(self):
        for obj in self.objects.keys(): self.objects[obj].delete()

    def reset(self):

        self.tuckArms()
        self.spawnButton()

        joints = None
        count = 0
        while not joints:
            try:
                joints = baxter_utils.IK(self.leftArm, self.neutralPosition, self.ee_orientation)
            except rospy.ServiceException:
                count += 1
                if count >10:
                    raise BaxterProblem("Problem in IK : Can't reach neutral position")
                continue
            
        self.leftArm.move_to_joint_positions(joints)
        time.sleep(const.RESET_TIME)
        self.rl.timNet.state = self.currentPositionFromSub
        return joints

    def move(self, coordinate, relative=True):
        """
        Coordinate is a array of size 3 (x,y,z)
        Relative can be either :
        - True : Move according to the position of the hand
          Ex : pos = 1,0,0 coord=0.5,0,0 => PosFinal = 1.5,0,0
        - False : Move to the coordinate given by the array
          Ex : pos = 1,0,0 coord=0.5,0,0 => PosFinal = 0.5,0,0
        """
        
        if relative:
            coordinate = np.array(coordinate)
            position = self.currentPositionFromSub+coordinate
        else:
            position = coordinate
            
        joints = None
        trial = 0
        while not joints:
            if trial == 2:
                time.sleep(const.RESET_TIME)
            if trial > 3:
                return False
            try:
                joints = baxter_utils.IK(self.leftArm, position, self.ee_orientation)
            except rospy.ServiceException :
                trial += 1

        self.leftArm.move_to_joint_positions(joints,timeout=const.ACTION_TIME)
        self.rl.timNet.state = self.currentPositionFromSub #Needed for dummy timNet, other timNet doesn't use this value
        return True
        
    def step(self,actionId):
        """
        Input : actionId (int)
        Return reward, done

        done is a code :
        - 0 (or False) : Not Done Yet
        - 1 (or True) : Done
        - 2 : Exit with a special case, need to redo experience (Button moved during experience for exemple)
        """

        if type(const.NO_BRAIN) is int:
            #actionId = int(raw_input("Todo (25 down) : "))
            actionId = const.NO_BRAIN

        lastPosition = np.array(self.currentPositionFromSub)
        #Apply action 
        movementVector = self.actions[actionId] #smarter way to select action, instead of doing if elif elif ...

        # Check if action could be done (Sometimes it fails because the point the robot wants to reach is too far)
        canMove = self.move(movementVector) #<=== Apply action here

        #Get reward or not, restarting experiment ??
        if not canMove: 
            print "Something is wrong with IK, restarting expe"
            reward = 0
            done = 2
            return reward, done

        try:
            button_moved = not np.allclose(self.objects['button1'].positionFromSub, self.buttonPosDefault[:2], rtol=0, atol=2e-2) #Check if button moved during step or not. You don't take z into account, because since the button is spawn above the table, the spawn z IS different from the actual position
        except TypeError,e:
            #The problem is that the button subscriber is still not working
            #So we use the default position of button
            button_moved = False

        #print "actual pos",self.currentPositionFromSub
        if self.objects['button1'].is_pressed():
            reward = 20
            done = 1
        elif button_moved:
            reward = 0
            done = 2
        elif not self.positionIsValid():
            print "Out of Bound : GAME OVER" 
            reward = -20
            done = 1
        else:
            if np.allclose(lastPosition+movementVector, self.currentPositionFromSub,rtol=0, atol=1e-2): #Something blocking the way ?
                reward = 0
                done = 0
            else:
                print "Boing, wall, should not be here"
                self.move(lastPosition, relative=False)
                time.sleep(0.5)
                #Waiting because sometimes the arm is still stuck on the table

                if const.TASK==3:
                    reward = 0
                    done=0
                else:
                    reward = -5
                    done=0

        return reward,done
