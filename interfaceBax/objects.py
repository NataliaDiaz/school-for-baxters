#!/usr/bin/python
#coding: utf-8

from __future__ import division

import arm_scenario_simulator as arm_sim #Needed to spawn button
import rospy

from gazebo_msgs.msg import ModelStates
import time


class SuperButton(arm_sim.Button):
    def __init__(self, name):
        arm_sim.Button.__init__(self, name)
        self.positionFromSub = None
        self.posSub = rospy.Subscriber("/gazebo/model_states",ModelStates,self.posButtonFromSub)

    def posButtonFromSub(self, message):
        buttonId = message.name.index('button1')
        button1 = message.pose[buttonId].position
        self.positionFromSub = [button1.x,button1.y]
