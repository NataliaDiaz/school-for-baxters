#!/usr/bin/python
#coding: utf-8
from __future__ import division
import numpy as np
import rosbag
import rospy
from arm_scenario_experiments import Recorder
from baxter_interface import Head, Limb

import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

import scipy.misc
import subprocess
import time
import random

def moveHead(head,direction):
    if direction==1 and head.pan()>1.3:
        direction = -1
    elif direction==-1 and head.pan()<-1.3:
        direction = 1

    panning = 0.07*direction #+ (random.random()/4-0.5/4)
    head.set_pan(head.pan()+panning)
    time.sleep(0.5)
    return direction
    
directory = 'simpleData/pose05_head_pan/'
bagPath = 'bag'
numImages = 70
#============= INIT ==================
#=====================================
bridge = CvBridge()

subprocess.call(["mkdir", directory])
subprocess.call(["mkdir", directory+'Images'])

rospy.init_node('dataCollector')

#recorder = Recorder(bagPath, 'test')
recorder = Recorder(bagPath, 'test', ['/cameras/head_camera_2/image','/robot/joint_states'])

head = Head()

rospy.loginfo('Starting recording right now')
recorder.new_bag('record')
recording = True

direction = -1
head.set_pan(1.31)
time.sleep(2)


#============= COLLECTING DATA =======
#=====================================
step = 0
try:
    while not rospy.is_shutdown() and step < numImages:
        step += 1
        direction = moveHead(head,direction)
        recorder.check_topics()
        recorder.dump_all()

except rospy.ROSInterruptException:
    print "closing Ros process" 
    raise rospy.ROSInterruptException

recorder.close_bag()
#============= Parsing DATA =========
#=====================================
bag = rosbag.Bag('bag/record.bag')
#bag = rosbag.Bag('here/pose2_head_pan.bag')
print bag

lastTopic = None
totImg = 0
txt_file = ''
firstTopic = None

for topic, msg, t in bag.read_messages():

    if not firstTopic:
        print "firstTopic",topic 
        firstTopic = topic
    if lastTopic == topic:
        continue

    if topic=='test/cameras/head_camera_2/image':
        img = bridge.imgmsg_to_cv2(msg,'rgb8')
        scipy.misc.toimage(img, cmin=0, cmax=255).save('{}Images/frame{}.jpg'.format(directory, str(totImg).zfill(4)))
        totImg += 1

    else:
        if txt_file=='':
            txt_file = '# time '+' '.join(msg.name)+'\n'
        txt_file += str(t)[:-8]+' '+' '.join(map(str,msg.position))+'\n'

    lastTopic = topic
    
print "lastTopic",lastTopic     
f = open(directory+'robot_joint_states.txt','w')
f.write(txt_file)
f.close()
bag.close()
