#!/usr/bin/env python

# Python libs
import sys, time
import tensorflow as tf
import inputProcessor
import baxterClassifier as baxter
import numpy as np
import os
from scipy.ndimage import filters
import cv2
from std_msgs.msg import String
# Ros libraries
import roslib
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage
# Baxter detector 
import inputProcessor
import baxterClassifier as baxter

VERBOSE=True


def listener():
	# subscribed Topic
    print("here")
    rospy.init_node('listener', anonymous=True)
    print("in it")
    subscriber = rospy.Subscriber("boxtalk", String, callback,  queue_size = 1)
    rospy.spin()

def callback(data):
    print("i heard")
    print(data)

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

#our subscriber takes an image and outbounds 4 bounding boxes, once that's done we want subscriber to publish the 4 coordinates