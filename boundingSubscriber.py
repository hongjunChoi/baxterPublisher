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
	rospy.init_node('listener', anonymous=True)
    box = Box()
	subscriber = rospy.Subscriber("box", box, callback,  queue_size = 1)
	rospy.spin()

def callback(data):
    rospy.login(rospy.get_caller_id() + "I heard %s", data.data)


if __name__ == '__main__':
    listener()

#our subscriber takes an image and outbounds 4 bounding boxes, once that's done we want subscriber to publish the 4 coordinates