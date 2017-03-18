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



def talker():
	'''Initialize ros publisher, ros subscriber'''
	# topic where we publish
	image_pub = rospy.Publisher("images",CompressedImage, queue_size=1)
	rospy.init_node('image_feature', anonymous=True)
	helper(image_pub);



def helper(ic):
    '''Initializes and cleanup ros node'''
    try:
        #####  READ IMAGE FROM DATA FILE TO SEND TO SUBSCRIBER 
        filename = 'src/beginner_tutorials/scripts/data/test_custom/both4.jpg'
        image_np = cv2.resize(cv2.imread(filename.strip()), (400, 400), interpolation=cv2.INTER_AREA)     


        #### FORMATTING IMAGE INTO MESSAGE TO PUBLISH 
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()

        # Publish new image
        ic.publish(msg)
        rospy.spin()
            
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
   	    talker()
    except rospy.ROSInterruptException:
    	pass
