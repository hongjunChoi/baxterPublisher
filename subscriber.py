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
	subscriber = rospy.Subscriber("images",CompressedImage, callback,  queue_size = 1)
	rospy.spin()


def callback(ros_data):
    '''Callback function of subscribed topic. 
    Here images get converted and features detected'''
    if VERBOSE :
        print 'received image of type: "%s"' % ros_data.format

    #### direct conversion to IMAGE NP ARRAY  ####
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    

    top_results = 2  # number of crops to show for detection
    predicting_class = 1
    predictions = []

    # TODO : tensorflow image detection starts here
    [meanImage, std] = inputProcessor.getNormalizationData("src/beginner_tutorials/scripts/data/custom_train_data.csv")
    baxterClassifier = baxter.BaxterClassifier()
    

    # Start Tensorflow Session
    with baxterClassifier.sess as sess:

        baxterClassifier.saver = tf.train.Saver()
        print("weight file to restore ... : ", baxterClassifier.weights_file)

        baxterClassifier.saver.restore(
            baxterClassifier.sess, "src/beginner_tutorials/scripts/"+baxterClassifier.weights_file)

        cv2.waitKey(1000)
        print("starting session... ")

        # GET IMAGE FROM USER INPUT
        batch = inputProcessor.regionProposal(image_np)

        if batch is None:
            print("something went wrong when getting images crops ... ")
            return 

        original_img = batch[0]
        image_batch = batch[1]
        boundingBoxInfo = batch[2]
        batch_size = len(image_batch)

        # CREATE INPUT IMAGE BATCH
        input_image = np.zeros(
            [len(image_batch), baxterClassifier.img_size, baxterClassifier.img_size, 3])

        for x in range(batch_size):
            input_image[x] = (image_batch[x] - meanImage) / std

        # RUN CASCADING DETECTOR
        print("batch size : ", batch_size)
        print("input tensor size : ", input_image.shape)

        prediction = sess.run(baxterClassifier.logits, feed_dict={
            baxterClassifier.x: input_image,
            baxterClassifier.batch_size: batch_size,
            baxterClassifier.dropout_rate: 1})

        # filter correctly detected crops
        for y in range(batch_size):
            prob = prediction[y][predicting_class]
            boundingBox = boundingBoxInfo[y]
            predictions.append([prob, boundingBox])

        # sort crops by logit values
        predictions.sort(reverse=True)

        for i in range(top_results):
            boundingBoxData = predictions[i]
            print(boundingBoxData)

            x = boundingBoxData[1][0]
            y = boundingBoxData[1][1]
            winW = boundingBoxData[1][2]
            winH = boundingBoxData[1][3]
            sendBox(boundingBoxData[1])

            # if boundingBoxData[0] > threshold:
            cv2.rectangle(original_img, (x, y),
                          (x + winW, y + winH), (0, 255, 0), 2)

        cv2.imshow("Window", original_img)
        cv2.waitKey(1000)
        time.sleep(1)
        cv2.destroyAllWindows()
        print("AFTER DESTROY")
        cv2.waitKey(1)
        print("AFTER 2ND WAIT")

    #self.subscriber.unregister()

def sendBox(data):
	rospy.init_node('talker', anonymous=True)
	boxPub = rospy.Publisher("box", MultiArrayLayout, queue_size = 1)
	boxPub.publish(data)
	rospy.spin()

if __name__ == '__main__':
    listener()

#our subscriber takes an image and outbounds 4 bounding boxes, once that's done we want subscriber to publish the 4 coordinates