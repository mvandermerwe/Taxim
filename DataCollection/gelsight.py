import ros_numpy
import rospy
from datetime import datetime
import csv
import rosbag
import os
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from sensor_msgs.msg import Image


def getCurrentTimeStamp():
    now = datetime.now()
    timestamp = time.mktime(now.timetuple())
    return int(timestamp)


'''
Sensor: GelSight Tactile Sensor
Data: Tactile Images
Format: .jpg
'''


class GelSight:
    def __init__(self, object_dir):
        self.object_dir = object_dir
        self.bridge = CvBridge()

        self.gelsight_count = 0
        self.gel_path = object_dir

        # First check the usb channel for the input video using 'ls -lrth /dev/video*'
        # Make the according changes in the usb_cam-test.launch for group: camera1 -> video_device/value ="/dev/video1"
        # Launch the file: roslaunch usb_cam usb_cam-test.launch

        print("reading in the gelsight images...")
        gelsight_topic = '/gelsight_mini/image_color_28N0_295H'
        self.gel_sub = rospy.Subscriber(gelsight_topic, Image, self.gelSightCallback)

    def gelSightCallback(self, img):
        self.img = ros_numpy.numpify(img)

    def stopRecord(self):
        self.gel_sub.unregister()

    def __str__(self):
        return 'GelSight'
