#! /usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time  # Import time module for timestamp

## A class that subscribes to the image stream and publishes to the velocity stream
class picture:

  def __init__(self):
    self.counter = 1  # Initialize counter
    self.base_img_out_path = '/home/fizzer/'  # Base path for saving images

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)

  def callback(self, data):
    self.image_sub.unregister()  # Unsubscribe after receiving the first image
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        rows, cols, channels = cv_image.shape
        print('# rows', rows)
        print('# cols', cols)

        # Create a unique filename for each image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        img_out_path = self.base_img_out_path + 'test_{}_{}.png'.format(self.counter, timestamp)

        # Write the image into the video
        cv2.imwrite(img_out_path, cv_image)

        # Increment the counter
        self.counter += 1

    except CvBridgeError as e:
        print(e)

def main():
  rospy.init_node('picture', anonymous=True)
  cam = picture()
  rospy.spin()

if __name__ == '__main__':
  main()
