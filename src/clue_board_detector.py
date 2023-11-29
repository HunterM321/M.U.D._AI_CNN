#! /usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import clue_board_CV as cb

## A class that subscribes to the image stream and publishes to the velocity stream
class clue_board_detector:

  def __init__(self):

    self.bridge = CvBridge()
    # Subscribes to the camera topic
    self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
    # Publishes to the clue board debugging topic
    self.clue_board_debug = rospy.Publisher('/R1/pi_camera/clue_board_debug', Image, queue_size=5)

    # self.rate = rospy.Rate(2)


  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      clue_board = cb.identify_clue(cv_image)
      # print(clue_board)
      # Convert the processed image back to a ROS image message
      ros_image = self.bridge.cv2_to_imgmsg(clue_board, "bgr8")
      self.clue_board_debug.publish(ros_image)

    except CvBridgeError as e:
      print(e)

def main():
  rospy.init_node('clue_board_detector', anonymous=True)
  detector = clue_board_detector()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main()