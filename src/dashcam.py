#! /usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

## A class that subscribes to the image stream and publishes to the velocity stream
class dashcam:

  def __init__(self):
    vid_out_path = '/home/fizzer/dashcam_full.avi'
    self.vid = cv2.VideoWriter_fourcc(*'MJPG')
    self.out = cv2.VideoWriter(vid_out_path, self.vid, 10.0, (1280, 720))

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,self.callback)

    # self.rate = rospy.Rate(2)


  def callback(self,data):
    try:
      
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

      rows,cols,channels = cv_image.shape
      print('# rows', rows)
      print('# cols', cols)

      # Write the image into the video
      self.out.write(cv_image)
      print('Writing frame...\n')

    except CvBridgeError as e:
      print(e)
  
  def release(self):
    self.out.release()

def main():
  rospy.init_node('dashcam', anonymous=True)
  cam = dashcam()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    cam.release()
    print("Shutting down")

if __name__ == '__main__':
    main()