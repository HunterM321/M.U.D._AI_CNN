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
import tensorflow as tf
import Levenshtein

## A class that subscribes to the image stream and publishes to the velocity stream
class clue_board_detector:

  def __init__(self):

    self.bridge = CvBridge()
    self.model = None

    try:
        self.model = tf.keras.models.load_model('text_recognition.h5')
    except Exception as e:
        rospy.logerr("Failed to load model: %s", str(e))
        return  # Optionally return or handle the error as appropriate

    if self.model is None:
        rospy.logerr("Model could not be loaded")
        return  # Optionally return or handle the error as appropriate

    # Subscribes to the camera topic
    self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
    # Publishes to the clue board debugging topic
    self.clue_board_debug = rospy.Publisher('/R1/pi_camera/clue_board_debug', Image, queue_size=5)
    # Publishes to the score tracker topic
    self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size=5)

    self.key2id = {
       'SIZE':    '1',
       'VICTIM':  '2',
       'CRIME':   '3',
       'TIME':    '4',
       'PLACE':   '5',
       'MOTIVE':  '6',
       'WEAPON':  '7',
       'BANDIT':  '8'
    }

    # Previous clue type
    self.prev_clue_type = None
    # List of clues associated with the current clue type
    self.list_curr_clue = []
    # History of clue types published
    self.published_clue_type_history = []
    # History of last clue board
    self.last_clue_history = []
    self.last_clue_count = 0

    # self.rate = rospy.Rate(2)

  def find_closest_key(self, predicted_key):
    """Find the closest key in self.key2id to the predicted_key."""
    closest_key = None
    min_distance = float('inf')

    for key in self.key2id.keys():
        distance = Levenshtein.distance(predicted_key, key)
        if distance < min_distance:
            min_distance = distance
            closest_key = key

    return closest_key, min_distance
  
  def find_clue_with_max_freq(self):
     return max(set(self.list_curr_clue), key=self.list_curr_clue.count)
  
  def find_clue_with_max_freq_last(self):
     return max(set(self.last_clue_history), key=self.last_clue_history.count)
  
  # Get rid of the randomly generated character plus space in between if clue has more than 12 characters
  def purify_clue(self, clue):
     if len(clue) > 12:
        return clue
     else:
        return clue[2:]

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      clue_board, predicted_key, predicted_val = cb.identify_clue(cv_image, self.model)
      if predicted_key != None and predicted_val != None:
        predicted_key = predicted_key.replace(' ', '')
        # print(predicted_key + predicted_val)
        # Check if predicted_key is in self.key2id, if not, find the closest key
        if predicted_key not in self.key2id:
            corrected_key, min_distance = self.find_closest_key(predicted_key)
            # If predicted key is too far off don't do anything
            if min_distance >= 3:
               return
            # rospy.loginfo(f"Corrected key: {corrected_key} from predicted key: {predicted_key}")
            predicted_key = corrected_key
        # If the robot is at the last clue board, collect some images and publish right away
        if predicted_key == 'BANDIT':
            self.last_clue_count += 1
            # we want to first collect 10 images before we publish anything
            if self.last_clue_count < 10:
               self.last_clue_history.append(predicted_val)
            # Once we have enough images, we are confident enough to publish
            elif self.last_clue_count == 10:
               clue = self.find_clue_with_max_freq_last()
               location_id = self.key2id[predicted_key]
               msg2pub = str('MUD_AI,' + '12345,' + location_id + ',' + clue)
               self.score_tracker.publish(msg2pub)
               return
        # If prev_clue_type is the same as current clue type, then we keep on collecting clues
        # If this is the very first clue we see then we also simply append
        if self.prev_clue_type == predicted_key or self.prev_clue_type == None:
           self.list_curr_clue.append(predicted_val)
        # Otherwise we know we have moved on and we start to analyze the next clue
        else:
           # We also publish the old clue here
           clue = self.find_clue_with_max_freq()
           print('Total of ' + str(len(self.prev_clue_type)) + ' used')
          #  print(self.prev_clue_type + ', ' + clue + ' published!')
           location_id = self.key2id[self.prev_clue_type]
          #  actual_clue = self.purify_clue(clue)
           msg2pub = str('MUD_AI,' + '12345,' + location_id + ',' + clue)
           # Only publish if we haven't published this clue before
           if self.prev_clue_type not in self.published_clue_type_history:
              print(msg2pub)
              self.score_tracker.publish(msg2pub)
           self.published_clue_type_history.append(self.prev_clue_type)
           self.list_curr_clue = []
           self.list_curr_clue.append(predicted_val)
        # Need to update the clue type in every iteration
        self.prev_clue_type = predicted_key
        # Convert key to ID
        self.id = self.key2id[predicted_key]
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