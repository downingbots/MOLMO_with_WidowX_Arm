from IPython import display
from PIL import Image
import skvideo
import cv2
# from cv2 import VideoCapture
# from cv2 import waitKey
import numpy as np
import math
import time
from datetime import datetime
import json
import os
import camera_snapshot
from widowx import WidowX
import widowx_client
import widowx_manual_control
import analyze_tabletop
import analyze_lines
import copy
from util_cv_analysis import *
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from util_radians import *

class widowx_image():
  def __init__(self):

      self.cfg = self.read_config()
      # initialize
      try:
        self.wdw = widowx_client.widowx_client()
        self.robot_camera = camera_snapshot.CameraSnapshot("widowx_image_tt_config.json")
        time.sleep(2)
        # first image isn't properly initialized. Do twice.
        self.latest_image, self.latest_image_file, self.latest_image_time = self.robot_camera.snapshot(True)
        self.latest_image, self.latest_image_file, self.latest_image_time = self.robot_camera.snapshot(True)
      except:
        print("WARNING: CAMERA and ROBOT not initialized!")

      # get 1920 x 1080 image of empty tabletap
      self.get_fixed_pos(self.robot_camera)
        
  #########################################################################
  #  The following code (plus widowx manual control) was used to gather the 
  #  calibration information and images in __init__().  Now replaced by a
  #  more automated approach.  Not used. 
  def get_fixed_pos(self, robot_camera):
      # self.wdw.moveRest()
      # self.wdw.set_move_mode('Absolute')
      # im, im_file, im_time = robot_camera.snapshot(True)
      # robot_image = Image.fromarray(np.array(im))
      # latest_image = self.calib_dir + "/" + "image_tabletop.jpg"
      # latest_image = "/tmp" + "/" + "image_tabletop.png"
      # robot_image.save(latest_image)
      # display.Image(im)
      # print("Empty desk with Resting ARm. Press return to continue.")
      # wait_for_return = input()

      # close gripper
      # get observation
      # get gripper pixel based on cv analysis
      
      self.wdw.moveOutCamera()
      im, im_file, im_time = robot_camera.snapshot(True)
      display.Image(im)
      # robot_image = Image.fromarray(np.array(im))
      # cv2.imwrite(im, image_filenm)
      print("Empty desk. Press return to continue.")
      wait_for_return = input()
      im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
      im = Image.fromarray(np.array(im))
      imfilenm = "/tmp/image_tt.png"
      im.save(imfilenm)

      print("image at:", imfilenm)
      return
      
      self.wdw.gripper(self.wdw.GRIPPER_OPEN)
      gripper_pos_fully_open = self.wdw.gripper_pos_open
      self.wdw.moveArmPick()
      self.wdw.wrist_rotate(angle = self.wdw.GRIPPER_ROT_TO_CAMERA)
      im, im_file, im_time = robot_camera.snapshot(True)
      robot_image = Image.fromarray(np.array(im))
      latest_image = "/tmp/image_empty_pick_go.jpg"
      robot_image.save(latest_image)
      display.Image(im)
      print("Empty desk with arm in PICK_POS", gripper_pos_fully_open)
      wait_for_return = input()
      self.wdw.gripper(self.wdw.GRIPPER_CLOSED)
      gripper_pos_fully_closed = self.wdw.gripper_pos_closed
      im, im_file, im_time = robot_camera.snapshot(True)
      robot_image = Image.fromarray(np.array(im))
      latest_image = "/tmp/image_empty_pick_gc.jpg"
      robot_image.save(latest_image)
      display.Image(im)
      print("Empty desk with PICK_POS and closed gripper", gripper_pos_fully_closed)
      wait_for_return = input()
      
      self.wdw.moveOutCamera()
      print("Take picture of objects on desk by pressing return")
      wait_for_return = input()
      # first snapshot isn't properly tuned; take snapshot & throw away.
      im, im_file, im_time = robot_camera.snapshot(True)
      robot_image = Image.fromarray(np.array(im))
      latest_image = "/tmp/image_objects.jpg"
      robot_image.save(latest_image)
      display.Image(im)

  #########################################################################
  def resize(self,image):
      image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
      image = tf.cast(image, tf.uint8)
      return image
  
#########################################################################

  def read_config(self):
      with open('widowx_image_tt_config.json') as config_file:
        config_json = config_file.read()
      self.config = json.loads(config_json)
      return self.config

  def get_initial_state(self):
      init_state = json.loads(self.config["initial_state"])
      print("init_state", init_state)
      return init_state

###################################################3

wi = widowx_image()
