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
      # initialize
      self.config_file = 'wdwx_config.json'
      self.config = self.read_config(self.config_file)
      # possible problem with trying to open same cameras twice.
      # open widow_image camera first. 
      # widow_calibrate should catch the error 
      # & still be usable for post-calibration transforms...
      self.robot_camera = camera_snapshot.CameraSnapshot(self.config_file)
      try:
        time.sleep(2)
        # first image isn't properly initialized. Do twice.
        self.latest_image, self.latest_image_file, self.latest_image_time = self.robot_camera.snapshot(True)
        self.latest_image, self.latest_image_file, self.latest_image_time = self.robot_camera.snapshot(True)
      except:
        print("WIDOWX_IMAGE WARNING: CAMERA and ROBOT not initialized!")
        
  ###############################
  # just take and return 1920 x 1080 image
  def get_big_image(self):
      im, im_file, im_time = self.robot_camera.snapshot(True)
      return im, im_file

  ####################################################################
  def resize(self,image):
      image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
      image = tf.cast(image, tf.uint8)
      return image
  
  def get_calib_image(self):
      img, im_file = self.get_big_image()
      self.resize(img)
      return img

  def transform_big_img_loc_to_calib_gripper_loc(self, big_img_w, big_img_h) :
      # calib_shape = [320, 256]
      # big_img_shape = [1260, 1024]  # wdw_config.json
      big_img_width = int(self.config["video_width"])
      big_img_height = int(self.config["video_height"])
      calib_img_width = int(self.config["calib_video_width"])
      calib_img_height = int(self.config["calib_video_height"])
      calib_img_w = round(big_img_w * calib_img_width / big_img_width)
      calib_img_h = round(big_img_h * calib_img_height / big_img_height)
      return calib_img_w, calib_img_h
      



#########################################################################

  def read_config(self, config_file):
      with open(config_file) as cfg_file:
        config_json = cfg_file.read()
      self.config = json.loads(config_json)
      return self.config

###################################################3

# wi = widowx_image()
