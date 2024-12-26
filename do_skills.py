# EXECUTION
# find pixel between linear approx of desired object
# line up above pixel
# lower to above object
#    - What is the highest pixel on the object?
#    - is the gripper directly above the object?
# raise up the object
#    - is the object grabbed by the gripper?
#    - validation: gripper width
#    - validation: compare to gripper image before grab around gripper
#
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
import widowx_calibrate
import analyze_tabletop
import copy
from util_cv_analysis import *
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from util_radians import *

class skills():
  def __init__(self, wdw_client, calib = None):
      if calib is not None:
        self.calib = calib
      else:
        self.calib = widowx_calibrate.widowx_calibrate()
      self.wdw = wdw_client
      self.vpl_status = {}
      self.vpl_status["FOUND"]  = False
      time.sleep(2)
      self.cvu = CVAnalysisTools()
      self.ARM_DOWN_Z = None
      self.ARM_UP_Z   = None
      self.GRIPPER_POINTED_DOWN = (-math.pi/2)
      self.click_image = None
      self.click_row   = None
      self.click_col   = None
      self.click_label = None

  def table_top_loc(px, py):
      gx,gy,gz = self.calib.pixel_to_gripper_loc(px, py, up_down="DOWN")
      return [gx, gy, gz]

  def pick(self, px, py, tt_loc=None, ugz=None, dgz=None, do_grasp=True):
      if ugz is None:
        gx,gy,ugz = self.calib.pixel_to_gripper_loc(px, py, up_down="UP")
      else:
        gx,gy,tmp_ugz = self.calib.pixel_to_gripper_loc(px, py, up_down="UP")
      if gx > 30 or gy > 30 or ugz > 20:
        print("Bad pixel to gripper loc:", px, py, "UP", gx,gy,ugz)
        x = 1/0
      print("Good pixel to gripper loc:", px, py, "UP", gx,gy,ugz)
      self.wdw.set_move_mode('Absolute')
      self.wdw.action(vx = gx, vy = gy, vz= ugz, vg=self.GRIPPER_POINTED_DOWN)
      if dgz is None and tt_loc is None:
        gx,gy,dgz = self.calib.pixel_to_gripper_loc(px, py, up_down="DOWN")
      elif dgz is not None and tt_loc is None:
        gx,gy,tmp_ugz = self.calib.pixel_to_gripper_loc(px, py, up_down="DOWN")
      elif tt_loc is not None:
        gx,gy,dgz = self.calib.pixel_to_gripper_loc(px, py, tt_loc=tt_loc)
      sw_angle = math.atan2(py, px)
      self.calib.compute_rot(px,py,sw_angle)
      succ = self.wdw.action(vr=rot)
      self.wdw.gripper(self.wdw.GRIPPER_OPEN)
      self.wdw.action(vz = dgz)
      if do_grasp:
        self.wdw.gripper(self.wdw.GRIPPER_CLOSED)
        if self.wdw.is_grasping:
          print("Grasping succeeded.")
        else:
          print("WARNING: Grasping failed.")
        self.wdw.action(vz=ugz)
      return [gx, gy, ugz]

  def place(self, px, py, tt_loc=None, ugz=None, dgz=None, do_open=True):
      if ugz is None:
        gx,gy,ugz = self.calib.pixel_to_gripper_loc(px, py, up_down="UP")
      else:
        gx,gy,tmp_ugz = self.calib.pixel_to_gripper_loc(px, py, up_down="UP")
      self.wdw.set_move_mode('Absolute')
      self.wdw.action(vz=ugz)
      self.wdw.set_move_mode('Absolute')
      self.wdw.action(vx = gx, vy = gy)
      if dgz is None and tt_loc is None:
        gx,gy,dgz = self.calib.pixel_to_gripper_loc(px, py, up_down="DOWN")
      elif dgz is not None and tt_loc is None:
        gx,gy,tmp_ugz = self.calib.pixel_to_gripper_loc(px, py, up_down="DOWN")
      elif tt_loc is not None:
        gx,gy,gz = self.calib.pixel_to_gripper_loc(px, py, tt_loc=ttloc)
      self.wdw.action(vz = gz)
      # TODO: add gripper  rotation
      if do_open:
        self.wdw.gripper(self.wdw.GRIPPER_OPEN)
        self.wdw.action(vz=ugz)
      return [gx, gy, gz]


  def wrist_rotate(self, angle):
      while angle > math.pi:
        angle -= 2*math.pi
      while angle < -math.pi:
        angle += 2*math.pi
      rotLim = (300/360)/2 * math.pi
      # for pushing, just need either flat side toward the object
      if angle < -rotLim:
        print("Wrist Rot: flipping angle from/to", angle, (angle+math.pi))
        angle = angle + math.pi
      if angle > rotLim:
        print("Wrist Rot: flipping angle from/to", angle, (angle-math.pi))
        angle = angle - math.pi
      self.wdw.action(vr=angle)
      pos = self.wdw.state()
      print("desired wrist rot:", angle)
      print("actual  wrist rot:", pos["Rot"])


  def push(self, start_px, start_py, end_px, end_py, end_up=False):
      self.wdw.set_move_mode('Absolute')
      start_gx,start_gy,ugz = self.calib.pixel_to_gripper_loc(start_px, start_py, up_down="UP")
      self.wdw.moveArmPick()
      self.wdw.set_move_mode('Absolute')
      # self.wdw.action(vz=ugz)
      self.wdw.action(vx = start_gx, vy = start_gy, vz=ugz)
      start_gx,start_gy,dgz = self.calib.pixel_to_gripper_loc(start_px, start_py, up_down="DOWN")
      end_gx,end_gy,dgz = self.calib.pixel_to_gripper_loc(end_px, end_py, up_down="DOWN")
      rot_angle = np.arctan2(start_gx - end_gx, start_gy - end_gy)
      rot_angle -= np.arctan2(start_gx, start_gy)  # adjust for arm angle
      self.wrist_rotate(angle = rot_angle)
      print("wrist angle:", rot_angle)
      self.wdw.gripper(self.wdw.GRIPPER_CLOSED)
      self.wdw.action(vz = dgz)
      print("arm moved down to ",dgz)
      self.wdw.action(vx=end_gx, vy=end_gy, vz=dgz)
      print("arm moved to end pos:", end_gx, end_gy, dgz, rot_angle)
      self.wdw.action(vz=ugz)
      print("arm moved up")
      return [end_px, end_py, ugz]

