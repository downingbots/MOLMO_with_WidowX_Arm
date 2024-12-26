import cv2
import do_skills
import widowx_client
import widowx_calibrate
import widowx_manual_control
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
import sys
import camera_snapshot
from widowx import WidowX
import widowx_client
import widowx_image
import widowx_calibrate
import analyze_tabletop
import copy
from util_cv_analysis import *
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from util_radians import *

class perfection():
  def __init__(self):
      self.wi = widowx_image.widowx_image()
      self.wdw = widowx_client.widowx_client()
      # define first so that widowx_calibrate() will not define the wdw_client
      # or robot_camera.
      # self.calib can only use post-calibration functions.
      self.calib = widowx_calibrate.widowx_calibrate()
      self.cfg = self.calib.read_config()
      self.wmc = widowx_manual_control.widowx_manual_control(self.wdw, self.wi.robot_camera, self.calib)
      self.cvu = CVAnalysisTools()
 
      # Begin Game-of-Perfection Defines
      self.HIGH_ABOVE_PIECE_HEIGHT_ON_TT = 10.2
      self.LOW_ABOVE_PIECE_HEIGHT_ON_TT = 5.5
      self.ABOVE_HOLE_HEIGHT_ON_TT = 4.5
      self.PIECE_PEG_HEIGHT_ON_TT = 2.8   # or 3.3?
      self.PIECE_PEG_HEIGHT_IN_HOLE = 1.1 # reached...
      self.PIECE_HEIGHT = 0.4 # not including peg
      self.PICK_TO_PLACE_HEIGHT = 4.5
      self.MIDDLE_SQUARE_LOC = [23,0]
      self.NUM_SQUARE = [5,5]
      self.BOARD_SIZE = [16.5, 16.5] # 6.5 inches = 16.5 cm
      self.ARM_UP_Z   = None
      self.GRIPPER_POINTED_DOWN = (-math.pi/2)
      # Shapes of pieces
      self.shapes = [ # L to R, Closest to Farthest
                     ["Kite","Pentagon","Tub","X","Equalateral Triangle"],
                     ["Pizza Slice", "Trapezoid", "6-Pointed Star", "Rhombus", "Octagon"],
                     ["Parallelogram", "Arch", "Circle", "Cross", "S"],
                     ["5-Pointed Star", "Hot Dog", "Right Triangle", "Y", "Rectangle"],
                     ["Semicircle", "Square", "Inverted S", "Hexagon", "Astrisk"]
                    ]
      self.skills = do_skills.skills(self.calib, self.wdw)


  # use larger image
  # reduce it to calibrated size 
  # is the circle on the center of the gripper destination?
  # reduce it to pick size
  # align gripper so that camera can see

  def shape_to_board_loc(self, shape):
      square = [None,None]
      for i in range(self.NUM_SQUARE[0]):
        for j in range(self.NUM_SQUARE[1]):
          if shape == self.shapes[i][j]:
            square = [i+1,j+1]
            break
      sq_size = self.BOARD_SIZE[0] / self.NUM_SQUARE[0]
      if (self.NUM_SQUARE[0] % 2 == 1 and self.NUM_SQUARE[1] % 2 == 1):
        middle_sq = [(self.NUM_SQUARE[0] + 1)//2, (self.NUM_SQUARE[1] + 1)//2]
      else:
        # oops, being too lazy to generalize
        print("Board size was assumed to have odd number of rows/columns")
 
      print("square:", square)
      print("sq size:", sq_size)
      print("middle sq:", middle_sq)
      print("MIDDLE SQ:", self.MIDDLE_SQUARE_LOC)

      loc = [(square[0] - middle_sq[0]) * sq_size + self.MIDDLE_SQUARE_LOC[0],
             (middle_sq[1] - square[1]) * sq_size + self.MIDDLE_SQUARE_LOC[1]]
      print("board loc:", loc)
      return loc


  def choose_shape(self):
      print("Shapes:")
      print(" ")
      for i in range(self.NUM_SQUARE[0]):
        for j in range(self.NUM_SQUARE[1]):
          num = i * self.NUM_SQUARE[0] + j + 1
          print(str(num) + ": " + self.shapes[i][j])
      print(" ")
      while True:
        print("Enter shape number:")
        shape_num = input()
        try:
          shape_num = int(shape_num)
        except:
          continue
        if (shape_num > 0 and shape_num <= self.NUM_SQUARE[0] * self.NUM_SQUARE[1]):
          shape_i = shape_num // self.NUM_SQUARE[0]
          shape_j = (shape_num-1) % self.NUM_SQUARE[0] 
          shape = self.shapes[shape_i][shape_j]
          break
      return shape, [shape_i, shape_j]

  # def rotate_piece_into_hole(self):
  def rotate_wrist_until_drop(self, place_z, clockwise=True):
      # starts from facing camera
      DELTA_ACTION = .1
      vr = self.wdw.getServoAngle(self.wdw.IDX_ROT)
      State = self.wdw.state()
      while State['Z'] > place_z + self.PIECE_HEIGHT:
        if clockwise:
          rot += DELTA_ACTION
        else:
          rot -= DELTA_ACTION
        if abs(rot) > rotLim:
          return False
        self.wdw.action(vr=rot)
        State = self.wdw.state()
      # went into slot
      YN = self.confirm_action("is piece in the board hole?", True)
      if YN == 1:
        self.wdw.gripper(self.wdw.GRIPPER_OPEN)
        return True
      YN = self.confirm_action("is piece in the board hole?", False)
      if YN == 1:
        return True
      elif YN == 2:
        return False

  def confirm_action(self, str_quest, do_man_act=True):
      YN = 0
      while True:
        big_img, big_img_file = self.wi.get_big_image()
        cv2.imshow('TableTop', big_img) 
        q = str_quest + " 1 = Yes, 2 = No"
        print(q)
        YN = input()
        # print("YN:", YN)
        try:
          if int(YN) not in [1,2]:
            continue
        except:
          continue
        break
      if do_man_act and int(YN) == 2:
        self.wmc.do_manual_action(True)
      return YN

  # TODO: move to skills?
  # self.skills.interactive_pick(cpx, cpy, above_z, grasp_z)
  def interactive_pick(self, px, py, above_z, grasp_z):
      cpx, cpy = self.wi.transform_big_img_loc_to_calib_gripper_loc(px, py)
      # pick shape from side height
      gx,gy,ugz = self.calib.pixel_to_gripper_loc(cpx, cpy, up_down="UP")

      self.wdw.set_move_mode('Absolute')
      self.wdw.action(vx = gx, vy = gy, vz= ugz, vg=self.GRIPPER_POINTED_DOWN)
      while True:
        sw_angle = np.arctan2(gy, gx)
        rot = self.calib.compute_rot(gx, gy, sw_angle)
        succ = self.wdw.action(vr=rot)
        self.wdw.action(vz= above_z)
        self.confirm_action("is gripper centered directly above piece\'s peg?")

        self.wdw.action(vz= grasp_z)
        self.confirm_action("is gripper ready to grab piece\'s peg?")
  
        self.wdw.gripper(self.wdw.GRIPPER_CLOSED)
        self.wdw.action(vz= above_z)

        if self.wdw.is_grasping:
          print("Grasping succeeded.")
          break
        else:
          YN = self.confirm_action("Did gripper grab piece\'s peg?", False)
          if YN == 1:
            break
          elif YN == 2:
            self.wdw.gripper(self.wdw.GRIPPER_OPEN)
          
  def interactive_place(self, gx, gy, above_z, place_z):
      # cpx, cpy = self.wi.transform_big_img_loc_to_calib_gripper_loc(px, py)
      # pick shape from side height
      # gx,gy,ugz = self.calib.pixel_to_gripper_loc(cpx, cpy, up_down="DOWN")

      self.wdw.set_move_mode('Absolute')
      self.wdw.action(vx = gx, vy = gy, vz= above_z, vg=self.GRIPPER_POINTED_DOWN)
      sw_angle = np.arctan2(gy, gx)
      rot = self.calib.compute_rot(gx, gy, sw_angle)
      succ = self.wdw.action(vr=rot)
      self.confirm_action("is gripper centered directly above piece\'s hole on the board?")

      self.wdw.set_move_mode('Absolute')
      self.wdw.action(vz= place_z)
      while True:
        YN = self.confirm_action("is gripper ready to rotate clockwise?", True)
        if YN == 1:
          succ = self.rotate_wrist_until_drop(clockwise=True)
          if succ:
            break
        succ = self.wdw.action(vr=rot)
        YN = self.confirm_action("is gripper ready to rotate counterclockwise?", False)
        if YN == 1:
          succ = self.rotate_wrist_until_drop(clockwise=False)
          if succ:
            break
        succ = self.wdw.action(vr=rot)
        print("place piece in the appropriate hole.")
        self.wmc.do_manual_action(True)
        break
      self.wdw.gripper(self.wdw.GRIPPER_OPEN)
      self.wdw.action(vz= above_z)


  def game_flow(self):
      # out of camera
      self.wdw.moveOutCamera()
      # click on object / press return
      # displaying the image 
      big_img, big_img_file = self.wi.get_big_image()
      print("big img shape:", big_img.shape)
      shape, [shape_i, shape_j] = self.choose_shape()

      q = "click on location of the " + shape + "\'s peg to pick up"
      # print("click on location of the piece's peg to pick up")
      print(q)

      self.wmc.set_click_image('TableTop', big_img)
      cv2.setMouseCallback('TableTop', self.wmc.click_event)
      cv2.waitKey(0)

      self.interactive_pick(self.wmc.click_row, self.wmc.click_col, 
                above_z = self.LOW_ABOVE_PIECE_HEIGHT_ON_TT,
                grasp_z = self.PIECE_PEG_HEIGHT_ON_TT)

      board_x,board_y = self.shape_to_board_loc(shape)
      print(board_x, board_y)
      self.interactive_place(board_x, board_y,
                above_z = self.LOW_ABOVE_PIECE_HEIGHT_ON_TT,
                place_z = self.PIECE_PEG_HEIGHT_IN_HOLE)
      self.wdw.set_move_mode('Absolute')
      self.wdw.action(vz= self.LOW_ABOVE_PIECE_HEIGHT_ON_TT)

def main(params):
    game = perfection()
    while True:
      game.game_flow()

if __name__ == "__main__":
    # only pass parameters to main()
    main(sys.argv[1:])

