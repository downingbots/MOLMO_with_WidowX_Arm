# CALIBRATION
# Use existing list as movement places
#    - go to next location
#       -- remove "take picture" line
#    - include description
#    - allow fine-tuning w "success" option
# Do close gripper image
# create gripper-image
#    - store image
# find middle of gripper pixel in image
#    - store mapping
# find width of gripper in pixels (for grabbing)
#    - is the object to be picked soft? -> any location ok
#    - thinnest location of pixels of object
#      -- reflects gripper rotation
#    - bottom and top of object in that fixed pixel width

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
import widowx_manual_control
import analyze_tabletop
import analyze_lines
import copy
from util_cv_analysis import *
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from util_radians import *

class widowx_calibrate():
  def __init__(self):

      # Following data is for Bootstrapping the calibration with
      # pre-defined positions of interest.  Not used if calibration 
      # config file exists in calib dir.
      self.calib_data = {
        "metadata": {
          'calibration_directory': '/home/ros/downing_bots_gpt4o/calibration_images', 
          'calibration_file': 'widowx_calibration.json', 
          "tabletop_segmentation_method": "COLOR",
          "num_color_clusters": 5,
          "closed gripper position": None,
          "open gripper position": None,
          "pixel_to_gripper": None,
          "best_vpl": None
        },
        "image_file_name_components": [["image","idx#"],
             ["empty", "object", "arm", "gripper_opening"],
             ["rest", "pick", "rlpick", "mfpick", "flbolt", "fledge", "mledge", "mreach", "mfreach", "mrreach", "swiv"],
             ["go","gc", "dgo", "dgc"]],
        "calibration_info": [
        {'image_name': 'image_empty.jpg', 'gripper_position': {'X': 13.892806544364808, 'Y': -0.010658253367584278, 'Z': 16.8636682665831, 'Gamma': -0.017645086943255, 'Rot': -0.00255913489736, 'Gripper': 1}, 'image_desc':"empty workspace with gripper out of view of camera", 'gripper_pixel':None}
#
# The following is an example predefined location used by the recalibrate() call used to
# calibrate the gripper positions to pixel locations. This has been replaced by
# the recalibrate_by_swivel() call that automatically sweeps through recalibration positions.
#
#         {'image_name': 'image_empty_pick_gc.jpg', 'gripper_position': {'X': 18.672066845833157, 'Y': 0.07162431714362012, 'Z': 12.197099127315601, 'Gamma': -1.568878382563325, 'Rot': -0.8982563489733599, 'Gripper': 2}, 'image_desc':"closed gripper in initial pick position", 'gripper_pixel':None},
#         {'image_name': 'image_empty_pick_go.jpg', 'gripper_position': {'X': 18.672066845833157, 'Y': 0.07162431714362012, 'Z': 12.197099127315601, 'Gamma': -1.568878382563325, 'Rot': -0.8982563489733599, 'Gripper': 1}, 'image_desc':"open gripper in initial pick position", 'gripper_pixel':None},
#         {'image_name': 'image_empty_pick_dgo.jpg', 'gripper_position': {'X': 17.427971688029412, 'Y': 0.04011112417317652, 'Z': 2.200391904578744, 'Gamma': -0.844662640196685, 'Rot': -0.8982563489733599, 'Gripper': 1}, 'image_desc':"automatically lowered closed gripper from the initial pick position", 'gripper_pixel':None},
#         {'image_name': 'image_empty_pick_dgc.jpg', 'gripper_position': {'X': 17.427971688029412, 'Y': 0.04011112417317652, 'Z': 2.200391904578744, 'Gamma': -0.844662640196685, 'Rot': -0.8982563489733599, 'Gripper': 2}, 'image_desc':"automatically lowered gripper from the initial pick position", 'gripper_pixel':None},
        ]
      }
      # print(self.calib_data)
      # print(self.calib_data["image_file_name_components"])
      # print(self.calib_data["image_file_name_components"][2])
      self.vpl_status = {}
      self.vpl_status["FOUND"]  = False
      self.cfg = self.read_config()
      self.calib_dir = self.cfg["calibration_directory"] 
      self.calib_file = self.cfg["calibration_file"] 
      # self.tmp_calib_file =  "/tmp/" + self.calib_file
      self.tmp_calib_file =  self.calib_dir + "/tmp_" + self.cfg["calibration_file"] 
      self.calib_file =  self.calib_dir + "/" + self.calib_file
      # initialize
      self.config = self.read_config()
      try:
        # tolerate failures if already have widowx_client / camera_snapshot
        self.wdw = widowx_client.widowx_client()
        self.robot_camera = camera_snapshot.CameraSnapshot()
        time.sleep(2)
        # first image isn't properly initialized. Do twice.
        self.latest_image, self.latest_image_file, self.latest_image_time = self.robot_camera.snapshot(True)
        self.latest_image, self.latest_image_file, self.latest_image_time = self.robot_camera.snapshot(True)
      except:
        print("WARNING: CAMERA and ROBOT not initialized!")
      self.cvu = CVAnalysisTools()
      self.prev_image = None
      self.prev_image_file = None
      self.prev_pos_name = None
      self.pos_name = None
      self.prev_gripper_pos = None
      self.gripper_pos = None
      self.open_gripper_pos = []
      self.closed_gripper_pos = []
      self.prev_tabletop_state = None
      self.tabletop_state = None
      self.main_range = [None, None] 
      self.ARM_DOWN_Z = None
      self.ARM_UP_Z   = None
      self.TT_WIDTH = self.config["TT_WIDTH"]
      self.GRIPPER_POINTED_DOWN = (-math.pi/2)
      self.armctr2tt = None
      try:
        self.read_calib()
      except:
        self.write_calib()

      self.calib_info = self.calib_data["calibration_info"]
      self.click_image = None
      self.click_row   = None
      self.click_col   = None
      self.click_label = None
      self.lna = analyze_lines.AnalyzeLines()
      self.tt = analyze_tabletop.AnalyzeTableTop()
      try:
        self.img_empty_file = self.calib_dir + "/" + self.get_calib_data(0, "image_name")
      except:
        self.img_empty_file = None
        
  def write_calib(self):
      # Serializing json
      # for debugging:
      json_object = json.dumps(self.calib_data, indent=4)
      # Bridgedata filename:
      # "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
      # filenm = self.config["dataset"] + '/' + "ep" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ".json"
      with open(self.tmp_calib_file, "w") as outfile:
        outfile.write(json_object)
        outfile.flush() # make sure that all data is on disk
        os.fsync(outfile.fileno())    
        os.replace(self.tmp_calib_file, self.calib_file) 

  def read_calib(self):
      with open(self.calib_file) as config_file:
        config_json = config_file.read()
      self.calib_data = json.loads(config_json)
      print("load calib_data:", self.calib_data["metadata"]["swivel_calib_info"])
      self.tt.set_best_vpl(self.calib_data["metadata"]["best_vpl"])
      return config


  #########################################################################
  #  The following code (plus widowx manual control) was used to gather the 
  #  calibration information and images in __init__().  Now replaced by a
  #  more automated approach.  Not used. 
  def get_fixed_pos(self, robot_camera):
      self.wdw.moveRest()
      # self.wdw.set_move_mode('Absolute')
      im, im_file, im_time = robot_camera.snapshot(True)
      robot_image = Image.fromarray(np.array(im))
      robot_images.append(robot_image)
      latest_img = self.calib_dir + "/" + "image_empty_rest.jpg"
      robot_image.save(latest_img)
      display.Image(im)
      print("Empty desk with Resting ARm. Press return to continue.")
      wait_for_return = input()

      # close gripper
      # get observation
      # get gripper pixel based on cv analysis
      
      self.wdw.moveOutCamera()
      im, im_file, im_time = robot_camera.snapshot(True)
      robot_image = Image.fromarray(np.array(im))
      robot_images.append(robot_image)
      # latest_img = "/tmp/image_empty.jpg"
      latest_img = self.calib_dir + "/" + "image_empty.jpg"
      robot_image.save(latest_img)
      display.Image(im)
      print("Empty desk. Press return to continue.")
      wait_for_return = input()
      
      self.wdw.gripper(self.wdw.GRIPPER_OPEN)
      gripper_pos_fully_open = self.wdw.gripper_pos_open
      self.wdw.moveArmPick()
      self.wdw.wrist_rotate(angle = self.wdw.GRIPPER_ROT_TO_CAMERA)
      im, im_file, im_time = robot_camera.snapshot(True)
      robot_image = Image.fromarray(np.array(im))
      robot_images.append(robot_image)
      latest_img = "/tmp/image_empty_pick_go.jpg"
      robot_image.save(latest_img)
      display.Image(im)
      print("Empty desk with arm in PICK_POS", gripper_pos_fully_open)
      wait_for_return = input()
      self.wdw.gripper(self.wdw.GRIPPER_CLOSED)
      gripper_pos_fully_closed = self.wdw.gripper_pos_closed
      im, im_file, im_time = robot_camera.snapshot(True)
      robot_image = Image.fromarray(np.array(im))
      robot_images.append(robot_image)
      latest_img = "/tmp/image_empty_pick_gc.jpg"
      robot_image.save(latest_img)
      display.Image(im)
      print("Empty desk with PICK_POS and closed gripper", gripper_pos_fully_closed)
      wait_for_return = input()
      
      self.wdw.moveOutCamera()
      print("Take picture of objects on desk by pressing return")
      wait_for_return = input()
      # first snapshot isn't properly tuned; take snapshot & throw away.
      im, im_file, im_time = robot_camera.snapshot(True)
      robot_image = Image.fromarray(np.array(im))
      robot_images.append(robot_image)
      latest_img = "/tmp/image_objects.jpg"
      robot_image.save(latest_img)
      display.Image(im)

  #########################################################################
  def as_gif(self, images, rbt=False):
    # Render the images as the gif:
    filenm = self.calib_dir + "/" + "robot_images.gif"
  
    images[0].save(filenm, save_all=True, append_images=images[1:], duration=1000, loop=0)
    gif_bytes = open(filenm,'rb').read()
    return gif_bytes
  
  def resize(self,image):
      image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
      image = tf.cast(image, tf.uint8)
      return image
  
#########################################################################

  def read_config(self):
      with open('widowx_config.json') as config_file:
        config_json = config_file.read()
      self.config = json.loads(config_json)
      return self.config

  def get_initial_state(self):
      init_state = json.loads(self.config["initial_state"])
      print("init_state", init_state)
      return init_state

###################################################3

  # from:
  #  https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
  # function to display the coordinates of 
  # of the points clicked on the image  
  def click_event(self, event, x, y, flags, params): 
      # checking for left or right mouse clicks 
      if (event == cv2.EVENT_LBUTTONDOWN or 
         event==cv2.EVENT_RBUTTONDOWN): 
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # b = self.click_image[y, x, 0] 
        # g = self.click_image[y, x, 1] 
        # r = self.click_image[y, x, 2] 
        center = (int(x),int(y))
        radius = 10
        cv2.circle(self.click_image,center,radius,(255,255,0),2)
        # cv2.putText(self.click_image, "* X:" + str(x) + ", Y:" + str(y),
        #             (x,y), font, 1, 
        #             (255, 255, 0), 2) 
        # cv2.putText(self.click_image, str(b) + ',' +
        #             str(g) + ',' + str(r), 
        #             (x,y), font, 1, 
        #             (255, 255, 0), 2) 
        cv2.imshow(self.click_label,self.click_image)
        self.click_col = x
        self.click_row = y
  
  def gripper_analysis(self, action, img_goc, prev_img_goc):
      print("gripper analysis", action, img_goc, prev_img_goc, self.img_empty_file)
      goc_pix = self.cvu.moved_pixels(self.img_empty_file, img_goc, init=False, add_edges=False)
      prev_goc_pix = self.cvu.moved_pixels(self.img_empty_file, prev_img_goc, init=False, add_edges=False)
      # goc_pix = cv2.cvtColor(prev_goc_pix, cv2.COLOR_RGB2BGR)
      # prev_goc_pix = cv2.cvtColor(prev_goc_pix, cv2.COLOR_RGB2BGR)
      gripper = {}
      gripper["arm_mask"] = goc_pix

      grasping_area = self.cvu.moved_pixels(goc_pix, prev_goc_pix, init=False, add_edges=False)
      gripper["grasping_area"] = grasping_area
      # open_gripper_bb = self.cvu.findBigArea(grasping_area)
      # x1 = (og["x"] - int(og[pixels_horiz] / 2 - 1)
      # x2 = (og["x"] + int(og[pixels_horiz] / 2 + 1)
      # y2 = (og["y"] + 1)
      # open_gripper_bb = [(x1,0), (x2,0), (x1,y2), (x2,y2)]

      if action == "GRIPPER_CLOSE":
        diff = cv2.absdiff(goc_pix, prev_goc_pix)
        grasping_mask = cv2.bitwise_and(diff, prev_goc_pix)
        # grasping_mask = cv2.bitwise_and(diff, goc_pix)
      elif action == "GRIPPER_OPEN":
        diff = cv2.absdiff(prev_goc_pix, goc_pix)
        grasping_mask = cv2.bitwise_and(diff, goc_pix)
        # grasping_mask = cv2.bitwise_and(diff, prev_goc_pix)
      else:
        print("ERROR: gripper analysis received bad action", action)
      gripper["grasping_mask"] = grasping_mask

      cv2.imshow("Arm", goc_pix)
      cv2.imshow("Prev Arm", prev_goc_pix)
      cv2.imshow("Grasping Area", grasping_mask)
      cv2.imshow("Gripper grasping area", grasping_area)
      print(grasping_area)
      gripper["bounding_box"] = self.cvu.findBigArea(grasping_mask)
      gripper["grasping_point"] = None
      found = False
      if gripper["bounding_box"] is None:
        print("ERROR: no grasping point found")
      else:
        [[min_col, min_row],[max_col, max_row]] = gripper["bounding_box"]
        # the "rest" position is the only horizontal grasp
        if self.pos_name == "rest":
          c = int((min_col + max_col)/2)
          for r in range(max_row, min_row, -1):
            if (grasping_mask[r,c]):
              found = True
              gripper["grasping_point"] = [c,r]
              break
        else:
          # vertical grasp
          c = int((min_col + max_col)/2)
          for r in range(min_row, max_row):
            if (grasping_mask[r,c]):
              found = True
              gripper["grasping_point"] = [c,r]
              break
        if not found:
          print("ERROR: no grasping point found")
          # potentially correct rotation to camera; but need to retake snapshot
          # rot = self.compute_rot(gpos["X"], gpos["Y"], sw_angle)
          # succ = self.wdw.action(vr=rot)
          image = cv2.imread(img_goc)
        else:
          print("gripper:", gripper)
          bb = [[[min_col,min_row]], [[min_col, max_row-1]], [[max_col, max_row-1]], [[max_col-1, min_row]]]
    
          g_bb = np.intp(bb)
          print("gbb1", bb)
          image = cv2.imread(img_goc)
          # g_bb2, image_bb2 = self.cvu.get_grasping_bounding_box(grasping_mask, image)
          # print("grasp_bb2:", g_bb2)
          image_bb1 = cv2.drawContours(image.copy(), [g_bb], 0, (0, 255, 0), 2)
          cv2.imshow("Grasping Bounding Box",image_bb1)
          # if g_bb2 is not None:
          #   cv2.imshow("alt bb2 alg",image_bb2)
          cv2.waitKey(0)
      YN = 0
      while True:
        if gripper["grasping_point"] is not None:
          print("do you agree with the bounding box or grasping point? 1 = Yes, 2 = No")
          YN = input()
          # print("YN:", YN)
          try:
            if int(YN) not in [1,2]:
              continue
          except:
            continue
        else:
          YN = 2
        if int(YN) == 1:
          break
        elif int(YN) == 2:
          self.click_row = None
          self.click_col = None
          self.click_image = image.copy()
          self.click_label = 'Grasping Bounding Box'
          cv2.imshow("Grasping Bounding Box",self.click_image)
          print("click on grasping point and press return:")
    
          # setting mouse handler for the image 
          # and calling the click_event() function 
          cv2.setMouseCallback('Grasping Bounding Box', self.click_event) 
          # wait for a key to be pressed to exit 
          cv2.waitKey(0) 
          gripper["bounding_box"] = None
          gripper["grasping_point"] = [self.click_col, self.click_row]
          print("Grasping Pnt:", self.click_col, self.click_row)
          
          if gripper["bounding_box"] is None:
            print("No bounding box found")
            cv2.imshow("Grasping Bounding Box", self.click_image.copy())
          else:
            [[min_col, min_row],[max_col, max_row]] = gripper["bounding_box"]
            g_bb = np.intp(bb)
            image_bb1 = cv2.drawContours(self.click_image.copy(), [g_bb], 0, (0, 255, 0), 2)
            cv2.imshow("Grasping Bounding Box",image_bb1)
            print("Bounding box:", g_bb)

      cv2.destroyAllWindows()
      return gripper

###################################################3
  def find_calib_index(self, image_name):
      for idx, calib_data in enumerate(self.calib_data["calibration_info"]):
        if image_name == self.get_calib_data(idx, "image_name"):
          return idx
      return None

  def find_calib_index_by_name(self, image_name):
      return self.find_calib_index(image_name)

  # X,Y must be same. Rest depends on match_type.
  # match type in [None, "UP", "DOWN", "SAME_Z", "SAME_ALL"]
  def match_position(self, pose1, pose2, match_type=None):
      DELTA_ACTION  = self.wdw.DELTA_ACTION
      DELTA_ANGLE  = self.wdw.DELTA_ANGLE
      VZ_UP = self.config["VZ_UP"]
      # print("mp:", pose1, pose2, match_type)
      if (abs(pose1["X"] - pose2["X"]) <= 2*DELTA_ACTION and
          abs(pose1["Y"] - pose2["Y"]) <= 2*DELTA_ACTION and
          (match_type is None or match_type == "SAME_Z" or
           (match_type == "SAME_ALL" and
            abs(pose1["Gamma"] - pose2["Gamma"]) <= 2*DELTA_ANGLE and
            abs(pose1["Rot"] - pose2["Rot"]) <= 2*DELTA_ANGLE and
            abs(pose1["Gripper"] - pose2["Gripper"]) == 0))):
        if (match_type is None or 
            (match_type == "UP" and pose1["Z"] > VZ_UP and pose2["Z"] > VZ_UP) or
            (match_type == "DOWN" and pose1["Z"] <= VZ_UP and pose2["Z"] <= VZ_UP) or
            (match_type == "SAME_Z" and abs(pose1["Z"] - pose2["Z"]) <= 2*DELTA_ACTION) or
            (match_type == "SAME_ALL" and abs(pose1["Z"] - pose2["Z"]) <= 2*DELTA_ACTION)):
          return True
        else:
          return False
      else:
        return False

  def find_calib_index_by_position(self, data, match_type="SAME_ALL"):
      data_pose = data["gripper_position"]
      for idx, calib_data in enumerate(self.calib_data["calibration_info"]):
        # idx_pose = self.get_calib_data(idx, "gripper_position")
        idx_pose = calib_data["gripper_position"]
        if self.match_position(data_pose, idx_pose, match_type):
          print(idx,"matches position for ",data_pose, idx_pose)
          return idx
      print("no position match for ",data_pose)
      return None

  def set_calib_data_by_position(self, data):
      idx = self.find_calib_index_by_position(data, "SAME_ALL")
      idx = self.add_calib_data(idx, data)
      return idx

  def add_calib_data(self, index, data):
      print("add_calib_data:", index, len(self.calib_data["calibration_info"]))
      if index is not None and len(self.calib_data["calibration_info"]) > index:
        self.calib_data["calibration_info"][index] = copy.deepcopy(data)
      elif index is None or len(self.calib_data["calibration_info"]) == index:
        if index is None:
          index = len(self.calib_data["calibration_info"])
        self.calib_data["calibration_info"].append(copy.deepcopy(data))
      else:
        print("ERROR: add_calib_data", index, 
              len(self.calib_data["calibration_info"]), data)
      return index

  def set_calib_data(self, index, key, value):
      print("index,key,value", index,key,value)
      self.calib_data["calibration_info"][index][key] = copy.deepcopy(value)
      # debugging test to avoid errors
      json_object = json.dumps(self.calib_data, indent=4)

  def get_calib_data(self, index, key):
      return self.calib_data["calibration_info"][index][key]

  def post_move_data(self, index, data=None):
      self.prev_image= Image.fromarray(np.array(self.latest_image))
      self.prev_image_file = self.latest_image_file
      self.latest_image, self.latest_image_file, self.latest_image_time = self.robot_camera.snapshot(True)
      print("pmd: latest_image_file:", self.latest_image_file)
      display.Image(self.latest_image)
      print("pmd: ws:", self.wdw.widowx.state)
      # print("dc:", copy.deepcopy(self.wdw.widowx.state))
      if index is not None:
        self.set_calib_data(index, "gripper_position", self.wdw.widowx.state)
        self.set_calib_data(index, "new_image", self.latest_image_file)
        print("pmd: gp:", self.get_calib_data(index, "gripper_position"))
      elif index is None and data is not None:
        data["gripper_position"] = copy.deepcopy(self.wdw.widowx.state)
        data["new_image"] = copy.deepcopy(self.latest_image_file)
        print("pmd: gp:", data["gripper_position"])

  def openCloseGripper(self, img_data_idx1, img_data_idx2, do_pmd=False):
      pose = self.get_calib_data(img_data_idx1, "gripper_position")
      gripper_state, gripper_pos = self.wdw.gripper(pose["Gripper"])
      open_close_gripper = [None,None]
      if gripper_state and pose["Gripper"] == 1:
        self.open_gripper_pos.append(gripper_pos)
        open_close_gripper[0] = gripper_pos
      elif gripper_state and pose["Gripper"] == 2:
        self.closed_gripper_pos.append(gripper_pos)
        open_close_gripper[1] = gripper_pos
      if do_pmd:
        self.post_move_data(img_data_idx2)  
      # take image and do analysis
      relative_img_nm = self.get_calib_data(img_data_idx2, "image_name")
      full_path_img_nm = self.calib_dir + "/" + relative_img_nm  
      cv2.imwrite(full_path_img_nm, self.latest_image)
      relative_prev_img_nm = self.get_calib_data(img_data_idx1, "image_name")
      print("OpenCloseGripper:", img_data_idx1, img_data_idx2, relative_prev_img_nm, relative_img_nm)

      full_path_img_nm = self.calib_dir + "/" + relative_img_nm  
      full_path_prev_img_nm = self.calib_dir + "/" + relative_prev_img_nm
      calib_action = self.check_name(relative_img_nm, relative_prev_img_nm)
      print("OpenCloseGripper: calib_action, img_nm, prv_img_nm:", calib_action, relative_img_nm, relative_prev_img_nm)
      # calib_action, img_nm: NEW_POSITION idx163_swivB_gc.jpg idxNone_swivA_gc.jpg
      img_nm = relative_img_nm
      prev_img_nm = relative_prev_img_nm
      if (calib_action in ["GRIPPER_OPEN", "GRIPPER_CLOSE"] and 
          (self.pos_name == self.prev_pos_name or
           (self.prev_pos_name == "swivA" and self.pos_name == "swivB"))):
        print("OpenCloseGripper: call gripper analysis(): calib_action, img_nm, prv_img_nm:", calib_action, full_path_img_nm, full_path_prev_img_nm)
        gripper_anal = self.gripper_analysis(calib_action, full_path_img_nm, full_path_prev_img_nm)
        self.set_calib_data(img_data_idx2, "gripper_bounding_box", gripper_anal["bounding_box"])
        self.set_calib_data(img_data_idx2, "gripper_grasping_point", gripper_anal["grasping_point"])
        self.set_calib_data(img_data_idx2, "open_close_gripper_pos", open_close_gripper)
        self.set_calib_data(img_data_idx2, "open_close_gripper_match", relative_prev_img_nm)
        # previous action has same BB/Grasp Pt (just did open/close)
        self.set_calib_data(img_data_idx1, "gripper_bounding_box", gripper_anal["bounding_box"])
        self.set_calib_data(img_data_idx1, "gripper_grasping_point", gripper_anal["grasping_point"])
        self.set_calib_data(img_data_idx1, "open_close_gripper_pos", open_close_gripper)
        self.set_calib_data(img_data_idx1, "open_close_gripper_match", relative_img_nm)
        print("OpenCloseGripper: idx1, data1:  ", img_data_idx1, self.calib_data["calibration_info"][img_data_idx1])
        print("OpenCloseGripper: idx2, data2:  ", img_data_idx2, self.calib_data["calibration_info"][img_data_idx2])
        return gripper_anal 
      print("OpenCloseGripper: skipped gripper analysis()!")
      return None

  def move_position(self, img_data_index):
      self.wdw.set_move_mode('Absolute')
      pose=self.get_calib_data(img_data_index, "gripper_position")
      print(img_data_index, "move position pose:", pose)
      self.wdw.action(vx=pose['X'],vy=pose['Y'], vz=pose['Z'], vg=pose['Gamma'],
                 vr=pose['Rot'], goc=pose['Gripper'], swivel=None)
      self.post_move_data(img_data_index)

  def moveUpDown(self, img_data_index):
      # self.wdw.set_move_mode('Relative')
      self.wdw.set_move_mode('Absolute')
      pose=self.get_calib_data(img_data_index, "gripper_position")
      self.wdw.action(vz=pose['Z'])
      self.post_move_data(img_data_index)

  def img_name_state(self, img_name):
      nm = img_name.split("_")
      return nm[2]

  def check_name(self, img_name, prev_img_name=None):
      if img_name == "image_empty.jpg":
        self.tabletop_state = "empty"
        return "EMPTY"
      nm1 = img_name.split("_")
      nm2 = None
      if prev_img_name is not None and prev_img_name != "image_empty.jpg":
        nm2 = prev_img_name.split("_")
      print("nm1,2: ", nm1, nm2)
      if nm1[0].endswith("None"):
        return "ERROR 8"
      elif nm1[0].startswith("idx"):
        gripper1 = nm1[2].split(".")
        gripper2 = nm2[2].split(".")
        self.pos_name = nm1[1]
        self.prev_pos_name = nm2[1]
        if nm1[1] == "swivA":
          return "NEW_POSITION"
      else:
        if nm2 is not None and (nm1[0] != nm2[0] or nm1[0] != "image"):
          print("Error: unexpected name format ", img_name, prev_img_name)
          return "ERROR 2"
        self.prev_tabletop_state = self.tabletop_state
        self.tabletop_state = nm1[1]
        # pos_names = self.calib_data["calibration_data"]["image_file_name_components"][2]
        pos_names = self.calib_data["image_file_name_components"][2]
        # print(pos_names, self.calib_data["calibration_data"]["image_file_name_components"][2], nm1[2])
        if nm1[2] not in pos_names:
          print(nm1[2],"not in", pos_names)
          return "ERROR 3"
        self.pos_name = nm1[2]
        if nm2 is not None:
          self.prev_pos_name = nm2[2]
        if nm2 is None or len(nm2)==2  or nm1[2] != nm2[2]:
          return "NEW_POSITION"
        gripper1 = nm1[3].split(".")
        gripper2 = nm2[3].split(".")
      self.gripper_pos = gripper1[0]
      self.prev_gripper_pos = gripper2[0]
      if gripper1[0] not in ["dgo", "dgc", "gc", "go"]:
        print("Error: unexpected named gripper position", img_name, gripper1)
        return "ERROR 4"
      if gripper2[0] not in ["dgo", "dgc", "gc", "go"]:
        print("Error: unexpected named gripper position", img_name, gripper2)
        return "ERROR 5"
      if gripper1[1] != "jpg" and gripper1[1] != "png":
        print("Error: unexpected file extension ", img_name)
        return "ERROR 6"
      elif nm2 is not None and (gripper2[1] != "jpg" and gripper2[1] != "png"):
        print("Error: unexpected file extension ", prev_img_name)
        return "ERROR 7"
      print("MD:", gripper1[0],gripper2[0])
      if (gripper1[0] == "dgc" and gripper2[0] == "gc"):
        return "MOVE_DOWN"
      elif (gripper1[0] == "dgo" and gripper2[0] == "go"):
        return "MOVE_DOWN"
      elif (gripper1[0] == "go" and gripper2[0] == "dgo"):
        return "MOVE_UP"
      elif (gripper1[0] == "gc" and gripper2[0] == "dgc"):
        return "MOVE_UP"
      elif (gripper1[0] == "gc" and gripper2[0] == "go"):
        return "GRIPPER_CLOSE"
      elif (gripper1[0] == "go" and gripper2[0] == "gc"):
        return "GRIPPER_OPEN"
      elif (gripper1[0] == "dgc" and gripper2[0] == "dgo"):
        return "GRIPPER_CLOSE"
      elif (gripper1[0] == "dgo" and gripper2[0] == "dgc"):
        return "GRIPPER_OPEN"
      else:
        return "NEW_POSITION"

  def arm_up_down(self, img_name):
      nm = img_name.split("_")
      if nm[0].startswith("idx"):
        gripper = nm[2].split(".")
      else: # nm[0].startswith("image"):
        gripper = nm[3].split(".")
      if gripper[0] in ["dgo", "dgc"]:
        return "DOWN"
      elif gripper[0] in ["gc", "go"]:
        return "UP"
      else:
        print("ERROR: bad up/down position for ", img_name)

###################################################3
  def bootstrap_calibration(self):
      # initialize 
      robot_images = [] # init robot arm
      
      #########################################################
      # {'image_name': 'image_empty.jpg', 'gripper_position': {'X': 13.892806544364808, 'Y': -0.010658253367584278, 'Z': 16.8636682665831, 'Gamma': -0.017645086943255, 'Rot': -0.00255913489736, 'Gripper': 1}, 'image_desc':"gripper not in view of camera", 'gripper_pixel':None},
      #########################################################
      # Go through ordered list of prior calibration data and
      # collect images and state information.
      #########################################################
      wmc = widowx_manual_control.widowx_manual_control(self.wdw, self.robot_camera, self)
      prev_img_data = None
      prev_img_nm   = None
      skipped = False
      for step, img_data in enumerate(self.calib_info):
        if not img_data["image_name"].startswith("image"):
          # this routine only calibrates predefined image locations
          break
        filenm = self.calib_dir + "/" + img_data["image_name"]
        if os.path.isfile(filenm):
          try:
            already_calib = self.get_calib_data(step, "calibrated")
            if already_calib: # could be manually overridden
              prev_img_data = img_data
              prev_img_nm = img_data["image_name"] 
              self.prev_image= Image.fromarray(np.array(cv2.imread(filenm)))
              robot_images.append(self.prev_image)
              self.prev_image_file = filenm
              print(img_data["image_name"],": complete")
              skipped = True
              if step == 0:
                self.img_empty_file = filenm
              continue # data already gathered; skip this step
          except:
            pass # continue with data gathering
          if skipped:
            # we skipped the previous action; need to move positions
            self.move_position(step)
            skipped = False
        ####################
        # store the image in sequence
        print("*********************")
        calib_action = self.check_name(img_data["image_name"], prev_img_nm) 
        print(calib_action)
        print(img_data["image_name"])
        print(img_data["image_desc"])
        if (calib_action.startswith("ERROR")):
          print(calib_action,"cannot determine calibration action")
          print(img_data)
          exit()
        if step == 0 and calib_action == "EMPTY": 
          # first image is always the empty workspace. No arm in image.  
          # Used to contrast with other images and to map tablespace.
          # position should be same as moveOutCamera...
          self.wdw.moveOutCamera()
          print("Ensure the tabletop is empty and press return.")
          wait_for_return = input()
          print("image description:", img_data["image_desc"])
          # self.img_empty = copy.deepcopy(self.latest_image)
          self.post_move_data(step)
          self.img_empty_file = self.latest_image_file
          # make sure calibration of new empty picture works as expected
          tt_anal = self.tabletop_analysis(self.img_empty_file)
        elif step == 0:
          print("ERROR: calibration requires the empty workspace to be the first image.", self.tabletop_state)
          wait_for_return = input()
        elif calib_action == "GRIPPER_CLOSE":
          # if only intended difference in img is "go" vs "gc", then skip tuning
          # and get gripper pixel location.
          gripper_anal = self.openCloseGripper(step-1,step,do_pmd = True)
        elif calib_action == "GRIPPER_OPEN":
          gripper_anal = self.openCloseGripper(step-1,step,do_pmd = True)
        elif calib_action == "MOVE_UP":
          self.moveUpDown(step)
        elif calib_action == "MOVE_DOWN":
          self.moveUpDown(step)
        elif calib_action == "NEW_POSITION":
          print("fine tune the position of the arm if necessary:")
          if self.pos_name not in ["rest"]:
            # Use IK to move gripper for vertical pick-up
            self.move_position(step)
          elif self.pos_name == "rest":
            # Rest does not point the gripper down. Use positioning.
            self.wdw.moveRest()
            pose=self.get_calib_data(step, "gripper_position")
            # now rotate wrist then open/close gripper
            self.wdw.action(vr=pose['Rot'])
            gripper_state, gripper_pos = self.wdw.gripper(pos["Gripper"])
            self.post_move_data(step)
          # allow angling gripper rotation always possible
          while wmc.do_manual_action(img_data["image_name"]): 
            self.post_move_data(step)
        
        robot_images.append(self.latest_image)
        filenm = self.calib_dir + "/image" + str(step) + "_" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ".png"
        # self.latest_image.save(filenm)
        cv2.imwrite(filenm, self.latest_image)
        print("Saved ",filenm)
        filenm = self.calib_dir + "/" + img_data["image_name"]
        # self.latest_image.save(filenm)
        cv2.imwrite(filenm, self.latest_image)
        self.set_calib_data(step, "calibrated", True)
        print("Saved ",filenm)
        self.write_calib()
        print("Saved calibration file")
        prev_img_data = img_data
        prev_img_nm = img_data["image_name"] 
      
      # now that we've updated the metadata, store the calibration images and the metadata
      # display.Image(self.as_gif(robot_images, True))
      # print("Press return to save the new calibration data.")
      # wait_for_return = input()
      
###################################################3
# CALIB BY SWIVEL FUNCTIONS
###################################################3
  ########################################
  # open / close gripper processing: does gripper analysis and stores calib_data entries
  # gn sequence is open/closed/closed/open 
  ########################################
  def calib_open_close_phase(self, gn_seq, sw_angle, reach, vz, redo_idx=None):
      g_n = gn_seq
      first_calib_idx = None
      print("cocp: gn,sw_angle,reach,vz:", g_n, sw_angle, reach, vz)
      for sw_pos in range(2):    # o,c then c,o loop
        if g_n == 4:
          g_n = 0
        if g_n in [0,3]:
          oc = "o"
          goc = self.wdw.GRIPPER_OPEN
        else:
          oc = "c"
          goc = self.wdw.GRIPPER_CLOSED
        VZ_DOWN = self.config["VZ_DOWN"]
        if vz > VZ_DOWN:
          udg = "_g"
        else:
          udg = "_dg"
        # if skip_initial_open_gripper:
        #   continue     # sync the goc restart
        # print("goc: ", goc, oc)
        print("cocp: performing action g_n, goc, oc:", g_n, goc, oc)
        self.wdw.action(goc = goc)
        data = {"gripper_position":copy.deepcopy(self.wdw.state()), "swivel":sw_angle, "gripper_dist":reach}
        print(sw_pos, "cocp: oc data:",data)
        if first_calib_idx is None:
          if redo_idx is not None:
            calib_idx = redo_idx
          else:
            calib_idx = self.find_calib_index_by_position(data, "SAME_ALL")
          print(sw_pos, "cocp: find index by pos:",calib_idx)
        else:
          try:
            matching_idx_name = self.get_calib_data(first_calib_idx, "open_close_gripper_match")
            calib_idx = self.find_calib_data_by_name(matching_idx_name)
            print(sw_pos, "cocp: find index by match:",calib_idx)
            # Rot may need correcting (eg. during check_and_fix). 
            # if we know the corresponding o/c gripper match, we can safely correct
            # before gripper analysis is performed.  
            gpos = self.wdw.state()
            rot = self.compute_rot(gpos["X"], gpos["Y"], sw_angle)
            print("cocp: desired rot, actual:", rot, gpos["Rot"])
            # compare rot with first_calib_idx and see if rot(calib_idx) > DELTA_ANGLE
            rot_dif = abs(gpos["Rot"] - rot)
            if rot_dif > DELTA_ANGLE:
              # redo open/close loop:
              # succ = self.wdw.action(vr=rot)
              # change rot causes non-match by position
              pass
          except:
            calib_idx = self.find_calib_index_by_position(data, "SAME_ALL")
            print(sw_pos, "cocp: find matching index by pos:",calib_idx)
            # Note, if Rot needed correcting, needs manual override to handle it.

        self.post_move_data(None, data)        # add std key values
        calib_idx = self.add_calib_data(calib_idx, data) # create or update entry
        print(sw_pos, "cocp: add_calib_data to",calib_idx, data)
        if first_calib_idx is None:
          # A then B means part of the same AB o/c or c/o sequence
          first_calib_idx = calib_idx
          img_name = "idx" + str(calib_idx) + "_swivA"
        else:
          img_name = "idx" + str(calib_idx) + "_swivB"
        img_name = img_name + udg + oc + ".jpg"
        self.set_calib_data(calib_idx, "image_name", img_name)
        print("cocp: idx1, idx2, img_name ", first_calib_idx, calib_idx, img_name)
        filenm = self.calib_dir + "/" + img_name
        cv2.imwrite(filenm, self.latest_image)
        print("cocp: write latest_img to ",filenm)
        print(calib_idx,"cocp: new calib:", self.calib_info[calib_idx])
        self.write_calib()
        # set up next time throught the loop
        g_n = g_n + 1

      ####
      # Rot may need correcting (eg. during check_and_fix). the best place before the picture is taken
      # and gripper analysis is performed.  But this will screw up the match during the next round!
      # Just do manual override specification of gripper pos instead.
      #
      gpos = self.wdw.state()
      rot = self.compute_rot(gpos["X"], gpos["Y"], sw_angle)
      print("cocp: desired rot, actual:", rot, gpos["Rot"])
      # compare rot with first_calib_idx and see if rot(calib_idx) > DELTA_ANGLE
      # if rot_dif > DELTA_ANGLE:
      #   redo open/close loop:
      #     succ = self.wdw.action(vr=rot)
      #     redo open/close
      #     self.post_move_data(, data)        # add std key values
      ####
      # gripper o/c analysis to finish off logic for gripper comparison
      g_anal = self.openCloseGripper(first_calib_idx, calib_idx)
      if g_anal is None:
        print("WARNING: bad gripper analysis for " + img_name)
      return g_n, first_calib_idx, calib_idx

  def compute_rot(self, X,Y,sw_angle):
      cam_y = -self.TT_WIDTH/2
      cam_x = self.config["CAM_X"]
      rot1  = -math.atan2(abs((Y - cam_y)), abs((X - cam_x)))
      rot2  = rad_sum(rot1, sw_angle) 
      rot = rot2
      while rot > 0:
        rot -= math.pi
      print("rot,1,2, sw_angle:", rot, rot1, rot2, sw_angle)
      return rot

  ######################
  # Do a single sweep by swiveling:
  # - set X = distance, Y = 0, keep same Z
  # - swivel to left, then incrementally swivel to right
  # - with every swivel, rotate gripper so that gripper it can be observed by the grippers
  #   then open/close the gripper and compute the gripping point
  # - store calibrated info
  def calib_swivel_sweep(self, reach, vz, min_sw_angle, max_sw_angle, udg, recalibrate=None):

      ####################################
      # X is the estimated distance from arm center to TT
      def compute_swivel(X,H):
          if X > H:
            X = H
          Y = math.sqrt(H*H - X*X)
          if Y > self.TT_WIDTH/2:
            Y = self.TT_WIDTH/2
            if Y > H:
              Y = H
            X = math.sqrt(H*H - Y*Y)
          sw_angle = np.arctan2(Y, X)
          rot = self.compute_rot(X,Y,sw_angle)
          print("sw_angle, rot", sw_angle, rot)
          return sw_angle, rot

      def virtual_next_swivel(virtual_pos, swiv_delta):
        x0 = virtual_pos["gripper_position"]['X']
        y0 = virtual_pos["gripper_position"]['Y']
        radius = math.sqrt(math.pow(x0,2) + math.pow(y0,2))
        curr_angle = math.atan2(y0, x0)
        # swivel is desired angle (absolute, not relative)
        # compute a swivel detectable by WidowX.cpp
        print("radius, x3,y0, angle: ",radius,x0,y0, curr_angle)
        new_angle = curr_angle - swiv_delta
        new_virtual_pos = copy.deepcopy(virtual_pos)
        new_virtual_pos["gripper_position"]['X'] = math.cos(new_angle) * radius 
        new_virtual_pos["gripper_position"]['Y'] = math.sin(new_angle) * radius 
        new_virtual_pos["gripper_position"]['Rot'] = self.compute_rot(new_virtual_pos["gripper_position"]['X'],new_virtual_pos["gripper_position"]['Y'],new_angle)
        new_virtual_pos["swivel"] = new_angle
        new_virtual_pos["gripper_dist"] = radius
        print("new swivel angle, delta: ", new_angle, swiv_delta)
        print("virtual Swivel :", new_virtual_pos)
        return new_virtual_pos

      def do_next_swivel(swiv_delta):
        sw_angle = self.wdw.do_swivel(left_right= "RIGHT", delta=swiv_delta)
        rot = None
        if sw_angle is not None:
          sw_num = 0
          goal_state = copy.deepcopy(self.wdw.state())
          X = goal_state["X"]
          Y = goal_state["Y"]
          rot = self.compute_rot(X,Y,sw_angle)
          print(sw_num, "rot,X,Y,sw_angle:", rot, X, Y, sw_angle)
          succ = self.wdw.action(swivel=sw_angle)
          succ = self.wdw.action(vr=rot)
        return sw_angle, rot

      ####################################

      calib_idx = len(self.calib_data["calibration_info"])
      # note: swivel angle = 0 is straight ahead.
      # swivel angle starts negative (left) and ends positive (right)
      left_sw_angle, rot = compute_swivel(self.armctr2tt, reach)
      if max_sw_angle is None:
        max_sw_angle = left_sw_angle
      if min_sw_angle is None:
        min_sw_angle = self.config["MIN_SW_ANGLE"]
      DELTA_ANGLE  = self.wdw.DELTA_ANGLE
      num_swiv = self.config["NUM_SWIVELS"]
      swiv_delta = abs((min_sw_angle - max_sw_angle) / num_swiv)
      # Do initial reach straight ahead, but with predicted rot
      success = self.wdw.action(vx=reach, vy=0, vz=vz, vr=0, goc = self.wdw.GRIPPER_OPEN)
      # Do initial swivel
      sw_angle = left_sw_angle
      succ = self.wdw.action(swivel=sw_angle)
      gpos = self.wdw.state()
      rot = self.compute_rot(gpos["X"], gpos["Y"], sw_angle)
      succ = self.wdw.action(vr=rot)
#      if completed_sw_angle is None or (completed_sw_angle is not None and completed_sw_angle < left_sw_angle):
#        sw_angle = left_sw_angle
#        succ = self.wdw.action(swivel=sw_angle)
#        if succ:
#          succ = self.wdw.action(vr=rot)
#      else:
#        succ = self.wdw.action(swivel=completed_sw_angle)
#        if succ:
#          succ = self.wdw.action(vr=rot)
#       sw_angle,rot = do_next_swivel(swiv_delta)
      if not succ:
        print("initial sweep action failed")
        return None, None, None
      print("sw_angle, max, min, num_swiv, sw_delta:", sw_angle, max_sw_angle, min_sw_angle, num_swiv, swiv_delta)
      virtual_pos = {"gripper_position":copy.deepcopy(self.wdw.state()), "swivel":sw_angle, "gripper_dist":reach}
      g_n = 0
      # swivel goes from max_sw_angle to min_sw_angle
      match_list = []
      while sw_angle is not None and sw_angle >= min_sw_angle:
        if recalibrate in ["RESUME","MAIN"]:
          # data = {"gripper_position":copy.deepcopy(self.wdw.state()), "swivel":sw_angle, "gripper_dist":reach}
          calib_idx = self.find_calib_index_by_position(virtual_pos, "SAME_ALL")
          print("sweep calib_idx, virtual_pos:", calib_idx, virtual_pos)
          if calib_idx is not None:
            calib_img_nm = self.get_calib_data(calib_idx, "image_name")
            calib_idx_pos = self.get_calib_data(calib_idx, "gripper_position")
            calib_idx_sw_angle = self.get_calib_data(calib_idx, "swivel")
            calib_idx_reach = self.get_calib_data(calib_idx, "gripper_dist")
            print("swivel angle previously done for this reach. Skip:", reach, sw_angle, calib_idx)
            match_found = False
            if calib_img_nm in match_list:
              match_found = True
            else:
              try:
                matching_img_nm = self.get_calib_data(calib_idx, "open_close_gripper_match")
                match_list.append(calib_img_nm)
                match_list.append(matching_img_nm)
                match_found = True
              except:
                try:
                  calib_img_nm = self.get_calib_data_by_name(calib_idx, "image_name")
                  if calib_img_nm.contains("swivA"):
                    matching_img_nm = self.get_calib_data(calib_idx+1, "image_name")
                    if matching_img_nm.contains("swivB"):
                      matching_idx_pos = self.get_calib_data(calib_idx+1, "gripper_position")
                      if self.match_position(calib_idx_pos, matching_idx_pos, "SAME_ALL"):
                        match_list.append(calib_img_nm)
                        match_list.append(matching_img_nm)
                        match_found = True
                except:
                  pass
              if match_found:
                virtual_pos = {"gripper_position":copy.deepcopy(calib_idx_pos), "swivel":calib_idx_sw_angle, "gripper_dist":calib_idx_reach}
                virtual_pos = virtual_next_swivel(virtual_pos, swiv_delta)
                sw_angle = virtual_pos['swivel']
                continue    # previously done: skip
              else:
                print("no match for ", calib_img_nm)
                if calib_idx_pos is not None:
                  print("Reanalyzing ", calib_idx_pos)
        g_n, first_calib_idx, calib_idx = self.calib_open_close_phase(g_n, sw_angle, reach, vz)
        print("min_sw_angle, sw_angle, swiv_delta", min_sw_angle, sw_angle, swiv_delta)
        if (swiv_delta != 0 and min_sw_angle < sw_angle - swiv_delta):
          sw_angle,rot = do_next_swivel(swiv_delta)
          print("do_next_swivel: ", sw_angle)
          print("sw_angle, max, min, num_swiv, sw_delta:", sw_angle, max_sw_angle, min_sw_angle, num_swiv, swiv_delta)
        else:
          sw_angle = None
          print("swivel completed:", calib_idx, sw_angle)
          break 

  def calib_swivel_angles(self, arm_reach):
      vpl = self.calib_data["metadata"]["best_vpl"]
      vpllns = self.tt.get_vpl_lines(vpl)
      DELTA_ACTION  = self.wdw.DELTA_ACTION
      VZ_UP = self.config["VZ_UP"]
      ggp = None
      # only works for shortest, longest in VX_SET[], the shortest "MAIN" calib reach.
      data_pos = {"gripper_position": {"X":arm_reach, "Y":0, "Z": VZ_UP}}
      for idx, calib_data in enumerate(self.calib_data["calibration_info"]):
        idx_pos = self.get_calib_data(idx, "gripper_position")
        if self.match_position(data_pos, idx_pos, "UP"):
          try:
            gripper_dist = self.get_calib_data(calib_idx, "gripper_dist")
            ggp = self.get_calib_data(calib_idx, "gripper_grasping_point")
            print("ggp, gripper_dist, gripper_pos", ggp, gripper_dist, data)
            break
          except:
            continue

      far_left_angle = None
      far_right_angle = None
      dist_to_arm_ctr = None
      print("ggp",ggp)
      if ggp is not None:
        # find pt on VPL intersecting front-facing Arm and convert to gripper space (cm)
        min_dist = 1000000000
        for n,vln in enumerate(vpllns):
          closest_pt = self.lna.closest_pt_on_ln(vln, ggp)
          closest_pt_cm = self.pixel_gripper_conversion(closest_pt[0], closest_pt[1], up_down="DOWN", dir="P2G")
          ggp_cm = self.pixel_gripper_conversion(ggp[0], ggp[1], up_down="DOWN", dir="P2G")
          dist_to_ggp = self.lna.get_dist(closest_pt_cm[0],closest_pt_cm[1], ggp_cm[0], ggp_cm[1])
          print("cp,cp_cm,ggp,d2ggp",closest_pt, closest_pt_cm,ggp_cm,dist_to_ggp)
          if min_dist > dist_to_ggp:
            min_dist = dist_to_ggp
        dist_to_arm_ctr = arm_reach - min_dist
        print("d2ac", dist_to_arm_ctr)
        ########
        # sin = opp/h where h = arm_reach, opp=dist_to_arm_ctr
        far_left_angle = math.pi/2 + math.asin(dist_to_arm_ctr / arm_reach)
        far_right_angle = -math.pi/2 - math.asin(dist_to_arm_ctr / arm_reach)
      return far_left_angle, far_right_angle, dist_to_arm_ctr

  def get_mode_ranges(self):
      VX_SET = self.config["VX_SET"]
      # if mode in ["MAIN","CLOSE","FAR","ALL"]:
      if self.main_range[0] is None:
        for n,vx in enumerate(VX_SET):
          if (n < len(VX_SET) and vx+1 == VX_SET[n+1]):
            self.main_range[0] = vx+1
          else:
            break
      if self.main_range[1] is None:
        for n,vx in enumerate(reversed(VX_SET)):
          if n < len(VX_SET) and vx-1 == VX_SET[-n-2]:
            self.main_range[1] = vx-1
          else:
            break
      print("main_range:",self.main_range)

  def calibrate_by_swivel(self, mode="MAIN", recalibrate=None):

      # initialize 
      robot_images = [] # init robot arm
      #########################################################
      print("Light the tabletop from camera side so shadows are behind arm in picture.")
      print("Tabletop should be empty.")
      prev_img_data = None
      prev_img_nm   = None
      DELTA_ANGLE  = self.wdw.DELTA_ANGLE
      DELTA_ACTION  = self.wdw.DELTA_ACTION

      ######################
      # start from pick position, find MIN/MAX positions
      # round(x,2)
      self.wdw.moveArmPick()
      self.wdw.set_move_mode('Absolute')
      calib_idx = len(self.calib_data["calibration_info"])
      ######################
      # "MAGIC NUMBERS" obtained from earlier calibration
      # FAR_LEFT_ARM_WRIST_ROT was from from FLBOLT
      # MIDDLE_ARM_WRIST_ROT was from pick
      VZ_DOWN = self.config["VZ_DOWN"]
      FAR_LEFT_ARM_WRIST_ROT = self.config["FAR_LEFT_ARM_WRIST_ROT"]
      MIDDLE_ARM_WRIST_ROT = self.config["MIDDLE_ARM_WRIST_ROT"]
      TT_FRONT = self.config["TT_FRONT"]
      tt_left = -(self.TT_WIDTH/2)  # cm
      GRIPPER_POINTED_DOWN = (-math.pi/2)

      straight_angle = np.arctan2(0, 1)
      num_swiv = self.config["NUM_SWIVELS"]
      delta_rot = (FAR_LEFT_ARM_WRIST_ROT - MIDDLE_ARM_WRIST_ROT) / num_swiv
      min_sw_angle = self.config["MIN_SW_ANGLE"]  # negative, right side

      #####################################
      # Restart calibration from where left off based on last entry of calibration file.
      # last swivel entry is index calib_idx-1
      ####
#      try:
#        img_name = self.get_calib_data(calib_idx-1, "image_name")
#      except:
#        img_name = None
#      print("swiv_calib img_name:", img_name)
#      try:
#        prev_img_name = self.get_calib_data(calib_idx-2, "image_name")
#      except:
#        prev_img_name = None
#      print("prev swiv_calib img_name:", prev_img_name)
      ################
      # init to be first time doing swivel calibration
      ################
      VX_SET = self.config["VX_SET"]
      VZ_UP = self.config["VZ_UP"]
      vx_set = []
      vz_up_set = {}
      sw_angle_set = {}
      self.armctr2tt = min(VX_SET) # guess at distance from arm center to tabletop
      if mode in ["MAIN","CLOSE","FAR","ALL"]:
        self.get_mode_ranges()
        for n,vx in enumerate(VX_SET):
          if mode == "MAIN" and (vx <= self.main_range[0] or vx >= self.main_range[1]):
            continue
          elif mode == "CLOSE" and vx < self.main_range[0]:
            continue
          elif mode == "FAR" and vx > self.main_range[1]:
            continue
          vx_set.append(vx)
          vz_up_set[vx] = VZ_UP
          sw_angle_set[vx] = [None, None]  # use defaults
          # if vx == min(VX_SET):
          #   SW_ANGLE_SET[vx] = [0, 0]
          # else:
          #   SW_ANGLE_SET[vx] = [None, None]  # use defaults
#      elif mode=="CLOSE":
#        print("CLOSE CALIB")
#        # must run after initial lst sqr computaton to conv pixels to gripper space (cm)
#        min_vx_reach = min(VX_SET)
#        max_vx_reach = max(VX_SET)
#        VX_SET = []
#        VZ_UP_SET = {}
#        far_left_angle, far_right_angle, self.armctr2tt = self.calib_swivel_angles(min_vx_reach)
#
#        VX_SET.append(min_vx_reach)
#        SW_ANGLE_SET[min_vx_reach] = [far_left_angle, far_right_angle]
#        VZ_UP_SET[min_vx_reach] = VZ_UP
#
#        close_sw = (min_vx_reach + self.armctr2tt)/2
#        # vz_delta = min_vx_reach - (min_vx_reach + self.armctr2tt)/2
#      elif mode=="CLOSE":
#        print("CLOSE CALIB")
#        # must run after initial lst sqr computaton to conv pixels to gripper space (cm)
#        min_vx_reach = min(VX_SET)
#        max_vx_reach = max(VX_SET)
#        VX_SET = []
#        VZ_UP_SET = {}
#        far_left_angle, far_right_angle, self.armctr2tt = self.calib_swivel_angles(min_vx_reach)
#
#        VX_SET.append(min_vx_reach)
#        SW_ANGLE_SET[min_vx_reach] = [far_left_angle, far_right_angle]
#        VZ_UP_SET[min_vx_reach] = VZ_UP
#
#        close_sw = (min_vx_reach + self.armctr2tt)/2
#        # vz_delta = min_vx_reach - (min_vx_reach + self.armctr2tt)/2
#        vz_delta = 0
#        VX_SET.append(close_sw)
#        SW_ANGLE_SET[close_sw] = [far_left_angle, far_right_angle]
#        VZ_UP_SET[close_sw] = VZ_UP + vz_delta
#
#        # VX_SET.append(self.armctr2tt)
#        # SW_ANGLE_SET[self.armctr2tt] = [0, 0]
#        # VZ_UP_SET[self.armctr2tt] = VZ_UP + 2*vz_delta
#
#      elif mode=="FAR":
#        print("FAR CALIB")
#        max_vx_reach = max(VX_SET)
#        # run after initial linear approximation to compute self.armctr2tt
#        far_left_angle, far_right_angle, self.armctr2tt = self.calib_swivel_angles(max_vx_reach)
#        # far_vz_min = VZ_DOWN + 3
#        # ratio (X^2 / VZ^2) needs to be about the same
#        vz_delta = 0
#        # far_vz_up  = VZ_UP - vz_delta
#        far_vz_up  = VZ_UP 
#        far_vx_main = max_vx_reach + 1
#        VX_SET = []
#        # while far_vz_up >= far_vz_min:
#        while far_vx_main <= 29:
#          VX_SET.append(far_vx_main)
#          VZ_UP_SET[far_vx_main] = far_vz_up
#          SW_ANGLE_SET[far_vx_main] = [None, None]  # use defaults
#          far_vx_main += 1
#          far_vz_up  -= vz_delta
#          # For testing reaches:
#          # success = self.wdw.action(vx=far_vx_main, vy=0, vz=far_vz_up, vr=0, goc = self.wdw.GRIPPER_OPEN)
#          # success = self.wdw.action(vx=far_vx_main, vy=0, vz=2.6, vr=0, goc = self.wdw.GRIPPER_OPEN)
#        # print("EXITING")
#        # exit()

      #####################################
      # Execute Swivel Calibration by starting where left off
      ###
      ###########################
      # set reach distance
      print("VX,VZ,SW", vx_set, vz_up_set, sw_angle_set)
      prev_delta_vz = 0
      for vx in vx_set:      # gripper distance loop
        min_vx_reach = min(vx_set)
        max_sw_angle, min_sw_angle = sw_angle_set[vx]
        # sync up/down loop to last run
        for vz in [vz_up_set[vx], VZ_DOWN]: # up/down loop
          if vz == vz_up_set[vx]:
            udg = "_g"
          else:
            udg = "_dg"
          self.calib_swivel_sweep(vx, vz, min_sw_angle, max_sw_angle, udg, recalibrate)

##################################

  def tabletop_analysis(self, img):
      # find linear approx to pick point
      #
      found_vpl = None
      seg_meth = ["CONFIG_CHOICE", "COLOR","COLOR_MINUS_ONE", "COLOR_PLUS_ONE","STEGO","NONE"]

      config_segmentation_method = self.calib_data["metadata"]["tabletop_segmentation_method"]
      for do_seg_meth in seg_meth:    
        if do_seg_meth == "CONFIG_CHOICE":
          found_vpl = self.tt.calibrate_tabletop(img, method=config_segmentation_method)
        elif do_seg_meth == config_segmentation_method:
          continue
        elif do_seg_meth == "COLOR" or do_seg_meth == "STEGO":
          found_vpl = self.tt.calibrate_tabletop(img, method=do_seg_meth)
        elif do_seg_meth == "COLOR_MINUS_ONE":
          self.calib_data["metadata"]["num_color_clusters"] -= 1
          found_vpl = self.tt.calibrate_tabletop(img, method="COLOR")
          self.calib_data["metadata"]["num_color_clusters"] += 1
        elif do_seg_meth == "COLOR_PLUS_ONE":
          self.calib_data["metadata"]["num_color_clusters"] += 1
          found_vpl = self.tt.calibrate_tabletop(img, method="COLOR")
          self.calib_data["metadata"]["num_color_clusters"] -= 1
        elif do_seg_meth == "NONE":
          found_vpl = self.tt.calibrate_tabletop(img, method=None)
        if found_vpl:
          break
      self.calib_data["metadata"]["best_vpl"] = self.tt.get_best_vpl()
      print("calibrate_tabletop: found_vpl", found_vpl)
      if found_vpl:
          self.vpl_status["FOUND"]  = True
      cv2.waitKey(0)



##################################
  #########################
  # Used by Swivel Calibration to map the pixel coordinates to gripper coords (cm)

# WARNING: no gripper_grasping_point for idx29_swivA_go.jpg ; Skipping.
# WARNING: no gripper_grasping_point for idx30_swivB_gc.jpg ; Skipping.
# WARNING: no gripper_grasping_point for idx159_swivA_gc.jpg ; Skipping.
# WARNING: no gripper_grasping_point for idx160_swivB_go.jpg ; Skipping.
# WARNING: no gripper_grasping_point for idx161_swivA_go.jpg ; Skipping.
# WARNING: no gripper_grasping_point for idx162_swivA_gc.jpg ; Skipping.
# WARNING: no gripper_grasping_point for idx163_swivB_gc.jpg ; Skipping.
# WARNING: no gripper_grasping_point for idx164_swivA_gc.jpg ; Skipping.
  def check_and_fix_calib_entries(self):
      self.wdw.set_move_mode('Absolute')
      redo = []
      for num, imgdata in enumerate(self.calib_info):
        try:
          img_name = imgdata["image_name"]
        except:
          print("WARNING: no img_name for calib idx ", num)
          redo.append(num)
          img_name = None
        if img_name == "image_empty.jpg":
          print("skipping ", img_name)
          continue
        try:
          gocp = self.get_calib_data(num, "open_close_gripper_pos")
        except:
          print("WARNING: ignoring missing open_close_gripper_pos for", img_name)
        try:
          [c,r] = self.get_calib_data(num, "gripper_grasping_point")
        except:
          print("WARNING: no gripper_grasping_point for", img_name)
          redo.append(num)
      print("Idx# that need to be fixed:", redo)
      redo_done = []
      for idx in redo:
        if idx in redo_done:
          continue
        print("fixing idx", idx)
        # get position
        idx_data = self.calib_data["calibration_info"][idx]
        idx_data2 = copy.deepcopy(self.calib_data["calibration_info"][idx])
        gpos  = self.get_calib_data(idx, "gripper_position")
        try:
          sw_angle = self.get_calib_data(idx, "swivel")
          reach = self.get_calib_data(idx, "gripper_dist")
        except:
          sw_angle = math.atan2(gpos['Y'], gpos['X'])
          reach = self.lna.get_dist(0, 0, gpos['Y'], gpos['X'])
        # gocp = self.get_calib_data(idx, "open_close_gripper_pos")
        succ = self.wdw.action(vx=reach, vy=0, vz=gpos['Z'], vr=0, goc = gpos["Gripper"])
        self.wdw.set_move_mode('Absolute')
        succ = self.wdw.action(swivel=sw_angle)
        succ = self.wdw.action(vr=gpos["Rot"])
        img_name = self.get_calib_data(idx, "image_name")
        if "go" in img_name:
          g_n = 0
        elif "gc" in img_name:
          g_n = 2
        g_n, first_calib_idx, calib_idx = self.calib_open_close_phase(g_n, sw_angle, reach, gpos['Z'], idx)
        if idx not in [first_calib_idx, calib_idx]:
          print("WARNING: bad idx match (expected, found):", idx, first_calib_idx, calib_idx)
        else:
          redo_done.append(first_calib_idx)
          redo_done.append(calib_idx)
        self.write_calib()
        print("Saved calibration file")
      redo_unfixed = list(set(redo) - set(redo_done))
      if len(redo_unfixed) > 0:
        print("Redo not fixed for idx ",redo_unfixed)
        wait_for_return = input()

  def calibrate_pixel_to_gripper(self, do_plots=False):
      bb_ugx,bb_ugy,bb_ugz,bb_upr,bb_upc = [],[],[],[],[]
      nobb_ugx,nobb_ugy,nobb_ugz,nobb_upr,nobb_upc = [],[],[],[],[]
      bb_dgx,bb_dgy,bb_dgz,bb_dpr,bb_dpc,bb_dbx,bb_dby = [],[],[],[],[],[],[]
      nobb_dgx,nobb_dgy,nobb_dgz,nobb_dpr,nobb_dpc,nobb_dbx,nobb_dby = [],[],[],[],[],[],[]
      ugx,ugy,ugz,upr,upc = [],[],[],[],[]
      dgx,dgy,dgz,dpr,dpc,dbx,dby = [],[],[],[],[],[],[]
      prv_img_nm = None
      self.closed_gripper_pos = []
      self.open_gripper_pos = []
      for num, imgdata in enumerate(self.calib_info):
        try:
          img_name = imgdata["image_name"]
        except:
          print("WARNING: no img_name for calib idx ", num)
          img_name = None
        if num == 0:
          if img_name != "image_empty.jpg":
            print("Warning: Empty image expected to be first calibration entry")
          continue
        gpos  = self.get_calib_data(num, "gripper_position")
        goc = gpos["Gripper"]
        try:
          gbb   = self.get_calib_data(num, "gripper_bounding_box")
        except:
          gbb   = None
        # print("gpos:", gpos)
        up_down = self.arm_up_down(img_name)
        try: 
          [c,r] = self.get_calib_data(num, "gripper_grasping_point")
        except:
          print("WARNING: no gripper_grasping_point for", img_name,"; Skipping.")
          continue
        try:
          gocp = self.get_calib_data(num, "open_close_gripper_pos")
          if goc == self.wdw.GRIPPER_CLOSED and gocp[1] is not None:
            self.closed_gripper_pos.append(gocp[1])
          elif goc == self.wdw.GRIPPER_OPEN and gocp[0] is not None:
            self.open_gripper_pos.append(gocp[0])
        except:
          print("WARNING: ignoring missing open_close_gripper_pos for", img_name)
        img_state = self.img_name_state(img_name)
        if img_state == "rest":
          print("skip rest positions")
          continue
        # print("img_st, r,c",img_state,r,c)
        if up_down == "UP":
          if gbb is None:
            nobb_ugx.append(gpos["X"])
            nobb_ugy.append(gpos["Y"])
            nobb_ugz.append(gpos["Z"])
          else:
            bb_ugx.append(gpos["X"])
            bb_ugy.append(gpos["Y"])
            bb_ugz.append(gpos["Z"])
          ugx.append(gpos["X"])
          ugy.append(gpos["Y"])
          ugz.append(gpos["Z"])
          # pixel row
          upr.append(r)
          upc.append(c)
        elif up_down == "DOWN":
          if gbb is None:
            nobb_dgx.append(gpos["X"])
            nobb_dgy.append(gpos["Y"])
            nobb_dgz.append(gpos["Z"])
            nobb_dpr.append(r)
            nobb_dpc.append(c)
            nobb_BEVr, nobb_BEVc = self.tt.BEV_x_y(r, c) 
            nobb_dbx.append(nobb_BEVr)
            nobb_dby.append(nobb_BEVc)
          else:
            bb_dgx.append(gpos["X"])
            bb_dgy.append(gpos["Y"])
            bb_dgz.append(gpos["Z"])
            bb_dpr.append(r)
            bb_dpc.append(c)
            bb_BEVr, bb_BEVc = self.tt.BEV_x_y(r, c) 

          dgx.append(gpos["X"])
          dgy.append(gpos["Y"])
          dgz.append(gpos["Z"])
          dpr.append(r)
          dpc.append(c)
          # BEV transform: use r,c or c,r ???
          BEVr, BEVc = self.tt.BEV_x_y(r, c) 
   
      # gx = A1px + B1py + C1
      # gy = A2px + B2py + C2
      # gz = C3
      dzc            = np.mean(dgz)
      self.ARM_DOWN_Z = dzc
      print("dgz:", dzc)
      uzc            = np.mean(ugz)
      self.ARM_UP_Z = uzc
      print("Pixel to Down Gripper position")
      bdpr = []
      bdpc = []
      for n, pr in enumerate(dpr):
        pc = dpc[n]
        BEVr, BEVc = self.tt.BEV_x_y(pr, pc)
        bdpr.append(BEVr)
        bdpc.append(BEVc)
      ix = []
      iy = []
      for n, bx in enumerate(dpr):
        by = dgy[n]
        bix, biy = self.tt.BEV_to_image_x_y(bx, by)
        ix.append(bix)
        iy.append(biy)
      print("Bev2img x:", ix)
      print("Bev2img x:", iy)

#     #############################
      # The following is done via the VPL math in analyze_tabletop
#     #############################
      print("########################")
      print("The Winner:  BEVxy -> Dwn Pix XY")
      rda2, rdb2, rdc2 = self.lst_sq(ix, iy,  dpr)
      cda2, cdb2, cdc2 = self.lst_sq(ix, iy,  dpc)

#     #############################
#     Now Solve for the Arm's X,Y in terms of snapshot's Pixel Row, C Col.
#     #############################
#     # Step 1:
#     # BX, BY in terms of PR, PC #
#     #############################
#
#     BX = (1 / (ra2 + rb2 * ca2 / cb2 ) * PR
#          - (rb2 / cb2 ) / (ra2 + rb2 * ca2 / cb2 )  * PC
#          + (rc2) / (ra2 + rb2 * ca2 / cb2 )
#     BY = (PC - ca2 * BX + cc2) / cb2
#
#     #############################
#     # Step 2:
#     # X, Y in terms of BX, BY   #
#     #############################
#
#     c = (1 - (self.BEV_M[0][1] / self.BEV_M[0][0]) * self.BEV_M[1][0])
#     x = (1 / self.BEV_M[0][0]) / c  * BEV_X
#         - self.BEV_M[0][1] / self.BEV_M[0][0])  / c* BEV_Y
#         + (self.BEV_M[0][1] / self.BEV_M[0][0]) * (self.BEV_M[1][2]/ self.BEV_M[1][0] - self.BEV_M[0][2] / c)
#     y = (BEV_y - self.BEV_M[1][0]*x - self.BEV_M[1][2])/ self.BEV_M[1][0]

#     #############################
#     # BX, BY in terms of PR, PC #
      pra = 1 / (rda2 + rdb2 * cda2 / cdb2 )
      prb = - (rdb2 / cdb2 ) / (rda2 + rdb2 * cda2 / cdb2 )
      prc = + (rdc2) / (rda2 + rdb2 * cda2 / cdb2 )

      X = np.array(dpr)
      Y = np.array(dpc)
      BX = pra * X + prb * Y + prc
      Z = np.array(BX)
      print("orig pra,b,c", pra, prb, prc)
      pra, prb, pr2 = self.lst_sq(X, Y, BX)
      print("lsqr pra,b,c", pra, prb, prc)

      if False and do_plots:
        print("Plotting PIX_C, PIX_R to BEV_X")
        self.plot(X,Y,Z, pra,prb,prc)

#     #############################
#     # BEV_X, BEV_Y to X,Y
      c = (1 - (self.tt.BEV_M[0][1] / self.tt.BEV_M[0][0]) * self.tt.BEV_M[1][0])
      xa = (1 / self.tt.BEV_M[0][0]) / c
      xb = - (self.tt.BEV_M[0][1] / self.tt.BEV_M[0][0])  / c
      xc = (self.tt.BEV_M[0][1] / self.tt.BEV_M[0][0]) * (self.tt.BEV_M[1][2] / self.tt.BEV_M[1][0] - self.tt.BEV_M[0][2] / c)

      BY = (dpc - cda2 * BX + cdc2) / cdb2
      if False and do_plots:
        print("BY:", BY)
        print("Plotting PIX_C, PIX_R to BEV_Y")

      pra2, prb2, prc2 = self.lst_sq(X, Y, BY)
      # x = xa*BX + xb*BY + xc
      xa, xb, xc = self.lst_sq(BX, BY,  dgx)
      if do_plots:
        print("Plotting BEV_X, BEV_Y to x")
        X = np.array(BX)
        Y = np.array(BY)
        # Z = np.array(x)
        Z = np.array(dgx)
        self.plot(X,Y,Z, xa,xb,xc)

      if True and do_plots:
        bb_X = np.array(bb_dpr)
        bb_Y = np.array(bb_dpc)
        bb_ix = []
        bb_iy = []
        for n, bx in enumerate(bb_dpr):
          by = bb_dgy[n]
          bix, biy = self.tt.BEV_to_image_x_y(bx, by)
          bb_ix.append(bix)
          bb_iy.append(biy)
        bb_rda2, bb_rdb2, bb_rdc2 = self.lst_sq(bb_ix, bb_iy,  bb_dpr)
        bb_cda2, bb_cdb2, bb_cdc2 = self.lst_sq(bb_ix, bb_iy,  bb_dpc)
        bb_prc = + (bb_rdc2) / (bb_rda2 + bb_rdb2 * bb_cda2 / bb_cdb2 )
        bb_pra, bb_prb, bb_pr2 = self.lst_sq(X, Y, BX)
        bb_BX = bb_pra * bb_X + bb_prb * bb_Y + bb_prc
        bb_BY = (bb_dpc - bb_cda2 * bb_BX + bb_cdc2) / bb_cdb2
        bb_xa, bb_xb, bb_xc = self.lst_sq(bb_BX, bb_BY,  bb_dgx)
        print("Plotting BB BEV_X, BEV_Y to bb_x")
        bb_X = np.array(bb_BX)
        bb_Y = np.array(bb_BY)
        bb_Z = np.array(bb_dgx)
        self.plot(bb_X,bb_Y,bb_Z, bb_xa,bb_xb,bb_xc)
      if True and do_plots:
        nobb_X = np.array(nobb_dpr)
        nobb_Y = np.array(nobb_dpc)
        nobb_ix = []
        nobb_iy = []
        for n, bx in enumerate(nobb_dpr):
          by = nobb_dgy[n]
          bix, biy = self.tt.BEV_to_image_x_y(bx, by)
          nobb_ix.append(bix)
          nobb_iy.append(biy)
        nobb_rda2, nobb_rdb2, nobb_rdc2 = self.lst_sq(nobb_ix, nobb_iy,  nobb_dpr)
        nobb_cda2, nobb_cdb2, nobb_cdc2 = self.lst_sq(nobb_ix, nobb_iy,  nobb_dpc)
        nobb_prc = + (nobb_rdc2) / (nobb_rda2 + nobb_rdb2 * nobb_cda2 / nobb_cdb2 )
        nobb_pra, nobb_prb, nobb_pr2 = self.lst_sq(X, Y, BX)
        nobb_BX = nobb_pra * nobb_X + nobb_prb * nobb_Y + nobb_prc
        nobb_BY = (nobb_dpc - nobb_cda2 * nobb_BX + nobb_cdc2) / nobb_cdb2
        nobb_xa, nobb_xb, nobb_xc = self.lst_sq(nobb_BX, nobb_BY,  nobb_dgx)
        print("Plotting NOBB BEV_X, BEV_Y to nobb_x")
        nobb_X = np.array(nobb_BX)
        nobb_Y = np.array(nobb_BY)
        nobb_Z = np.array(nobb_dgx)
        self.plot(nobb_X,nobb_Y,nobb_Z, nobb_xa,nobb_xb,nobb_xc)
      if True and do_plots:
        # RESULT: bb needs to be adjusted to be more like nobb 
        nobb_xa2, nobb_xb2, nobb_xc2 = nobb_xa.copy(), nobb_xb.copy(), nobb_xc.copy()
        bb_xa, bb_xb, bb_xc = self.lst_sq(bb_BX, bb_BY,  bb_dgx)
        print("   bb_xa, xb, xc:", bb_xa,bb_xb,bb_xc)
        print("no bb_xa, xb, xc:", nobb_xa,nobb_xb,nobb_xc)
        print("      xa  sb  xc:", xa,xb,xc)

        self.plot(nobb_X,nobb_Y,nobb_Z, nobb_xa,nobb_xb,nobb_xc)
      #  plot(self, X, Y, Z, a, b, c)
      # plots
      # plot plane fit as grid of green dots
      # xs = np.linspace(X.min(), X.max(), 10)
      # ys = np.linspace(min(Y), max(Y), 10)
      # xv, yv = np.meshgrid(xs, ys)
      # zv = a*xv + b*yv + c

      # print("Plotting BEV_X, BEV_Y to y")
      # y = (BY - self.tt.BEV_M[1][0]*x - self.tt.BEV_M[1][2])/ self.tt.BEV_M[1][0]
      # ya, yb, yc = self.lst_sq(BX, BY,  y)
      ya, yb, yc = self.lst_sq(BX, BY,  dgy)
      print("ya, yb, yc:", ya, yb, yc)

      if do_plots:
        # print("Y:", y)
        Z = np.array(dgy)
        print("Plotting BEV_X, BEV_Y to y")
        self.plot(X,Y,Z, ya,yb,yc)

      print("BEV2Img xy -> Dwn Pix XY")
      self.pix_to_grip = {"X_Y_to_BEV_X":[rda2, rdb2, rdc2], "X_Y_to_BEV_Y":[cda2, cdb2, cdc2],
                          "PIX_R_C_to_BEV_X":[pra, prb, prc],
                          "PIX_R_C_to_BEV_Y":[pra2, prb2, prc2],
                          "BEV_X_Y_to_X":[xa, xb, xc], "BEV_X_Y_to_Y":[ya, yb, yc]}
      print("pix_to_grip:", self.pix_to_grip)
      self.calib_data["metadata"]["pixel_to_gripper"] = copy.deepcopy(self.pix_to_grip)
      print("open gripper position:", self.open_gripper_pos)
      print("mean open gripper position:", np.mean(np.array(self.open_gripper_pos)))
      print("closed gripper position:", self.closed_gripper_pos)
      print("mean closed gripper position:", np.mean(np.array(self.closed_gripper_pos)))
      print("up gripper position:", np.mean(np.array(ugz)), ugz)
      print("down gripper position:", np.mean(np.array(dgz)), ugz)
      self.calib_data["metadata"]["open gripper position"] = np.mean(np.array(self.open_gripper_pos))
      self.calib_data["metadata"]["closed gripper position"] = np.mean(np.array(self.closed_gripper_pos))
      self.calib_data["metadata"]["up gripper position"] = np.mean(np.array(ugz))
      self.calib_data["metadata"]["down gripper position"] = np.mean(np.array(dgz))
      self.write_calib()



  def plot(self, X, Y, Z, a, b, c):
      # plots
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')

      # plot data as big red crosses
      ax.scatter(X, Y, Z, color='r', marker='+', linewidth=10)

      # plot plane fit as grid of green dots
      xs = np.linspace(X.min(), X.max(), 10)
      ys = np.linspace(min(Y), max(Y), 10)
      xv, yv = np.meshgrid(xs, ys)
      zv = a*xv + b*yv + c

      ax.scatter(xv, yv, zv, color = 'g')
      ax.plot_wireframe(xv, yv, zv, color = 'g') # alternative fit plane plot
      plt.show()

  # https://stackoverflow.com/questions/42841632/get-best-linear-function-which-approximate-some-dots-in-3d
  def lst_sq(self, X, Y, Z):
      X = np.array(X)
      Y = np.array(Y)
      Z = np.array(Z)
      print("X_C", X)
      print("Y_R", Y)
      print("CRXY", Z)
      # z = a*x + b*y + c
      # least squares fit
      A = np.vstack([X, Y, np.ones(len(X))]).T
      # print("A", A)
      a,b,c=  np.linalg.lstsq(A, Z)[0]
      # self.plot(X,Y,Z, a,b,c)
      return a,b,c

  def pixel_gripper_conversion(self, x_r, y_c, up_down="DOWN",dir="G2P"):
      #  self.pix_to_grip keys: ["X_Y_to_BEV_X", "X_Y_to_BEV_Y",
      #                          "PIX_R_C_to_BEV_X", "PIX_R_C_to_BEV_Y",
      #                          "BEV_X_Y_to_X", "BEV_X_Y_to_Y"]
      self.pix_to_grip = self.calib_data["metadata"]["pixel_to_gripper"]
      gz_up = self.calib_data["metadata"]["up gripper position"]
      gz_down = self.calib_data["metadata"]["down gripper position"]
      if up_down == "DOWN":
        z = gz_down
      elif up_down == "UP":
        z = gz_up
      else:
        print("ERROR1: pixel_gripper_conversion")
        return None

      if (dir == "G2P"):
        a1,b1,c1 = self.pix_to_grip["X_Y_to_BEV_X"]
        BX = a1*x_r + b1*y_c + c1
        a2,b2,c2 = self.pix_to_grip["X_Y_to_BEV_Y"]
        BY = a2*x_r + b2*y_c + c2
        return BX, BY, z
      elif (dir == "P2G"):
        a1,b1,c1 = self.pix_to_grip["PIX_R_C_to_BEV_X"]
        print("a1,b1,c1:", a1,b1,c1)
        BX = a1*x_r + b1*y_c + c1
        a2,b2,c2 = self.pix_to_grip["PIX_R_C_to_BEV_Y"]
        # print("a2,b2,c2:", a2,b2,c2)
        BY = a2*x_r + b2*y_c + c2
        # print("P2G: pr, pc, BX, BY:",  x_r, y_c, BX, BY)

        a3,b3,c3 = self.pix_to_grip["BEV_X_Y_to_X"]
        # print("a3,b3,c3:", a3,b3,c3)
        x = (a3*BX + b3*BY + c3)
        a4,b4,c4 = self.pix_to_grip["BEV_X_Y_to_Y"]
        # print("a4,b4,c4:", a4,b4,c4)
        y = (a4*BX + b4*BY + c4)
        # y = (BY - self.tt.BEV_M[1][0]*x - self.tt.BEV_M[1][2])/ self.tt.BEV_M[1][0]
        print("P2G: BX, BY, x, y, z:",  BX, BY, x, y, z)
        return x, y, z
      else:
        print("ERROR2: pixel_gripper_conversion")
        return None

  # pr, pc = pixel row, column at Z_DOWN
  # tt_loc[0] = pixel row at drop Z
  # tt_loc[1] = pixel col at drop Z
  def pixel_to_gripper_loc(self, pr, pc, up_down=None, tt_loc=None):
      if tt_loc is None:
        BEVx, BEVy = self.tt.BEV_x_y(pr, pc) 
        print("FYI: BEV r,c=> x,y:", pr,pc, BEVx, BEVy)
        return self.pixel_gripper_conversion(pr, pc, up_down=up_down, dir="P2G")
      elif up_down is not None: 
        pix_c, pix_r = tt_loc[0], tt_loc[1]
        xd,yd,zd = self.pixel_gripper_conversion(pr, pc, up_down="DOWN", dir="P2G")
        pru,pcu,zu = self.pixel_gripper_conversion(xd, yd, up_down="UP", dir="G2P")

        max_pix_dif = np.linalg.norm(np.array([pru,pcu]) - np.array([pr, pc]))
        drop_pix_dif = np.linalg.norm(np.array([pix_r,pix_c]) - np.array([pr, pc]))
        # use pix_dif ratio to max_pz_up to determine gz pickup/dropoff point
        if drop_pix_dif > max_pix_up:
          drop_z = zu
          print("WARNING21 down xyz, up pix cr, drop_z:", xd, yd, zd, pru, pcu, zu, drop_z)
        else:
          drop_z = (zu - zd) * (drop_pix_dif / max_pix_dif) + zd

        if drop_z < zd or drop_z > zu:
          print("WARNING2: down xyz, up pix cr, drop_z:", xd, yd, zd, pru, pcu, zu, drop_z)

        print("drop x,y,z:", xd,yd, drop_z)
        return (xd, yd, drop_z)

  def test_arm_reach(self):
      self.wdw.set_move_mode('Absolute')
      for vz in [12.2, 3.5]:
        for reach in range(12,30,2):
          success = self.wdw.action(vx=reach, vy=0, vz=vz, vr=0, goc = self.wdw.GRIPPER_OPEN)
          print("press return for VX,VZ:", reach, vz)
          wait_for_return = input()

