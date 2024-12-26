# TableTop Analysis:
#
# assumes 3 sides of a tabletop can be seen, and deduces the 2 corners.
# From these 2 corners, Vanishing Point analysis can be done to produce
# a birds-eye-view of the tabletop.  With calibration of the robot arm,
# this BEV produces a camera-pixel to robot arm XY transformation.
#
# The BEV computations is based on:
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#
# The overall code is based on the much more complex ALSET Map Analysis, which created  
# a merged birds-eye-view map as the robot moves around the tabletop and deduces things
# like robot location.
#

# import the necessary packages
import numpy as np
from numpy import asarray
from PIL import Image
import cv2
import argparse
# import skgstat as skg
import skg
from analyze_lines import *
from imutils import paths
# from svr_state import *
import imutils
from matplotlib import pyplot as plt
from shapely.geometry import *
from util_cv_analysis import *
from util_radians import *
# from util_borders import *
import statistics 
from operator import itemgetter, attrgetter
from skimage.metrics import structural_similarity as ssim  
# from scipy      import optimize
import scipy
import scipy.misc
import scipy.cluster
import random
# from do_ratslam import *
from do_stego import *
from IPython import display
import tensorflow as tf

# from __future__ import print_function
import imageio
import binascii
import struct
import json

# based on: 
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
class AnalyzeTableTop():
    def __init__(self):
        self.tt_calib = {}
        self.tt_calib["VPL"] = []
        self.tt_calib["VPL_quality"] = []
        # All tt_calib keys:
        # "VPL_FOUND", "BEST_VPL", "BEST_VPL_QUALITY", "VPL", "VPL_quality"
        # "BEV_image"

        with open('widowx_config.json') as config_file:
          config_json = config_file.read()
        self.config = json.loads(config_json)
        self.calib_dir = self.config["calibration_directory"]
        calib_file =  self.calib_dir + "/" + self.config["calibration_file"]
        with open(calib_file) as cal_file:
          config_json = cal_file.read()
        self.calib_data = json.loads(config_json)
        self.set_best_vpl(self.calib_data["metadata"]["best_vpl"])
        self.best_vpl_num = -1

        self.border_multiplier = 2   # move_img
        self.color_quant_num_clust = 0
        self.scale_factor = 1
        self.add_border = True

        try:
          self.segmentation_method = self.calib_data["metadata"]["tabletop_segmentation_method"]
        except:
          # default: if calib data not saved yet
          self.segmentation_method = "COLOR"  
        if self.segmentation_method == "COLOR":
          self.IMG_W = 320
          self.IMG_H = 256
        elif self.segmentation_method == "STEGO":
          print("Warning: TT Analysis of STEGO needs to be re-debugged")
          self.IMG_W = 416
          self.IMG_H = 416
        self.tt_calib["IMG_W_H"] = [self.IMG_W, self.IMG_H]
   
        self.INFINITE = 100000000000000000

        self.same_point = 40
        self.lna = AnalyzeLines()
        self.stego = Stego()
        self.use_stego = True
        self.curr_stego = None
        self.frame_lines = []
        self.click_row   = None
        self.click_col   = None
        self.BEV_M = None
        self.BEV_image = None

    ######################
    def load_analysis(self, tt_analysis):
        self.tt_anal = copy.deepcopy(tt_analysis)
        self.BEV_M = self.tt_anal["BEV_M"]
        [self.IMG_W, self.IMG_H] = self.tt_calib["IMG_W_H"] 
        self.best_vpl_num = self.tt_calib["BEST_VPL_NUM"] 
        self.best_vpl = self.tt_calib["BEST_VPL"] 


    #################################
    # CREATE TABLETOP
    #################################

    # compare 2 different VPL's and determine which one has the best quality
    # returns quality of vpl1 if no vpl2 given.
    def better_vp_quality(self, vpl1, vpl2=None):
            # vpl = [left_bottom_end_point, left_end_point, 
            #        right_end_point, right_bottom_end_point]
            if vpl1 is None:
              return None, None
            print("new vpl:", vpl1)
            print("old vpl:", vpl2)
            # calculate middle line angle (slope) in radian
            angle = np.arctan2(vpl1[1][1] - vpl1[2][1], vpl1[2][0] - vpl1[2][0])
            # convert to degree, compare to 180
            middle_line_degree1 = abs(180 - abs(angle * (180 / np.pi)))
            tot_line_lens1 = (self.lna.get_line_len(vpl1[0],vpl1[1])
                            + self.lna.get_line_len(vpl1[1],vpl1[2])
                            + self.lna.get_line_len(vpl1[2],vpl1[3]))
            min_line_lens1 = min(
                              self.lna.get_line_len(vpl1[0],vpl1[1]),
                              self.lna.get_line_len(vpl1[1],vpl1[2]),
                              self.lna.get_line_len(vpl1[2],vpl1[3]))
            # middle line and left line match end points; check if perp lines
            left_rads  = self.lna.get_line_angle((vpl1[0],vpl1[1]),(vpl1[1],vpl1[2]))
            right_rads = self.lna.get_line_angle((vpl1[1],vpl1[2]),(vpl1[2],vpl1[3]))
            radian_dif1 = abs(right_rads - left_rads)
            print("right_rads, left_rads, radian_dif1",right_rads, left_rads, radian_dif1)
            # radian_dif1 = abs(right_rads - left_rads)
            print("radian_dif1",radian_dif1)
            # ARD: an important quality check: post-BEV angle (close to 90deg)
            vpl_quality1 = [tot_line_lens1, min_line_lens1, radian_dif1]
            ###############
            if vpl2 is None:
              return vpl1, vpl_quality1
            if self.is_same_vpl(vpl1, vpl2):
              print("exact same vpl:", vpl1)
              return vpl1, vpl_quality1
            # calculate middle line angle (slope) in radian
            angle = np.arctan2(vpl2[1][1] - vpl2[2][1], vpl2[2][0] - vpl2[2][0])
            # convert to degree, compare to 180
            middle_line_degree2 = abs(180 - abs(angle * (180 / np.pi)))
            tot_line_lens2 = (self.lna.get_line_len(vpl2[0],vpl2[1])
                            + self.lna.get_line_len(vpl2[1],vpl2[2])
                            + self.lna.get_line_len(vpl2[2],vpl2[3]))
            min_line_lens2 = min(
                              self.lna.get_line_len(vpl2[0],vpl2[1]),
                              self.lna.get_line_len(vpl2[1],vpl2[2]),
                              self.lna.get_line_len(vpl2[2],vpl2[3]))
            # middle line and left line match end points; check if perp lines
            left_rads  = self.lna.get_line_angle((vpl1[0],vpl1[1]),(vpl1[1],vpl1[2]))
            right_rads = self.lna.get_line_angle((vpl1[1],vpl1[2]),(vpl1[2],vpl1[3]))
            radian_dif2 = abs(right_rads - left_rads)
            print("right_rads, left_rads, radian_dif1",right_rads, left_rads, radian_dif2)
            vpl_quality2 = [tot_line_lens2, min_line_lens2, radian_dif2]

            cnt = 0
            # if middle_line_degree2 - middle_line_degree1 > 5: 
              # cnt += 1
            # within a single image, tot_line_len is the most important.
            if tot_line_lens1 > tot_line_lens2:
              cnt += 1
            # if min_line_lens1 > min_line_lens2:
              # cnt += 1
            # if radian_dif1 < radian_dif2:
              # cnt += 1
            # if cnt >= 2:
            if cnt >= 1:
              return vpl1, vpl_quality1
            else:
              return vpl2, vpl_quality2

    def same_line_endpoint(self, ln1, ln2, same_point=40):
        [l1_x1,l1_y1,l1_x2,l1_y2] = ln1[0]
        [l2_x1,l2_y1,l2_x2,l2_y2] = ln2[0]
        for endptnum in range(4):
          if endptnum < 2:
            ep1 = [l1_x1, l1_y1]
            ep1b = [l1_x2, l1_y2]
          else:
            ep1 = [l1_x2, l1_y2]
            ep1b = [l1_x1, l1_y1]
          if endptnum % 2:
            ep2 = [l2_x1, l2_y1]
            ep2b = [l2_x2, l2_y2]
          else:
            ep2 = [l2_x2, l2_y2]
            ep2b = [l2_x1, l2_y1]
          dist = np.linalg.norm(np.array(ep1) - np.array(ep2))
          if dist <= same_point:
            # print("same_point, dist:", ep1,ep2, dist)
            return True, [ep1, ep2], [ep1b, ep2b]
        return False, None, None

    def is_same_point(self, pt1, pt2, same_point=40):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if pt1[1] == -1088 or pt2[1] == -1088:
          print("pt1 pt2 dist:", pt1, pt2, dist)
        if dist < same_point:
          return True
        return False

    def is_same_vpl(self, vpl1, vpl2):
        return (self.is_same_point(vpl1[0], vpl2[0], same_point=0) and
                self.is_same_point(vpl1[1], vpl2[1], same_point=0) and
                self.is_same_point(vpl1[2], vpl2[2], same_point=0) and
                self.is_same_point(vpl1[3], vpl2[3], same_point=0))

    def get_vpl_lines(self, vpl):
        return [
                [[vpl[0][0], vpl[0][1], vpl[1][0], vpl[1][1]]],
                [[vpl[1][0], vpl[1][1], vpl[2][0], vpl[2][1]]],
                [[vpl[2][0], vpl[2][1], vpl[3][0], vpl[3][1]]]
               ]

    def is_overlapping_vpl(self, vpl1, vpl2):
        # must have same top line
        if (not self.is_same_point(vpl1[1], vpl2[1]) or
            not self.is_same_point(vpl1[2], vpl2[2])):
          return False, None

        # other two must be parallel intersecting lines
        vln1 = self.get_vpl_lines(vpl1)
        vln2 = self.get_vpl_lines(vpl2)
        if not (self.lna.is_same_line(vln1[0], vln2[0])
            and self.lna.is_same_line(vln1[2], vln2[0])):
          return False, None

        # extend the vpl lines
        [x1a,y1a,x2a,y2a] = self.extend_line(vln1[0], vln2[0])
        [x1b,y1b,x2b,y2b] = self.extend_line(vln1[2], vln2[2])

        new_vpl = [[x1a,y1a], 
                [int((vpl1[1][0]+vpl2[1][0])/2), int((vpl1[1][1]+vpl2[1][1])/2)],
                [int((vpl1[2][0]+vpl2[2][0])/2), int((vpl1[2][1]+vpl2[2][1])/2)],
                [x2b,y2b]] 
        print("extended vpl:", new_vpl)
        return True, new_vpl

  ###################################################3
    def get_primary_color_tabletop(self, image):
        NUM_CLUSTERS = self.calib_data["metadata"]["num_color_clusters"]
  
        ar = np.asarray(image)
        shape = ar.shape
        print("shape:", shape)
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
  
        print('finding clusters')
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
        print('cluster centres:\n', codes)
  
        vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
  
        index_max = scipy.argmax(counts)                    # find most frequent
        peak = codes[index_max]
        colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
        print('most frequent is %s (#%s)' % (peak, colour))
        
        # bonus: save image using only the N most common colours
        c = ar.copy()
        for i, code in enumerate(codes):
            c[scipy.r_[scipy.where(vecs==i)],:] = code
        
        calib_image = c.reshape(*shape).astype(np.uint8)
        calib_color_file = self.calib_dir + "/" + "image_empty_bycolor.png"
        if os.path.isfile(calib_color_file):
          print(calib_color_file,": already exists.")
          calib_image = cv2.imread(calib_color_file)
        else:
          imageio.imwrite(calib_color_file, calib_image)
          print('saved clustered image:', calib_color_file)
        return calib_image

    ###################################################3
    # Calibration:
    # Do a vanishing point line (VPL) check first: 
    #
    # Methods: "STEGO", "color", None
    # 3 possible methods to compute segmentation for VPL.
    # set in calibration config file:
    # 
    # But we need to transform the tabletop to be horizontal when complete.
    def calibrate_tabletop(self, img, method="STEGO"):
        print("=============")
        print("CALIBRATE_TABLETOP")
        print("=============")
        try:
          # ORIGINAL_IMAGE not stored (just cached)
          # if self.tt_calib["ORIGINAL_IMAGE"] != None:
            cv_img = self.get_calib_image(img)
            self.tt_calib["ORIGINAL_IMAGE"] = cv_img
            # BEV_image not stored
            # self.BEV_image = self.tt_calib["BEV_image"].copy()
            self.best_vpl = copy.deepcopy(self.tt_calib["BEST_VPL"])
            # best_vpl_quality not stored
            # self.best_vpl_quality = self.tt_calib["BEST_VPL_QUALITY"]
            # best_vpl_num : not stored
            # self.best_vpl_num = self.tt_calib["BEST_VPL_NUM"]
            if self.tt_calib["VPL_FOUND"]:
              print("Calibration already done")
              return True
            else:
              print("Previous calibration failed")
        except:
          # failed a test; recompute calibration
          print("computing calibration")
          cv_img = self.get_calib_image(img)
        cv2.imshow('calib image', cv_img)
        cv2.waitKey(0)
        self.frame_lines = []
        lines = self.lna.get_frame_lines(cv_img)
        # TODO: black out gripper before get_frame_lines
        # get_frame_lines combines hough_lines with FLD with combine_lines
        same_point = self.same_point
        found_vpl = False
        best_lines = {}
        best_lines["LEFT"] = []
        best_lines["RIGHT"] = []
        best_lines["CENTER"] = []
        potential_corner = []
        same_line = []
        vpl_lns = False
        if lines is not None and len(lines) > 0:
          potential_corners = []
          self.best_vpl = None
          min_center_line_len = self.IMG_H / 4
          for i1, ln1 in enumerate(lines):
            for i2, ln2 in enumerate(lines):
              if i1 == i2:
                continue
              vplnum1 = self.find_vpl_line(ln1)
              vplnum2 = self.find_vpl_line(ln2)
              if vplnum1 is not None and vplnum1 == 1:
                if vplnum2 is not None and vplnum2 == 2:
                  print("VPL#", vplnum1, "ln1:", ln1)
                  print("VPL#", vplnum2, "ln2:", ln2)
                  vpl_lns = True
              if not self.lna.is_min_line_len(ln2, min_center_line_len):
                continue
              corner, ep12_match, ep12_nonmatch = self.same_line_endpoint(ln1, ln2)
              if not corner:
                continue
              else:
                vplnum3 = None
                if vpl_lns:
                  print("corner1:", ln1, ln2)
                lc_angle = self.lna.get_line_angle2(ln1,ln2)
                dif_from_90_deg, lc_dir = dif_from_90_degrees(lc_angle)
                if abs(dif_from_90_deg) > 10:
                  if vplnum3:
                    print("LC diff 90 deg:", ln1, ln2, lc_angle, dif_from_90_deg)
                  continue
                print("LC 90 deg:", ln1, ln2, lc_angle, dif_from_90_deg, lc_dir)
                for i3, ln3 in enumerate(lines):
                  if i3 == i1 or i3 == i2:
                    continue
                  corner, ep23_match, ep23_nonmatch = self.same_line_endpoint(ln2, ln3)
                  if not corner:
                    continue
                  else:
                    # vplnum = self.find_vpl_line(ln1)
                    # if vplnum is not None:
                    #     print("VPL#", vplnum, "ln1:", ln1)
                    # vplnum = self.find_vpl_line(ln2)
                    # if vplnum is not None:
                    #     print("VPL#", vplnum, "ln2:", ln2)
                    if vpl_lns:
                      vplnum3 = self.find_vpl_line(ln3)
                      if vplnum3 is not None and vplnum3==3:
                        print("VPL#", vplnum3, "ln3:", ln3)
                        print("ALL VPLs FOUND")
                    print("corner2:", ln2, ln3)
                    ep2a_match = ep12_match[1]
                    ep2b_match = ep23_match[0]
                    # ensure diff endpoints of ln2 are matched
                    # to ln1 and ln3
                    if self.is_same_point(ep2a_match, ep2b_match):
                      print("same endp:", ep2a_match, ep2b_match)
                      continue

                    cr_angle = self.lna.get_line_angle2(ln2,ln3)
                    dif_from_90_deg2, cr_dir = dif_from_90_degrees(cr_angle)
                    if abs(dif_from_90_deg2) > 11:
                      if abs(dif_from_90_deg2) > 45:
                        print("CR diff 90 deg:", ln2, ln3, cr_angle, dif_from_90_deg2, cr_dir)
                        # if not(vplnum3 is not None and vplnum3==3):
                        continue
                    print("CR 90 deg:", ln2, ln3, cr_angle, dif_from_90_deg2, cr_dir)

                  vp1 = copy.deepcopy(ep12_nonmatch[0])  # LEFT endpoint
                  vp1 = [round(vp1[0]), round(vp1[1])]
                  # vp2 = copy.deepcopy(ep12_match[1])  # CENTER endpoint
                  vp2 = self.lna.line_intersection(ln1, ln2, infinite_lines=True)
                  vp2 = [round(vp2[0]), round(vp2[1])]
                  # vp3 = copy.deepcopy(ep23_match[0])  # other CENTER endpoint
                  vp3 = self.lna.line_intersection(ln2, ln3, infinite_lines=True)
                  vp3 = [round(vp3[0]), round(vp3[1])]
                  vp4 = copy.deepcopy(ep23_nonmatch[1])  # RIGHT endpoint
                  vp4 = [round(vp4[0]), round(vp4[1])]
                  # vp4 = copy.deepcopy(ep23_match[1])  # RIGHT endpoint
                  # should take intersect point of lines
                  new_vanishing_point = [vp1, vp2, vp3, vp4]
                  if (self.is_same_point(vp1,vp2) or 
                      self.is_same_point(vp2,vp3) or
                      self.is_same_point(vp3,vp4) or
                      self.is_same_point(vp1,vp4)):
                      # if self.is_same_point(vp3,vp4):
                      #   print("fake not same point", new_vanishing_point)
                      # else:
                        print("Duplicated VP point:", new_vanishing_point)
                        continue # reject bad VP

                  # check for convex vpl
                  lc_angle = self.lna.get_line_angle([vp1,vp2], [vp2,vp3])
                  dif_from_90_deg_lc, lc_dir = dif_from_90_degrees(lc_angle)
                  print("dif_from_90_deg_lc, lc_dir:", dif_from_90_deg_lc, lc_dir)
                  cr_angle = self.lna.get_line_angle([vp2,vp3], [vp3,vp3])
                  dif_from_90_deg_cr, cr_dir = dif_from_90_degrees(lc_angle)
                  print("dif_from_90_deg_cr, cr_dir:", dif_from_90_deg_cr, cr_dir)
                  if (lc_dir != cr_dir):
                    # right angles need to form convex vpl
                    print("vpl not convex:", new_vanishing_point)
                    continue

                  # confirm end-points near sides of image
                  if self.add_border:
                      w = cv_img.shape[0] - same_point
                      h = cv_img.shape[1] - same_point
                      if ((vp3[0] > w and vp4[0] > w) or
                          (vp2[0] > w and vp3[0] > w) or
                          (vp1[0] > w and vp2[0] > w) or
                          (vp3[1] > h and vp4[1] > h) or
                          (vp2[1] > h and vp3[1] > h) or
                          (vp1[1] > h and vp2[1] > h)):
                        print("vpl cannot be at bordered edge")
                        continue
                  else:
                      w = self.IMG_W
                      h = self.IMG_H
                  found_same_vpl = False
                  for v, vpl in enumerate(self.tt_calib["VPL"]):
                    # effectively a very similar vpl has already 
                    # been found.  Only store "the best" one
                    if((self.is_same_point(vp1,vpl[0]) and 
                        self.is_same_point(vp2,vpl[1]) and
                        self.is_same_point(vp3,vpl[2]) and
                        self.is_same_point(vp4,vpl[3])) or
                       (self.is_same_point(vp1,vpl[3]) and 
                        self.is_same_point(vp2,vpl[2]) and
                        self.is_same_point(vp3,vpl[1]) and
                        self.is_same_point(vp4,vpl[0]))):
                      best_new_vpl, best_new_vpl_quality = self.better_vp_quality(new_vanishing_point, vpl)
                      self.tt_calib["VPL"][v] = copy.deepcopy(best_new_vpl)
                      self.tt_calib["VPL_quality"][v] = copy.deepcopy(best_new_vpl_quality)
                      # running tally
                      self.best_vpl, self.best_vpl_quality = self.better_vp_quality(best_new_vpl, self.best_vpl)
                      print("same vpl already found:", vpl)
                      found_same_vpl = True
                      break
                    is_overlapping, combined_vpl = self.is_overlapping_vpl(new_vanishing_point, vpl)
                    if is_overlapping:
                      self.tt_calib["VPL"][v] = copy.deepcopy(combined_vpl)
                      # running tally
                      self.best_vpl, self.best_vpl_quality = self.better_vp_quality(combined_vpl, self.best_vpl)
                      found_same_vpl = True
                      break
                  
                  if not found_same_vpl:
                    self.tt_calib["VPL"].append([vp1, vp2, vp3, vp4])
                    vp, qual = self.better_vp_quality(new_vanishing_point)
                    # quality by longest total line len
                    self.tt_calib["VPL_quality"].append(qual)
                    print("FOUND NEW VP LCR:", new_vanishing_point)

                    # running tally of the "best" vpl
                    self.best_vpl, self.best_vpl_quality = self.better_vp_quality(new_vanishing_point, self.best_vpl)
                    if self.is_same_vpl(self.best_vpl, new_vanishing_point):
                      self.best_vpl_num = len(self.tt_calib["VPL"]) - 1

          ###################### 
          # vanishing point found
          ###################### 
          if self.best_vpl is not None:
              # qual = self.better_vp_quality(new_vanishing_point)
              print("VANISHING POINT FOUND:", self.best_vpl)

#             ##############################
#             # The following are stats that you can derive from the VPL.
#             # No need to store.
#             angle1 = np.arctan2(vpl[1][1] - vpl[2][1], vpl[1][0] - vpl[2][0])
#             angle2 = np.arctan2(vpl[0][1] - vpl[1][1], vpl[0][0] - vpl[1][0])
#             angle3 = np.arctan2(vpl[2][1] - vpl[3][1], vpl[2][0] - vpl[3][0])
#             corner_angle_left = (angle2 - angle1)
#             corner_angle_right = (angle2 - angle3)
#             desired_angle = (corner_angle_left + corner_angle_right)/2
#             delta_angle = desired_angle - RADIAN_RIGHT_ANGLE
#             bev_brdr = self.lna.vpl_border(desired_angle)
#             if bev_brdr is None:
#               bev_brdr = 0
#             bev_side_len = 2*bev_brdr + self.IMG_H
#             calib_delta_angle = delta_angle
#             calib_vpl_corner_angle_left = corner_angle_left
#             calib_vpl_corner_angle_right = corner_angle_right
#             self.calib_vpl_corners = [[vpl[1][0], vpl[1][1]], [vpl[2][0], vpl[2][1]]]
#             self.calib_vpl_corner_lines = [[[[vpl[0][0], vpl[0][1], vpl[1][0], vpl[1][1]]],
#                                            [[vpl[1][0], vpl[1][1], vpl[2][0], vpl[2][1]]]], 
#                                            [[vpl[2][0], vpl[2][1], vpl[3][0], vpl[3][1]]]], 
#             ##############################

              print("FOUND VPLs:")
              for num, found_vpl in enumerate(self.tt_calib["VPL"]):
                print(num, ":", self.tt_calib["VPL"][num])
              # display VPL
              num_vpls = len(self.tt_calib["VPL"]) + 1
              for num in range(num_vpls):
                self.tt_calib["VPL_FOUND"] = True
                if num == 0:
                  self.display_vpl(cv_img, self.best_vpl)
                else:
                  self.display_vpl(cv_img, self.tt_calib["VPL"][num-1])
                cv2.waitKey(0)
                print("is best VPL on display? 1=Y, 2=N")
                YN = input()
                try:
                  if int(YN) == 1:
                    if num == 0:
                      break
                    else:
                      self.best_vpl_num = num - 1
                      self.best_vpl = copy.deepcopy(self.tt_calib["VPL"][num-1])
                      # leave self.best_vpl_quality the same?
                      # todo: store other vpls?
                    break
                  elif int(YN) == 2:
                    print("displaying", num, ":", round(self.tt_calib["VPL_quality"][num][0]))
                    continue
                except:
                  continue
              # TODO: allow manual clicking of BEV

              self.BEV_image = self.compute_birds_eye_view(cv_img,self.best_vpl)
              self.tt_calib["BEV_image"] = self.BEV_image.copy()
              self.tt_calib["VPL_FOUND"] = True
              self.tt_calib["BEST_VPL"] = copy.deepcopy(self.best_vpl)
              self.tt_calib["BEST_VPL_QUALITY"] = self.best_vpl_quality
              self.tt_calib["BEST_VPL_NUM"] = self.best_vpl_num 
          else:
              print("NO VANISHING POINT FOUND")

        cv2.destroyAllWindows()
        return found_vpl

    def display_vpl(self, image, vpl):
        # img2 = self.stego.convert_plot_to_cv(image)
        print("display_vpl:", vpl)
        img2 = image.copy()
        img2 = cv2.line(img2, (vpl[0][0],vpl[0][1]), (vpl[1][0],vpl[1][1]), (255,0,0), 3, cv2.LINE_AA)
        img2 = cv2.line(img2, (vpl[1][0],vpl[1][1]), (vpl[2][0],vpl[2][1]), (255,0,0), 3, cv2.LINE_AA)
        img2 = cv2.line(img2, (vpl[2][0], vpl[2][1]), (vpl[3][0],vpl[3][1]), (255,0,0), 3, cv2.LINE_AA)
        cv2.imshow('VPL', img2)

    def show_and_eval_vpl(self, image, vpl):
        self.display_vpl(image, self.best_vpl)
        while True:
          print("Is the tabletop properly highlighted?  1=Y, 2=N")
          YN = input()
          try:
            if int(YN) == 1:
              return True
            elif int(YN) == 2:
              return False
          except:
            continue

    # points clicked on the image
    def click_event(self, event, x, y, flags, params):
        # checking for left or right mouse clicks
        if (event == cv2.EVENT_LBUTTONDOWN or
           event==cv2.EVENT_RBUTTONDOWN):
          # displaying the coordinates
          # on the Shell
          print(x, ' ', y)
          self.click_row = x
          self.click_col = y 

    def manual_eval_vpls(self):
        # check if correct about best VPL quality:
        cv_image = self.tt_calib["ORIGINAL_IMAGE"] 
        if self.show_and_eval_vpl(cv_image, self.best_vpl):
          return self.tt_calib

        # iterate through found VPLs and check if correct
        for num, vpl in enumerate(self.tt_calib["VPL"]):
          if num == self.best_vpl_num:
            continue
          self.display_vpl(cv_image, vpl)
          if self.show_and_eval_vpl(cv_image, vpl):
            self.best_vpl_num = num
            self.best_vpl = copy.deepcopy(vpl)
            return self.tt_calib
        
        # no satisfactory vpl; manually click to define vpl
        # alternative: vpl computation is non-deterministic
        # setting mouse handler for the image
        # and calling the click_event() function
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.setMouseCallback('VPL', self.click_event)
        while True:
          self.click_row = None
          self.click_col = None
          click_image = copy.deepcopy(cv_image)
          cv2.imshow("VPL",click_image)
          print("click bottom left end point of VPL:")
          new_vpl = [[self.click_row, self.click_col]]
          cv2.waitKey(0)
          center = (int(self.click_row),int(self.click_col))
          radius = 10
          cv2.circle(click_image,center,radius,(255,255,0),2)
          # cv2.putText(click_image, "* X:" + str(x) + ", Y:" + str(y),
          #           (x,y), font, 1,  (255, 255, 0), 2) 
          cv2.imshow("VPL",click_image)
          print("click top left corner of VPL:")
          cv2.waitKey(0)
          new_vpl.append([self.click_row, self.click_col])
          center = (int(self.click_row),int(self.click_col))
          cv2.circle(click_image,center,radius,(255,255,0),2)
          cv2.imshow("VPL",click_image)
          print("click top right corner of VPL:")
          cv2.waitKey(0)
          new_vpl.append([round(self.click_row), round(self.click_col)])
          center = (int(self.click_row),int(self.click_col))
          cv2.circle(click_image,center,radius,(255,255,0),2)
          cv2.imshow("VPL",click_image)
          print("click bottom right end point of VPL:")
          cv2.waitKey(0)
          if self.show_and_eval_vpl(cv_image, new_vpl):
            self.tt_calib["VPL"].append(copy.deepcopy(new_vpl))
            vp, qual = self.better_vp_quality(new_vpl)
            self.tt_calib["VPL_quality"].append(qual)
            self.best_vpl = copy.deepcopy(new_vpl)
            self.best_vpl_num = len(self.tt_calib["VPL"]) - 1
            return self.tt_calib

    # For MC:
    #  +-/
    # D|/
    #  V A = angle
    # looking at corner, don't quite reach 90deg.
    # 
    #
    # For MSLC
    # actual intersect
    # +     A = corner/actual intersect angle
    # +-------Crnr
    # |   D
    # |
    # Robot Center
    def get_dist_at_90deg(self, A,d):
      return cos(abs(90-A))*d

    ##########################################################3
    #  TWO_CORNER VPL computations
    ##########################################################3
    ######################
    # Transforms
    ######################

    def compute_M_x_y(self, M, x, y):
        new_x = round(M[0][0]*x + M[0][1]*y + M[0][2])
        new_y = round(M[1][0]*x + M[1][1]*y + M[1][2])
        return new_x, new_y

    def compute_rotated_image_line2(self, line, angle):
        # convert to BEV, rotate by BEV_rot, convert back to x/y
        BEV_x1, BEV_y1 = self.BEV_x_y(line[0][0],line[0][1])
        BEV_x2, BEV_y2 = self.BEV_x_y(line[0][2],line[0][3])
        if BEV_x1 is None or BEV_x2 is None:
          [BEV_x1, BEV_y1] = [line[0][0],line[0][1]]
          [BEV_x2, BEV_y2] = [line[0][2],line[0][3]]
        BEV_line = [[BEV_x1, BEV_y1, BEV_x2, BEV_y2]]
        BEV_rot_line = self.compute_rotated_line(BEV_line, angle)
        img_x1, img_y1 = self.BEV_to_image_x_y(BEV_rot_line[0][0],BEV_rot_line[0][1])
        img_x2, img_y2 = self.BEV_to_image_x_y(BEV_rot_line[0][2],BEV_rot_line[0][3])
        rot_line = [[img_x1, img_y1, img_x2, img_y2]]
        return rot_line

    # generalized version of compute_rotated_line
    def compute_rotated_image_line(self, line, angle, center=None):
        rot_M = self.compute_rotate_M(angle, center)
        if rot_M is None:
          return None
        new_x1,new_y1 = self.compute_M_x_y(rot_M, line[0][0], line[0][1])
        new_x2,new_y2 = self.compute_M_x_y(rot_M, line[0][2], line[0][3])
        new_line = [[new_x1,new_y1,new_x2,new_y2]]
        return new_line

    def compute_rotated_line(self, line, angle):
        rot_M = self.compute_rotate_M(angle)
        if rot_M is None:
          return None
        new_x1,new_y1 = self.compute_M_x_y(rot_M, line[0][0], line[0][1])
        new_x2,new_y2 = self.compute_M_x_y(rot_M, line[0][2], line[0][3])
        new_line = [[new_x1,new_y1,new_x2,new_y2]]
        return new_line

    def get_best_vpl(self):
        # algorithm was designed for moving robot. 
        # could iterate and return the one with the greatest quality / tot ln len
        # return self.vanishing_point_lines["VPL"]
        return self.best_vpl

    def set_best_vpl(self, bestVPL):
        self.best_vpl = copy.deepcopy(bestVPL)

    def compute_BEV_M(self):
        vpl = self.get_best_vpl()
        print("VPL:", vpl)
        self.compute_birds_eye_view(None, vpl)
        return True

    def xform_to_BEV_line(self, line):
        x1,y1,x2,y2 = line[0]
        BEV_x1, BEV_y1 = self.BEV_x_y(x1,y1)
        BEV_x2, BEV_y2 = self.BEV_x_y(x2,y2)
        return [[BEV_x1, BEV_y1, BEV_x2, BEV_y2]]

    def BEV_to_image_x_y(self, BEV_x, BEV_y):
        if self.BEV_M is None:
          if not self.compute_BEV_M():
            return BEV_x, BEV_y   # prev-BEV just uses image
        # print("BEV_M", self.BEV_M)
        y = (BEV_y - self.BEV_M[1][0]*((BEV_x - self.BEV_M[0][2]) / self.BEV_M[0][0]) - self.BEV_M[1][2]) / self.BEV_M[1][1] / (1+ (self.BEV_M[1][0]*self.BEV_M[0][1])/self.BEV_M[1][1])
        x = (BEV_x - self.BEV_M[0][1]*y - self.BEV_M[0][2]) / self.BEV_M[0][0]
        return round(x), round(y)

    # Once a vanishing point is known, compute xformed X,Y locations of a pixel 
    def BEV_x_y(self, x, y):
        if self.BEV_M is None:
          if not self.compute_BEV_M():
            return x, y  # if BEV is not yet known, use image x y
            # return None, None
        # M[0][0], M[1][1] sign is for reflection (if negative)
        # M[0][0], M[1][1] is for scaling
        # M[0][2], M[2][2] is for shifting
        # M[0][1], M[2][0] is for shearing
        # M[0][1] (cos), M[0][1] (-sin), M[1][0] (sin), M[1][1] (cos) is for rotation of angle
        # print("BEV_M:", self.BEV_M)
        new_x = round(self.BEV_M[0][0]*x + self.BEV_M[0][1]*y + self.BEV_M[0][2])
        new_y = round(self.BEV_M[1][0]*x + self.BEV_M[1][1]*y + self.BEV_M[1][2])
        return new_x, new_y

    ######################################################################3333
    # BEV: Bird's Eye View of Tabletop
    ######################################################################3333
    # not fully generalized computation of BEV (birds eye view)
    # Debugged version using fixed camera that doesn't see full table-top.
    # Fixed Camera is above outside corner of tabletop at 45 deg angle.
    def compute_birds_eye_view(self, image, vpl):
        # original X of top vanishing point line
        # image above the top vanishing point line will be cropped off in BEV
        if vpl is None:
          return None
        #
        print("vpl:", vpl)
        top_y = min(vpl[1][1], vpl[2][1])
        orig_w = (self.IMG_W-1) # zero-based
        orig_h = (self.IMG_H-1) # zero-based

        # orig vanishing point lines on side
        left_line = [[vpl[0][0], vpl[0][1], vpl[1][0], vpl[1][1]]]
        right_line = [[vpl[2][0], vpl[2][1], vpl[3][0], vpl[3][1]]]


        ##################
        # compute a new bottom_line parallel to the top vpl line such that
        # the Parallel Bottom Line (pbl) touches the bottom line.
        ##################

        # need to find intercept at x = 0 or x = IMG_W, 
        # then adjust delta_y based on that.
        if vpl[1][1] < vpl[2][1]:
          delta_y = (self.IMG_H-1) - vpl[1][1] 
          axis = [[0, vpl[1][1]+delta_y, 0, vpl[2][1]+delta_y]]
        else:
          delta_y = (self.IMG_H-1) - vpl[2][1]
          axis = [[(self.IMG_W-1), vpl[1][1]+delta_y, (self.IMG_W-1), vpl[2][1]+delta_y]]
        pbl = [[vpl[1][0], vpl[1][1]+delta_y, vpl[2][0], vpl[2][1]+delta_y]]
        pbl_intersect = self.lna.line_intersection(axis, pbl, infinite_lines=True)
        print("PBL_Intersect", pbl_intersect, delta_y)
        delta_y += round((self.IMG_W-1)-pbl_intersect[1])
        
        pbl = [[vpl[1][0], vpl[1][1]+delta_y, vpl[2][0], vpl[2][1]+delta_y]]

        left_pbl_intersect = self.lna.line_intersection(left_line, pbl, infinite_lines=True)
        right_pbl_intersect = self.lna.line_intersection(right_line, pbl, infinite_lines=True)
        print("left_pbl_intersect, right_pbl_intersect:", left_pbl_intersect, right_pbl_intersect)

        ######
        # Find 4 corners of VPL if completed to full size.
        ######
        top_ln_len = np.linalg.norm(np.array(vpl[1]) - np.array(vpl[2]))
        left_ln_len = np.linalg.norm(np.array(vpl[1]) - np.array(left_pbl_intersect))
        right_ln_len = np.linalg.norm(np.array(vpl[2]) - np.array(right_pbl_intersect))
        bottom_ln_len = np.linalg.norm(np.array(left_pbl_intersect) - np.array(right_pbl_intersect))
        print("LLL, RLL:", left_ln_len, right_ln_len)

        left_pbl_intersect[0] = round(left_pbl_intersect[0])
        left_pbl_intersect[1] = round(left_pbl_intersect[1])
        right_pbl_intersect[0] = round(right_pbl_intersect[0])
        right_pbl_intersect[1] = round(right_pbl_intersect[1])

        # Note: needs to be generalized...
        bbuf = lbuf = rbuf = 0
        if left_pbl_intersect[0] < 0:
          lbuf = abs(left_pbl_intersect[0])
        if left_pbl_intersect[1] > self.IMG_H:
          bbuf = left_pbl_intersect[1] - self.IMG_H
        if right_pbl_intersect[0] > self.IMG_W:
          rbuf = right_pbl_intersect[0] - self.IMG_W
        if right_pbl_intersect[1] > self.IMG_H:
          rbbuf = right_pbl_intersect[0] - self.IMG_H
          bbuf = max(bbuf, rbbuf)
        print("r,b buf", rbuf, bbuf)
        if image is not None:
          BEV_perspective = cv2.copyMakeBorder(image,
	     top=0, bottom=bbuf, left=lbuf, right=rbuf,
                                 borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
          # cv2.imshow("BEV perspective", BEV_perspective )
          # cv2.waitKey(0)

        new_h = round(max(top_ln_len, bottom_ln_len))
        new_w = round(max(left_ln_len, right_ln_len))

        if left_pbl_intersect[0] >= 0:
          src = np.float32([left_pbl_intersect, right_pbl_intersect, vpl[1], vpl[2]])
        else:
          print("generalized BEV not tested")
          l_vpl = [vpl[1][0] + lbuf, vpl[1][1]]
          r_vpl = [vpl[2][0] + lbuf, vpl[2][1]]
          src = np.float32([left_pbl_intersect, right_pbl_intersect, l_vpl, r_vpl])

        dst = np.float32([[0, new_h], [new_w, new_h], [0, 0], [new_w, 0]])
        print("dst:", dst)

        # Image warping
        self.BEV_M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        if image is not None:
          birds_eye_view = cv2.warpPerspective(BEV_perspective, self.BEV_M, (new_w, new_h))

          cv2.imshow("birds_eye_view", birds_eye_view )
          cv2.waitKey(0)
        else:
          birds_eye_view = None
        self.tt_calib["BEV_image"] = self.BEV_image
        self.tt_calib["BEV_M"] = self.BEV_M
        self.tt_calib["PBL"] = copy.deepcopy(pbl)
        self.tt_calib["BEV_src"] = copy.deepcopy(src)
        self.tt_calib["BEV_line_lengths"] = [top_ln_len, left_ln_len, bottom_ln_len, right_ln_len]
        return birds_eye_view

    # get BEV using stored ratios
    # good for pre-calibration approximations for moving robots
    def get_approx_birds_eye_view(self, image): 
        w = image.shape[0]
        h = image.shape[1]
        ######
        # To change perspective, go to config file and change ratios
        # TODO: make the expanded width much wider and length much longer
        ######
        y = int(self.BIRDS_EYE_RATIO_H*h )
        new_w = int(self.BIRDS_EYE_RATIO_W*w)
        ######
        bdr   = int((new_w-w)/2)
        BEV_canvas = cv2.copyMakeBorder(image,
                             top=0, bottom=0, left=bdr, right=bdr,
                             borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        src = np.float32([[bdr, h], [w+bdr, h], [bdr, 0], [w+bdr, 0]])
        dst = np.float32([[bdr, h+y], [w+bdr, h+y], [0, 0], [new_w, 0]])
        self.approx_BEV_M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        # Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
        # Image warping
        approx_BEV_image = cv2.warpPerspective(BEV_canvas, self.BEV_M, (new_w, h+y)) 

        # cv2.imshow("b4", image )
        # cv2.imshow("birds_eye_view", BEV_image )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # not used for non-moving robots
        # self.tt_calib["Approx_BEV_image"] = approx_BEV_image
        # self.tt_calib["Approx_BEV_M"] = self.approx_BEV_M
        return approx_BEV_image

    def get_calib_image(self, img):
        if self.segmentation_method == "STEGO":
          self.stego = Stego()
          stego_name = "stego_empty.jpg"
          stego_path = "/tmp/" + stego_name
          # resized_image = tf.image.resize_with_pad(img, target_width=416, target_height=416)
          # resized_image = tf.cast(resized_image, tf.uint8)

          resized_shape = (416,416)
          # with open(img_path, 'r') as f:
          #   curr_img = cv2.imread(img_path)
          force_stego_run = True
          if force_stego_run:
            curr_stego = self.stego.run(img, add_border=True)
            cv2.imwrite(stego_path, curr_stego)
          else:
            try:
              with open(stego_path, 'r') as f:
                curr_stego = cv2.imread(stego_path)
            except:
              cv2.imwrite(stego_path, curr_stego)
          plot_stego = self.stego.convert_cv_to_plot(curr_stego)
          nice_stego = self.stego.convert_to_high_contrast_stego_img(plot_stego)
          nice_stego = self.stego.convert_plot_to_cv(nice_stego)
          if (self.scale_factor != 1):
            # resized_shape = (self.IMG_W, self.IMG_H)
            # resized_shape = (self.IMG_W, self.IMG_H)
            resized_shape = (416,416)
            plot_stego = cv2.resize(plot_stego, resized_shape)
          resized_shape = (self.IMG_W, self.IMG_H)
          nice_stego = cv2.resize(nice_stego, resized_shape)
          print("img")
          display.Image(img)
          cv_img = self.stego.convert_plot_to_cv(img)
          # cv2.imshow('orig img', img)
          # cv2.imshow('resized img', resized_img)
          cv2.waitKey(0)
          print("nice stego")
          im = self.stego.convert_plot_to_cv(nice_stego)
          if True:
            img2 = nice_stego.copy()
          else:
            img2 = self.stego.convert_plot_to_cv(img)
            if self.add_border:
              img2 = self.stego.pad_to_resize(img2, (416, 416))
            else:
              img2 = cv2.resize(img2, (416, 416))
        elif self.segmentation_method == "COLOR":
          img2 = cv2.imread(img)
          img2 = self.get_primary_color_tabletop(img2)
        elif self.segmentation_method is None:
          img2 = cv2.imread(img)
        else:
          print("Calibrate_tabletop: bad method", self.segmentation_method)
        return img2

    # For Debugging: returns vpl line # for specified known vpl 
    # if line matches.
    def find_vpl_line(self, ln):
      # VPL to search for:
      # vpl = [ [5, 118],[170, 15],[95,56],[203, 43] ]
      # vpl = [ [93,247],[5, 118],[171,22],[203, 43] ]
      # vpl = [[97, 253], [14, 128], [146, 5], [206, 38]]
      # vpl = [[97, 253], [14, 128], [146, 5], [206, 38]]
      # VPL: [5, 118] [170, 15] [95,56] [203, 43]
      vpl = [[88, 244], [0, 110], [169, 9], [316, 109]]

      ep1 = False
   
      if (((ln[0][0] > vpl[0][0] - 40 and ln[0][0] < vpl[0][0] + 40) 
          and (ln[0][1] > vpl[0][1] - 40 and ln[0][1] < vpl[0][1] + 40))
          and ((ln[0][2] > vpl[1][0] - 40 and ln[0][2] < vpl[1][0] + 40) 
          and(ln[0][3] > vpl[1][1] - 40 and ln[0][3] < vpl[1][1] + 40))):
          return 1
      if (((ln[0][0] > vpl[1][0] - 40 and ln[0][0] < vpl[1][0] + 40) 
          and (ln[0][1] > vpl[1][1] - 40 and ln[0][1] < vpl[1][1] + 40))
          and ((ln[0][2] > vpl[2][0] - 40 and ln[0][2] < vpl[2][0] + 40) 
          and(ln[0][3] > vpl[2][1] - 40 and ln[0][3] < vpl[2][1] + 40))):
          return 2
      if (((ln[0][0] > vpl[2][0] - 40 and ln[0][0] < vpl[2][0] + 40) 
          and (ln[0][1] > vpl[2][1] - 40 and ln[0][1] < vpl[2][1] + 40))
          and ((ln[0][2] > vpl[3][0] - 40 and ln[0][2] < vpl[3][0] + 40) 
          and(ln[0][3] > vpl[3][1] - 40 and ln[0][3] < vpl[3][1] + 40))):
          return 3
      return None
    
    
