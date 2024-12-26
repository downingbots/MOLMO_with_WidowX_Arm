from util_radians import *
# from util_borders import *
# from svr_config import *
from util_cv_analysis import *
import scipy.stats as stats
# from svr_state import *
# from do_arm_nav import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
import copy
from math import sin, cos, pi, sqrt, atan2
# from util_dataset import *
from PIL import Image, ImageChops 
import imutils
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.cluster.hierarchy import ward
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn import metrics, linear_model
import matplotlib.image as mpimg
from sortedcontainers import SortedList, SortedSet, SortedDict

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

########################
# Map Line Analysis
#
# On a tabletop and roads, edges are typically well deliniated and visible.
# For the initial tabletop apps, the line analysis has proven more reliable
# than the keypoint analysis.
#
# Note: using stego instead of color_quant_num_clust 
#################################
class AnalyzeLines():

    #########################
    # Map Line Analysis 
    #########################
    def evaluate_lines_in_image(self, image):
      border = None
      linesP, imglinesp = self.cvu.get_lines(image)
      # print("border:", border)
      max_num_lines = 10
      lines = []
      line = {}
      for i, [[l0,l1,l2,l3]] in enumerate(linesP):
        # make sure it's not just a line along the border
        if not(border is None or line_in_border(border, (l0,l1), (l2,l3))):
          print("not line in border")
          continue
        line["line"] = [l0,l1,l2,l3]
        dx = l0 - l2
        dy = l1 - l3
        line["len"] = np.sqrt(dx**2 + dy**2)
        line["slope"] = rad_arctan2(dx, dy)
        line["intersect"] = []
        # limit to top X longest lines
        min_len = self.INFINITE
        min_len_num = self.INFINITE
        if len(lines) < max_num_lines:
          lines.append(line.copy())
        else:
          for ln_num, ln in enumerate(lines):
            if min_len > ln["len"]:
              min_len_num = ln_num
              min_len = ln["len"]
          if min_len < line["len"]:
            lines[min_len_num] = line.copy()

      for i,i_line in enumerate(lines):
        [l0,l1,l2,l3] = i_line["line"] 
        i_ls = LineString([(l0, l1), (l2, l3)])
        for j,j_line in enumerate(lines):
          if i >= j:
            continue
          # matching intersections within borders can determine proper angle as the points will line up.
          # intersections will have same distance to robot location.
          [l0,l1,l2,l3] = j_line["line"] 
          j_ls = LineString([(l0, l1), (l2, l3)])
          intersect = i_ls.intersection(j_ls)
          if (not intersect.is_empty and intersect.geom_type.startswith('Point') and (border is None or 
              point_in_border(border, [round(intersect.coords[0][0]), round(intersect.coords[0][1])]))):
            print("intersect:", intersect, i, i_ls, j, j_ls)
            i_line["intersect"].append([i,j, [round(intersect.coords[0][0]), round(intersect.coords[0][1])]])
          if not intersect.is_empty and intersect.geom_type.startswith('Line'):
            # combine the lines into one
            print("Combine two lines into one:", i_ls, j_ls, intersect)

      print("lines:", lines)
      return lines

    def get_line_len(self, pt1, pt2):
        x_dif = pt2[0] - pt1[0]
        y_dif = pt2[1] - pt1[1]
        line_len = sqrt(abs(x_dif*x_dif + y_dif*y_dif))
        return line_len

    def get_line_len2(self, line):
        return self.get_line_len([line[0][0], line[0][1]], [line[0][2], line[0][3]])

    # line format: ((x1 y1)(x2 y2))
    def get_line_angle(self, line0, line1):
        # middle line and left line match end points; check if perp lines
        # angle = np.arctan2(line0[1] - line1[1], line0[0] - line1[0])
        # angle = np.arctan2(line0[1], line0[0]) - np.arctan2(line1[1], line1[0])
        angle0 = np.arctan2(line0[0][1] - line0[1][1], line0[0][0] - line0[1][0])
        angle1 = np.arctan2(line1[0][1] - line1[1][1], line1[0][0] - line1[1][0])
        # angle = angle0 - angle1
        angle = rad_dif(angle0, angle1)

        print("line0, line1, angle:", line0, line1, angle0, angle1, angle)
        # if angle < 0:
        #   angle += 2 * np.pi
        return angle

    # line format: [[x1,y1,x2,y2]]
    def get_line_angle2(self, line0, line1):
        # middle line and left line match end points; check if perp lines
        # angle = np.arctan2(line0[1] - line1[1], line0[0] - line1[0])
        # angle = np.arctan2(line0[1], line0[0]) - np.arctan2(line1[1], line1[0])
        angle0 = np.arctan2(line0[0][1] - line0[0][3], line0[0][0] - line0[0][2])
        angle1 = np.arctan2(line1[0][1] - line1[0][3], line1[0][0] - line1[0][2])
        # angle = angle0 - angle1
        angle = rad_dif(angle0, angle1)

        print("line0, line1, angle:", line0, line1, angle0, angle1, angle)
        # if angle < 0:
        #   angle += 2 * np.pi
        return angle

    # vpl angles and the extended bottom image line (IMG_H).
    def vpl_border(self, rads):
        # using basic trig, use the known angle and known IMG_H to compute
        # the left corner (which is the negative distance from 0 axis).
        delta_from_right_angle = RADIAN_RIGHT_ANGLE - rads
        length = self.IMG_H / math.sin(delta_from_right_angle)
        border = abs(round(length * math.cos(delta_from_right_angle)))
        return border


    ################
    # line analysis for tracking objects, rectangular boundaries, tabletops
    ################
    def line_intersection(self, line1, line2, infinite_lines=False, whole_pix=False):
        xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
        ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3])
    
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
    
        div = det(xdiff, ydiff)
        if div == 0:
           return None
           # raise Exception('lines do not intersect')
    
        # d = (det(line1[0][0:1], line1[0][2:3]), det(line2[0][0:1],line2[0][2:3]))
        d = (det(line1[0][0:2], line1[0][2:4]), det(line2[0][0:2],line2[0][2:4]))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        # print("intersection: ", line1, line2, [x,y])
        if infinite_lines:
          if whole_pix:
            return [round(x), round(y)]
          return [x, y]
        if ( x >= max(line1[0][0], line1[0][2]) + 2
          or x <= min(line1[0][0], line1[0][2]) - 2
          or x >= max(line2[0][0], line2[0][2]) + 2
          or x <= min(line2[0][0], line2[0][2]) - 2
          or y <= min(line1[0][1], line1[0][3]) - 2
          or y >= max(line1[0][1], line1[0][3]) + 2
          or y <= min(line2[0][1], line2[0][3]) - 2
          or y >= max(line2[0][1], line2[0][3]) + 2):
          # intersection point outside of line segments' range
#          if x >= max(line1[0][0], line1[0][2]) + 2: print(1)
#          if x <= min(line1[0][0], line1[0][2]) - 2: print(2)
#          if x >= max(line2[0][0], line2[0][2]) + 2: print(4)
#          if x <= min(line2[0][0], line2[0][2]) - 2: print(5)
#          if y <= min(line1[0][1], line1[0][3]) - 2: print(6)
#          if y >= max(line1[0][1], line1[0][3]) + 2: print(7)
#          if y <= min(line2[0][1], line2[0][3]) - 2: print(8)
#          if y >= max(line2[0][1], line2[0][3]) + 2: print(9)
#          print("outside of range:", x, y)
          return None
           
        xdiff = (line1[0][0] - line1[0][2], line2[0][0] - line2[0][2])
        ydiff = (line1[0][1] - line1[0][3], line2[0][1] - line2[0][3])
        print("intersect:", x, y, line1, line2)
        if whole_pix:
          return [round(x), round(y)]
        return [x, y]
    
    def is_same_line(self, line1, line2, same_point=40):
        if self.is_parallel(line1, line2) and self.line_intersection(line1, line2) is not None:
          # print("same line: line1, line2", line1, line2)
          return True
        if self.is_parallel(line1, line2):
          for i in range(2):
            for j in range(2):
              dist = np.linalg.norm(np.array([line1[0][i*2],line1[0][i*2+1]]) - np.array([line2[0][j*2], line2[0][j*2+1]]))
              if dist < same_point:
                return True
        return False
    
    def extend_line(self, line1, line2):
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        def get_dist(x1,y1, x2, y2):
            return sqrt((x2-x1)**2 + (y2-y1)**2)
    
        if not self.is_parallel(line1, line2):
          print("not parallel", line1, line2)
          return None
    
        dist0 = get_dist(line1[0][0], line1[0][1], line2[0][0], line2[0][1])
        dist1 = get_dist(line1[0][0], line1[0][1], line2[0][2], line2[0][3])
        dist2 = get_dist(line1[0][2], line1[0][3], line2[0][0], line2[0][1])
        dist3 = get_dist(line1[0][2], line1[0][3], line2[0][2], line2[0][3])
    
        extended_line = None
        lst = [dist0,dist1,dist2,dist3]
        lst.sort()
        for i in range(4):
          if dist0 == lst[i]:
            extended_line = [[line1[0][0], line1[0][1], line2[0][0], line2[0][1]]]
          elif dist1 == lst[i]:
            extended_line = [[line1[0][0], line1[0][1], line2[0][2], line2[0][3]]]
          elif dist2 == lst[i]:
            extended_line = [[line1[0][2], line1[0][3], line2[0][0], line2[0][1]]]
          elif dist3 == lst[i]:
            extended_line = [[line1[0][2], line1[0][3], line2[0][2], line2[0][3]]]
       
          if (self.is_parallel(line1, extended_line) and self.is_parallel(line2, extended_line)):
            break
          else:
            if i == 4: 
              print("No Matches: extended line not parallel", line1, line2, extended_line)
              if (get_dst(line1[0][2], line1[0][3], line1[0][0], line1[0][1]) >
                  get_dst(line2[0][2], line2[0][3], line2[0][0], line2[0][1])):
                return line1
              else:
                return line2

        # print("broken line: ", line1, line2, extended_line)
        return extended_line
  
    def is_broken_line(self, line1, line2, same_point=600):
        if self.line_intersection(line1, line2) is not None:
          return None
        if self.is_parallel(line1, line2):
          dist = np.linalg.norm(np.array([line1[0][0],line1[0][1]]) - np.array([line2[0][0], line2[0][1]]))
          if dist <= same_point:
            return self.extend_line(line1, line2)
          dist = np.linalg.norm(np.array([line1[0][2],line1[0][3]]) - np.array([line2[0][0], line2[0][1]]))
          if dist <= same_point:
            return self.extend_line(line1, line2)
          dist = np.linalg.norm(np.array([line1[0][0],line1[0][1]]) - np.array([line2[0][2], line2[0][3]]))
          if dist <= same_point:
            return self.extend_line(line1, line2)
          dist = np.linalg.norm(np.array([line1[0][2],line1[0][3]]) - np.array([line2[0][2], line2[0][3]]))
          if dist <= same_point:
            return self.extend_line(line1, line2)
        return None
    
    def is_parallel(self, line1, line2):
        angle1 = np.arctan2((line1[0][0]-line1[0][2]), (line1[0][1]-line1[0][3]))
        angle2 = np.arctan2((line2[0][0]-line2[0][2]), (line2[0][1]-line2[0][3]))
        allowed_delta = .1
        if abs(angle1-angle2) <= allowed_delta:
          # print("is_parallel line1, line2", line1, line2, angle1, angle2)
          return True
        if abs(np.pi-abs(angle1-angle2)) <= allowed_delta:
          # note: .01 and 3.14 should be considered parallel
          # print("is_parallel line1, line2", line1, line2, angle1, angle2)
          return True
        return False

    # use with class cvu.HoughBundler() 
    def combine_lines(self, lines, img=None, dbg=False):
      unchanged = False
      while not unchanged:
        unchanged = True
        combined_lines = []
        combined_line_numbers = {}
        for l1_n, line1 in enumerate(lines):
          replace = False
          extended = False
          for l2_n, line2 in enumerate(lines):
            if l2_n < l1_n:
              continue
            # if line2 is extended, then we need to check again 
            try_again = True
            while try_again:
              try_again = False
              try:
                if l2_n in combined_line_numbers[l2_n]:
                  line3 = lines[combined_line_numbers[l2_n]]
                else:
                  line3 = line2
              except:
                continue

              ##############
              # broken lines
              extended_line = self.is_broken_line(line1, line3)
              if extended_line is not None:
                if replace:
                  combined_lines[len(combined_lines)-1] = copy.deepcopy(extended_line)
                else:
                  combined_lines.append(copy.deepcopy(extended_line))
                  replace = True
                try_again = True
                extended = True
                if dbg:
                  print("broken_line", line1, line2, extended_line)
              ################
              # parallel intersecting lines
              elif self.is_same_line(line1, line3):
                extended_line = self.extend_line(line1, line3)
                if replace:
                  combined_lines[len(combined_lines)-1] = copy.deepcopy(extended_line)
                else:
                  combined_lines.append(copy.deepcopy(extended_line))
                  replace = True
                combined_line_numbers[l2_n] = l2_1
                try_again = True
                extended = True
                if dbg:
                  print("same_lines", line1, line2, extended_line)
          if extended or try_again:
            unchanged = False
          if not extended:
            # just add original line
            combined_lines.append(copy.deepcopy(line1))

        if dbg and img is not None:
          if combined_lines is None:
            return None
          tmpimg = copy.deepcopy(img)
          for line in combined_lines:
            for x1,y1,x2,y2 in line:
              cv2.line(tmpimg,(x1,y1),(x2,y2),(255,0,0),5)
          cv2.imshow("combined lines", tmpimg)
          # cv2.waitKey(0)

        # ARD: TODO: many short parallel lines combined together to form
        # a long line.
        return combined_lines
    
    def parallel_dist(self, line1, line2, dbg=False):
        if not self.is_parallel(line1, line2):
          return None
        # line1, line2 [[151 138 223 149]] [[ 38  76 139  96]]
    
        # y = mx + c
        # pts = [(line1[0][0], line1[0][2]), (line1[0][1], line1[0][3])]
        pts = [(line1[0][0], line1[0][1]), (line1[0][2], line1[0][3])]
        x_coords, y_coords = zip(*pts)
        A = np.vstack([x_coords,np.ones(len(x_coords))]).T
        l1_m, l1_c = np.linalg.lstsq(A, y_coords)[0]
        if dbg:
          print("x,y,m,c", x_coords, y_coords, l1_m, l1_c)
    
        pts = [(line2[0][0], line2[0][1]), (line2[0][2], line2[0][3])]
        x_coords, y_coords = zip(*pts)
        A = np.vstack([x_coords,np.ones(len(x_coords))]).T
        l2_m, l2_c = np.linalg.lstsq(A, y_coords)[0]
        if dbg:
          print("x,y,m,c", x_coords, y_coords, l2_m, l2_c)

        # Why not take parallel line vertical distance? 
    
        # coefficients = np.polyfit(x_val, y_val, 1)
        # Goal: set vert(y) the same on both lines, compute horiz(x).
        # with a vertical line, displacement will be very hard to compute
        # unless same end-points are displaced.
        if ((line1[0][0] >= line2[0][0] >= line1[0][2]) or
            (line1[0][0] <= line2[0][0] <= line1[0][2])):
          x1 = line2[0][0]
          y1 = line2[0][1]
          y2 = y1
          x2 = (y2 - l1_c) / l1_m
          # y2 = l1_m * x1 + l1_c
          # x2 = (y1 - l2_c) / l2_m
        elif ((line1[0][0] >= line2[0][2] >= line1[0][2]) or
              (line1[0][0] <= line2[0][2] <= line1[0][2])):
          x1 = line2[0][2]
          y1 = line2[0][3]
          y2 = y1
          x2 = (y2 - l1_c) / l1_m
          # y2 = l1_m * x1 + l1_c
          # x2 = (y1 - l2_c) / l2_m
        elif ((line2[0][0] >= line1[0][0] >= line2[0][2]) or
              (line2[0][0] <= line1[0][0] <= line2[0][2])):
          x1 = line1[0][0]
          y1 = line1[0][1]
          y2 = y1
          x2 = (y2 - l2_c) / l2_m
        elif ((line2[0][0] >= line1[0][2] >= line2[0][2]) or
              (line2[0][0] <= line1[0][2] <= line2[0][2])):
          x1 = line1[0][2]
          y1 = line1[0][3]
          y2 = y1
          x2 = (y2 - l2_c) / l2_m
        elif ((line1[0][1] >= line2[0][1] >= line1[0][3]) or
              (line1[0][1] <= line2[0][1] <= line1[0][3])):
          x1 = line2[0][0]
          y1 = line2[0][1]
          y2 = y1
          x2 = (y2 - l1_c) / l1_m
        elif ((line1[0][1] >= line2[0][3] >= line1[0][3]) or
              (line1[0][1] <= line2[0][3] <= line1[0][3])):
          y1 = line2[0][3]
          x1 = line2[0][2]
          y2 = y1
          x2 = (y2 - l1_c) / l1_m
        elif ((line2[0][1] >= line1[0][1] >= line2[0][3]) or
              (line2[0][1] <= line1[0][1] <= line2[0][3])):
          y1 = line1[0][1]
          x1 = line1[0][0]
          y2 = y1
          x2 = (y2 - l2_c) / l2_m
        elif ((line2[0][1] >= line1[0][3] >= line2[0][3]) or
              (line2[0][1] <= line1[0][3] <= line2[0][3])):
          y1 = line1[0][3]
          x1 = line1[0][2]
          y2 = y1
          x2 = (y2 - l2_c) / l2_m
        else:
          return None
        # print("parallel_dist", (x1-x2),(y1-y2))
        return x1-x2, y1 - y2

    # moved/copied from analyze_gripper
    def gripper_mask_line(self, image, lines, l_gripper_mask, r_gripper_mask):
        # note: mask defined by percentages of width / height
        # l_gripper_mask, r_gripper_mask are ignored.
        height = len(image)
        width = len(image[0])
  
        min_height = int(.66 * height)
        max_height = height
        l_min_width = 0
        l_max_width = int(.3333 * width)
        r_min_width = int(.6666 * width)
        r_max_width = width
        gml1 = [[l_min_width, min_height, l_max_width, min_height]]
        gml2 = [[l_max_width, min_height, l_max_width, max_height]]
        gml3 = [[r_min_width, min_height, r_max_width, min_height]]
        gml4 = [[r_max_width, min_height, r_max_width, max_height]]
        delta = 5
        gm_lines = [] 
        for ln in lines:
          masked = False 
          if len(ln) == 1:
            l1 = ln[0]
          else:
            l1 = ln
  
          [x1,y1,x2,y2] = l1
          if (((x1 > l_min_width-delta and x1 < l_max_width + delta and
                x2 > l_min_width-delta and x2 < l_max_width + delta) or
               (x1 > r_min_width-delta and x1 < r_max_width + delta and
                x2 > r_min_width-delta and x2 < r_max_width + delta)) and
              y1 > min_height-delta and y1 < max_height + delta and
              y2 > min_height-delta and y2 < max_height + delta):
            # print("gripper filtered line:", l1)
            continue
          elif (self.line_intersection(gml1, [l1]) is not None or
                self.line_intersection(gml2, [l1]) is not None or
                self.line_intersection(gml3, [l1]) is not None or
                self.line_intersection(gml4, [l1]) is not None):
            # print("gripper filtered line:", l1)
            continue
          else:
            gm_lines.append([[x1,y1,x2,y2]])
        # print("gm_lines:", gm_lines)
        return gm_lines
    
    def get_hough_lines(self, img, max_line_gap = 10):
          # rho_resolution = 1
          # theta_resolution = np.pi/180
          # threshold = 155
          rho = 1  # distance resolution in pixels of the Hough grid
          theta = np.pi / 180  # angular resolution in radians of the Hough grid
          #  threshold = 15  # minimum number of votes (intersections in Hough grid cell)
          threshold = 10  # minimum number of votes (intersections in Hough grid cell)
          # min_line_length = 50  # minimum number of pixels making up a line
          min_line_length = 40  # minimum number of pixels making up a line
          # max_line_gap = 10  # maximum gap in pixels between connectable line segments
          # max_line_gap = 5  # maximum gap in pixels between connectable line segments
    
          # Gets both Hough Lines and FLD lines and combines them
          # Output "lines" is an array containing endpoints of detected line segments
          hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
          # print("Hough_lines:", hough_lines)
          fld_lines = self.FLD(img)
          # print("FLD:", fld_lines)
          for line in hough_lines:
            # ensure that all lines are the same format
            x1,y1,x2,y2 = line[0]
            fld_lines.append([[x1,y1,x2,y2]])
          # return hough_lines
          return fld_lines
    
    def get_hough_lines_img(self, hough_lines, gray):
        hough_lines_image = np.zeros_like(gray)
        if hough_lines is not None:
          for line in hough_lines:
            for x1,y1,x2,y2 in line:
              print("hough_lines_image",(x1,y1),(x2,y2))
              # cv2.line(hough_lines_image,(x1,y1),(x2,y2),0,3, cv2.LINE_AA)
              hough_lines_image = cv2.line(hough_lines_image,(x1,y1),(x2,y2),255,3, cv2.LINE_AA)
              # cv2.line(imglinesp, (l0[0], l0[1]), (l1[0], l1[1]), (0,255,0), 3, cv2.LINE_AA)
              cv2.imshow("hough_lines_image:",hough_lines_image)
              # cv2.waitKey(0)

        else:
          print("No houghlines")
        return hough_lines_image

    def FLD(self, image):
        # Create default Fast Line Detector class
        fld = cv2.ximgproc.createFastLineDetector()
        # Get line vectors from the image
        lines = fld.detect(image)
        # Draw lines on the image
        line_on_image = fld.drawSegments(image, lines)
        # Plot
        # plt.imshow(line_on_image, interpolation='nearest', aspect='auto')
        # plt.show()
        if False:
          cv2.imshow("FLD:",line_on_image)
          # cv2.waitKey(0)
        # return line_on_image
        int_lines = []
        for line in lines:
          for x1,y1,x2,y2 in line:
            int_lines.append([[round(x1), round(y1), round(x2), round(y2)]])
        return int_lines

    def same_as_endpt(self, pt, ln, same_point=40):
        dist = np.linalg.norm(np.array([ln[0][0],ln[0][1]]) - np.array(pt))
        if dist <= same_point:
          return True
        dist = np.linalg.norm(np.array([ln[0][2],ln[0][3]]) - np.array(pt))
        if dist <= same_point:
          return True
        return False
    
    def is_min_line_len(self, ln, min_line_len):
        x1,y1,x2,y2 = ln[0]
        line_len = np.linalg.norm(np.array([x1,y1]) - np.array([x2, y2]))
        if line_len < min_line_len:
          return False
        else:
          return True

    def closest_pt_on_ln(self, ln, pt):
        [[x1, y1, x2, y2]] = ln
        x3, y3 =  pt
        dx, dy = x2-x1, y2-y1
        u =  ((x3 - x1) * dx + (y3 - y1) * dy) / float(dx*dx + dy*dy)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        x = x1 + u * dx
        y = y1 + u * dy
        return x, y

    def get_dist(self, x1,y1, x2, y2):
      return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def get_frame_lines(self, curr_img, min_line_len=40):
          try:
            gray = cv2.cvtColor(curr_img.copy(), cv2.COLOR_BGR2GRAY)
          except:
            gray = curr_img.copy()
            # gray = cv2.bitwise_not(gray)
          edges = cv2.Canny(gray, 50, 200, None, 3)
          edges = cv2.dilate(edges,None,iterations = 1)
          # cv2.imshow("edges1", edges)
          # cv2.imshow("edges2", edges)
          hough_lines_image = np.zeros_like(curr_img)
          real_hough_lines = self.get_hough_lines(edges)
          for line in real_hough_lines:
            # print("line: ", line)
            if len(line) == 1:
              ln = line[0]
            else:
              ln = line
            [x1,y1,x2,y2] = ln
            cv2.line(hough_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
          if False:
            cv2.imshow("Orig box Lines houghlines", hough_lines_image)
            # cv2.waitKey(0)
          hough_lines = self.cvu.get_lines(edges)
          hough_lines = hough_lines[0]
          hough_lines_image = np.zeros_like(curr_img)
          # print("hough_lines", hough_lines)
          if hough_lines is None:
            return None
          for line in real_hough_lines:
            # ensure that all lines are the same format
            x1,y1,x2,y2 = line[0]
            hough_lines.append([[x1,y1,x2,y2]])
          # if l_gripper_mask is not None and r_gripper_mask is not None:

          hough_lines = self.combine_lines(hough_lines)
          print("final lines", hough_lines)
          for line in hough_lines:
            if len(line) == 1:
              ln = line[0]
            else:
              ln = line
            [x1,y1,x2,y2] = ln
            cv2.line(hough_lines_image,(x1,y1),(x2,y2),(255,0,0),5)
          if True:
            cv2.imshow("combined houghlines", hough_lines_image)
            # cv2.waitKey(0)

          ##########################
          # Add subsections of lines that intersect so that
          # their corners can be analyzed.
          # Ensure min len.
          final_lines = []
          intersecting_subsections = []
          for i, ln1 in enumerate(hough_lines):
            if not self.is_min_line_len(ln1, min_line_len):
              # print("line too short:", ln1)
              continue
            else:
              final_lines.append(ln1)
          for i, ln1 in enumerate(final_lines):
            for j, ln2 in enumerate(final_lines):
              if i == j:
                continue
              # Use intersection of 2 lines, not the average of crnrs
              # print("check intersect:", ln1, ln2)
              crnr = self.line_intersection(ln1, ln2, whole_pix=True)
              if crnr is None:
                continue
              if not self.same_as_endpt(crnr, ln1):
                new_line1 = [[ln1[0][0], ln1[0][1], crnr[0], crnr[1]]]
                if self.is_min_line_len(new_line1, min_line_len):
                  intersecting_subsections.append(new_line1)
                new_line2 = [[crnr[0], crnr[1], ln1[0][2], ln1[0][3]]]
                if self.is_min_line_len(new_line2, min_line_len):
                  intersecting_subsections.append(new_line2)
              if not self.same_as_endpt(crnr, ln2):
                new_line3 = [[ln2[0][0], ln2[0][1], crnr[0], crnr[1]]]
                if self.is_min_line_len(new_line3, min_line_len):
                  intersecting_subsections.append(new_line3)
                new_line4 = [[crnr[0], crnr[1], ln2[0][2], ln2[0][3]]]
                if self.is_min_line_len(new_line3, min_line_len):
                  intersecting_subsections.append(new_line4)
              
          print("intersecting_subsections:", intersecting_subsections)
          final_lines = final_lines + intersecting_subsections
          return final_lines
    
    def exact_line(self, l1, l2):
        if l1[0][0]==l2[0][0] and l1[0][1]==l2[0][1] and l1[0][2]==l2[0][2] and l1[0][3]==l1[0][3]:
          return True
        return False
    
    def display_lines(self, img_label, line_lst, curr_img):
        lines_image = np.zeros_like(curr_img)
        for line in line_lst:
          print("line:", line)
          # for x1,y1,x2,y2 in line[0]:
          x1,y1,x2,y2 = line[0]
          cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),5)
        cv2.imshow(img_label, lines_image)
    
    def display_line_pairs(self, img_label, line_pair_lst, curr_img, mode=2):
        lines_image = np.zeros_like(curr_img)
        for line0, line1, rslt in line_pair_lst:
          if mode == 0:
            # print("line0:", img_label, line0)
            pass
          elif mode == 1:
            # print("line1:", img_label, line1)
            pass
          elif mode == 2:
            pass
            # print("line0,line1:", img_label, line0, line1)
          if mode == 0 or mode == 2:
            x1,y1,x2,y2 = line0[0]
            cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0),3)
          if mode == 1 or mode == 2:
            x1,y1,x2,y2 = line1[0]
            cv2.line(lines_image,(x1,y1),(x2,y2),(130,0,0),5)
        cv2.imshow(img_label, lines_image)
    

    def transform_end_points(self, from_frame_num, from_line_end_points, to_frame_num):
        transformed_lines = []
        ln = copy.deepcopy(from_line_end_points)
        for fr_num in range(from_frame_num+1, to_frame_num):
          if self.M_inv is not None:
            # self.M_inv[fr_num]
            # inverse matrix of simple rotation is reversed rotation.
            points = []
            for x1,y1,x2,y2 in ln[0]:
              # points = np.array([[35.,  0.], [175., 0.], [105., 200.], [105., 215.], ])
              points.append([x1,y1])
              points.append([x2,y2])
            # add ones
            ones = np.ones(shape=(len(points), 1))
            points_ones = np.hstack([points, ones])
            # transform points
            transformed_points = self.M_inv[fr_num].dot(points_ones.T).T
            transformed_lines = []
            x = 1
            for x,y in transformed_points:
              if x == 1:
                ln = [x,y]
              else:
                ln.append(x)
                ln.append(y)
                transformed_lines.append(ln)
              x = 3-x
          else:
            transformed_lines = []
            for i, [x1,y1,x2,y2] in enumerate(line_end_points):
              # [delta_x,delta_y] = self.pix_moved[frame_num]
              [delta_x,delta_y] = self.alset_state.get_pixels_moved(frame_num)
              ln = [x1+delta_x, y1+delta_y, x2+delta_x, y2+delta_y]
              transformed_lines.append(ln)
          ln = [ln]
          print("ln", ln, transformed_lines)
        return ln


    def __init__(self):
        # [[ds_num, img_num]]
        # self.frame_lines is maintained by AnalyzeLines for alset_state
        self.INFINITE = 1000000000000000000
        # self.IMG_H = 416
        self.IMG_H = 256
        self.line_state = {}
        self.line_state["same_frame_analysis"] = []
        self.line_state["cross_frame_analysis"] = {}
        self.line_state["cross_frame_analysis"]["same_lines"] = []
        self.line_state["cross_frame_analysis"]["broken_lines"] = []
        self.line_state["cross_frame_analysis"]["parallel_lines"] = []
        # self.M_inv is maintained by AnalyzeLines
        self.M_inv = []
        self.cvu = CVAnalysisTools()
