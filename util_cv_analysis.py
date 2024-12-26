import numpy as np
import cv2
import math
import statistics
from shapely.geometry import *
from imutils import *
# from svr_config import *
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.measure import label
from shapely.geometry import *
from pprint import pprint
# from util_borders import *

class CVAnalysisTools():
  def __init__(self):
      # foreground/background for movement detection
      self.background = None
      self.foreground = None
      self.unmoved_pix = None
      self.INFINITE = 1000000000000000000
      self.BLACK = [0,0,0]
      self.prev_foreground = None
      # store the number of points and radius for color histogram
      self.n_points = None
      self.radius = None
      self.TEXTURE_METHOD = 'uniform'
      self.textures = {}

  def moved_pixels(self, prev_img, curr_img, init=False, add_edges=False):
      try:
        prev_img = cv2.imread(prev_img)
        curr_img = cv2.imread(curr_img)
      except:
        pass  # assume already in image format
      if add_edges:
        prev_img = cv2.Canny(prev_img, 50, 200, None, 3)
        curr_img = cv2.Canny(curr_img, 50, 200, None, 3)
      thresh = 15
      # thresh = 10  # don't include shadows
      prev_img = cv2.GaussianBlur(prev_img, (5, 5), 0)
      curr_img = cv2.GaussianBlur(curr_img, (5, 5), 0)
      try:
        self.background = cv2.cvtColor(prev_img, cv2.COLOR_RGB2BGR)
        self.background = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_curr_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR)
        gray_curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
      except:
        self.background = prev_img
        gray_curr_img = curr_img
      diff = cv2.absdiff(gray_curr_img, self.background)
      self.foreground = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY_INV)[1]
      # cv2.imshow("Gripper FG", self.foreground)
      # cv2.imshow("Gripper IM", gray_curr_img)
      # invert mask and combine with original image - this makes the black outer edge white
      mask_inv = cv2.bitwise_not(self.foreground)
      moved_pix = cv2.bitwise_or(gray_curr_img, mask_inv)
      # cv2.imshow("moved pix", moved_pix)
      # cv2.waitKey(0)
      # print("Moved Pixels. Press return to continue.")
      # wait_for_return = input()
      return self.foreground

      # left_bb, right_bb, moved_pix = self.get_gripper_bounding_box(moved_pix, curr_img)
      # return left_bb, right_bb, moved_pix

  # NOTES: from ALSET utilborders.py
  def get_min_max_borders(self, border):
      b = []
      for bdr in border:
        b.append(list(bdr[0]))
      poly = Polygon(b)
      minw, minh, maxw, maxh = poly.bounds
      return int(maxw), int(minw), int(maxh), int(minh)

  def get_grasping_bounding_box(self, unmoved_pix, image):
      ret, contour_thresh = cv2.threshold(unmoved_pix.copy(), 125, 255, 0)
      contour_thresh = cv2.dilate(contour_thresh,None,iterations = 10)

      contours, hierarchy = cv2.findContours(contour_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      # print("hierarchy:", hierarchy)

      # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
      #   cv2.CHAIN_APPROX_SIMPLE)

      # for accuracy in range(10):
      #   accuracy_factor = .1 * (accuracy+1)
      # accuracy_factor = .01
      accuracy_factor = .1
      x,y,ch = image.shape
      # gripper_bounding_box = [[[0, 0]], [[x-1, 0]], [[x-1, y-1]], [[ 0, y-1 ]]]
      bb_found = False
      bb = None
      max_accuracy = 0
      for c in contours:
        if len(contours) > 1:
          area = cv2.contourArea(c)
          if area < 16:
            continue
          # Calculate accuracy as a percent of the contour perimeter
          accuracy = accuracy_factor * cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, accuracy, True)
          if len(approx) < 2:
            continue
          if accuracy > max_accuracy:
            max_accuracy = accuracy 
            print("Found Better Gripper Bounding Box")
            if len(approx) == 2:
              maxx = max(approx[0][0][0], approx[1][0][0])
              minx = min(approx[0][0][0], approx[1][0][0])
              maxy = max(approx[0][0][1], approx[1][0][1])
              miny = min(approx[0][0][1], approx[1][0][1])
              print("0maxy, miny, maxx, minx:", maxy, miny, maxx, minx)
            else:
              maxx, minx, maxy, miny = self.get_min_max_borders(approx)
              print("1maxx, minx, maxy, miny:", maxx, minx, maxy, miny)
            bb = [[[minx,miny]], [[minx, y-1]], [[maxx, y-1]], [[maxx, miny]]]
            bb_found = True
            print("approx:", approx)
            print("g_bb:", bb)
            # break

      if bb is None:
        g_bb = None
      else:
        g_bb = np.intp(bb)
        image = cv2.drawContours(image.copy(), [g_bb], 0, (0, 255, 0), 2)
      print("g_bb", g_bb)

      return g_bb, image

  def findBigArea(self, arr, row=None, col=None):
      # labarr = skimage.measure.label(arr)
      labarr = label(arr)
      unique_val, unique_counts = np.unique(labarr, return_counts=True)
      # print("unique_val", unique_val)
      # print("unique_counts", unique_counts)
      # Find indices with send highest count
      found = False
      for uc in range(1,len(unique_counts)) :
        if row is None or col is None:
          # find the maximum non-zero counts
          max_column = max(unique_counts[uc:]) # 0 is the highest count
        else:
          # try all non-zero counts to find the specified row/col
          max_column = unique_counts[uc]      # 0 is the highest count
        print(uc, "max unique_counts", max_column)
        index_tpl = np.where(unique_counts == max_column)  
        # print("index:",index_tpl)
        index_num = np.array(index_tpl)[0,0]
        print("index_num",index_num)
        # print("index_num",np.array(index_num)[0,0]),
        print(unique_val[index_num])
        mask = labarr[:,:] == unique_val[index_num]
        # Find array indices of area
        index_nums = np.array(np.nonzero(mask))  
        min_row = min(index_nums[1,:])
        max_row = max(index_nums[1,:])
        min_col = min(index_nums[0,:])
        max_col = max(index_nums[0,:])
        if row is None or col is None:
          # stop with highest non-zero count
          found = True
          break
        elif min_row <= row and row <= max_row and min_col <= col and col <= max_col:
          found = True
          break
      if not found: 
        print("No grasping bounding box found")
        return None
      print("minr, maxr, minc, maxc", min_row, max_row, min_col, max_col)
      num_pixels_horiz = max_col - min_col
      num_pixels_vert= max_row - min_row
      bot_x = int((max_col + min_col) / 2)
      bot_y = int((max_row + min_row) / 2)
      print("middle of gripper:", bot_x, bot_y)
      print("num_pixels horiz, vert:", num_pixels_horiz, num_pixels_vert)
      # return {"x": bot_x, "y":bot_y, 
      #    "pixels_horiz": num_pixels_horiz, "pixels_vert":num_pixels_vert}
      # return bb
      return [[int(min_row), int(min_col)],[int(max_row), int(max_col)]]   


#####################################
# NOTES: from ALSET utilborders.py

  def replace_border(img, desired_height, desired_width, offset_height, offset_width):
    # 224 224 225 225 0
    shape, border = real_map_border(img)
    maxw, minw, maxh, minh = self.get_min_max_borders(border)
    print("maxh, minh, maxw, minw :", maxh, minh, maxw, minw, desired_height, desired_width, offset_height, offset_width)
    extract_img_rect = img[minh:maxh, minw:maxw]
    extract_height, extract_width = extract_img_rect.shape[:2]
    insert_height = int(extract_height + 2*abs(offset_height))
    insert_width  = int(extract_width + 2*abs(offset_width))
    insert_img_rect = np.zeros((insert_height,insert_width,3),dtype="uint8")
    print("ext_h, ins_h, off_h:", extract_height, insert_height, offset_height)
    for eh in range(extract_height):
      for ew in range(extract_width):
        new_w = ew + abs(offset_width) + offset_width
        new_h = eh + abs(offset_height) + offset_height
        insert_img_rect[new_h, new_w] = extract_img_rect[eh,ew]

    border_top = int((desired_height - insert_height)/2)
    border_bottom = desired_height - border_top - insert_height
    border_left = int((desired_width - insert_width)/2)
    border_right = desired_width - border_left - insert_width
    print("replace_border:",border_top, border_bottom, border_left, border_right, offset_height, offset_width)
    bordered_img = cv2.copyMakeBorder(
        insert_img_rect,
        top=border_top,
        bottom=border_bottom,
        left=border_left,
        right=border_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # black
    )
    return bordered_img


  ###############################################
  def get_lines(self, img):
      # Canny: Necessary parameters are:
      #   image: Source/Input image of n-dimensional array.
      #   threshold1: It is the High threshold value of intensity gradient.
      #   threshold2: It is the Low threshold value of intensity gradient.
      # Canny: Optional parameters are:
      #   apertureSize: Order of Kernel(matrix) for the Sobel filter. 
      #      Its default value is (3 x 3), and its value should be odd between 3 and 7. 
      #      It is used for finding image gradients. Filter is used for smoothening and 
      #      sharpening of an image.
      #   L2gradient: This specifies the equation for finding gradient magnitude. 
      #      L2gradient is of boolean type, and its default value is False.
      # edges = cv2.Canny(img, 75, 200, None, 3)
      # edges = cv2.Canny(img, 50, 200, None, 3)
      # edges = cv2.Canny(img.copy(), 50, 200, None, 3)
      edges = cv2.Canny(img.copy(), 10, 245, None, 3)
      # edges = cv2.Canny(img,100,200)
      # Copy edges to the images that will display the results in BGR
      imglinesp = np.copy(img)
      # HoughLinesP Parameters:
      #   image: 8-bit, single-channel binary source image. 
      #   lines:Output vector of lines.
      #   rho: Distance resolution of the accumulator in pixels.
      #   theta: Angle resolution of the accumulator in radians.
      #   threshold: Accumulator threshold parameter. Only those lines are returned that get 
      #       enough votes ( >threshold ).
      #   minLineLength: Minimum line length. Line segments shorter than that are rejected.
      #   maxLineGap: Maximum allowed gap between points on the same line to link them.
      #
      #   linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 10, 10)
      # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 20, 20)
      linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 40, 40)
      HB = HoughBundler()
      mergedLinesP = HB.process_lines(linesP, edges)
      mlLinesP = []                 # put back in HoughLineP format
      if mergedLinesP is not None:
          if linesP is not None:
            print("num linesP:", len(linesP), len(mergedLinesP))
          for i in range(0, len(mergedLinesP)):
              l0 = mergedLinesP[i][0]
              l1 = mergedLinesP[i][1]
              cv2.line(imglinesp, (l0[0], l0[1]), (l1[0], l1[1]), (0,255,0), 3, cv2.LINE_AA)
              mlLinesP.append([[l0[0], l0[1], l1[0], l1[1]]])
      # cv2.imshow("lines:", imglinesp)
      # cv2.waitKey(0)
      return mlLinesP, imglinesp

  def find_longest_line(self, linesP, border=None):
      max_dist = 0
      map_line = None
      map_dx = None
      map_dy = None
      map_slope = None
      in_brdr_cnt = 0
      for [[l0,l1,l2,l3]] in linesP:
          if border is None or self.line_in_border(border, (l0,l1), (l2,l3)):
            # print("in border:", l0,l1,l2,l3)
            in_brdr_cnt += 1
            dx = l0 - l2
            dy = l1 - l3
            dist = np.sqrt(dx**1 + dy**2)
            if max_dist < dist:
              map_line = [l0,l1,l2,l3]
              map_dx = dx
              map_dy = dy
              map_slope = np.arctan2(map_dx, map_dy)
              if map_slope < 0:   # keep all slopes positive
                map_slope = 2 * np.pi + map_slope
              max_dist = dist
      return max_dist, map_line, map_dx, map_dy, map_slope, in_brdr_cnt

  def rectangle_within_image(self, img):
      # the following is not what we're looking for mapping, but may serve 
      # as a prototype for now. The following cuts away more than the borders
      # to get a clean image. We just want to know what the external borders are.
      #
      # convert the stitched image to grayscale and threshold it
      # such that all pixels greater than zero are set to 255
      # (foreground) while all others remain 0 (background)

      # input image is expected to already be gray
      # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]

      # find all external contours in the threshold image then find
      # the *largest* contour which will be the contour/outline of
      # the image
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
      	cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      c = max(cnts, key=cv2.contourArea)
      # allocate memory for the mask which will contain the
      # rectangular bounding box of the image region
      mask = np.zeros(thresh.shape, dtype="uint8")
      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
      # create two copies of the mask: one to serve as our actual
      # minimum rectangular region and another to serve as a counter
      # for how many pixels need to be removed to form the minimum
      # rectangular region
      minRect = mask.copy()
      sub = mask.copy()
      # keep looping until there are no non-zero pixels left in the
      # subtracted image
      while cv2.countNonZero(sub) > 0:
      	# erode the minimum rectangular mask and then subtract
      	# the thresholded image from the minimum rectangular mask
      	# so we can count if there are any non-zero pixels left
      	minRect = cv2.erode(minRect, None)
      	sub = cv2.subtract(minRect, thresh)
      # find contours in the minimum rectangular mask and then
      # extract the bounding box (x, y)-coordinates
      cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
      	cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      c = max(cnts, key=cv2.contourArea)
      (x, y, w, h) = cv2.boundingRect(c)
      return (x, y, w, h) 

  
  def color_quantification(self, img, num_clusters):
      # try color quantification
      Z = img.copy()
      Z = Z.reshape((-1,3))
      # convert to np.float32
      Z = np.float32(Z)
      # define criteria, number of clusters(K) and apply kmeans()
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      K = num_clusters
      # compactness : sum of squared distance from each point to their centers.
      # labels : the label array where each element marked '0', '1'.....
      # centers : This is array of centers of clusters.
      ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
      # Now convert back into uint8, and make original image
      # print("compactness, centers:", ret, center)
      # ret is a single float, label is ?, center is RGB
      center = np.uint8(center)
      res = center[label.flatten()]
      res2 = res.reshape((img.shape))
      return res2

  def order_points(self, pts):
      # initialzie a list of coordinates that will be ordered
      # such that the first entry in the list is the top-left,
      # the second entry is the top-right, the third is the
      # bottom-right, and the fourth is the bottom-left
      rect = np.zeros((4, 2), dtype = "float32")

      # the top-left point will have the smallest sum, whereas
      # the bottom-right point will have the largest sum
      s = pts.sum(axis = 1)
      rect[0] = pts[np.argmin(s)]
      rect[2] = pts[np.argmax(s)]

      # now, compute the difference between the points, the
      # top-right point will have the smallest difference,
      # whereas the bottom-left will have the largest difference
      diff = np.diff(pts, axis = 1)
      rect[1] = pts[np.argmin(diff)]
      rect[3] = pts[np.argmax(diff)]

      # return the ordered coordinates
      return rect

  def four_point_transform(self, image, pts):
      # obtain a consistent order of the points and unpack them
      # individually
      rect = self.order_points(pts)
      (tl, tr, br, bl) = rect
      # compute the width of the new image, which will be the
      # maximum distance between bottom-right and bottom-left
      # x-coordiates or the top-right and top-left x-coordinates
      widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
      widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
      maxWidth = max(int(widthA), int(widthB))
      # compute the height of the new image, which will be the
      # maximum distance between the top-right and bottom-right
      # y-coordinates or the top-left and bottom-left y-coordinates
      heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
      heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
      maxHeight = max(int(heightA), int(heightB))
      # now that we have the dimensions of the new image, construct
      # the set of destination points to obtain a "birds eye view",
      # (i.e. top-down view) of the image, again specifying points
      # in the top-left, top-right, bottom-right, and bottom-left
      # order
      dst = np.array([
            [0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]
            ], dtype = "float32")
     
      # compute the perspective transform matrix and then apply it
      M = cv2.getPerspectiveTransform(dst, rect)
      warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),  cv2.WARP_INVERSE_MAP)
      # return the warped image
      return warped

  #########################
  # Contours
  #########################
  def get_contours(self,img):
      blurred = cv2.GaussianBlur(img, (5, 5), 0)
      gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
      lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
      thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
      # find contours in the thresholded image
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
              cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      return cnts
      
  def draw_contours(self,img,cnt,text="",def_clr=(0,255,0)):
      for i,c in enumerate(cnt):
          # compute the center of the contour
          M = cv2.moments(c)
          # cX = int((M["m10"] / M["m00"]) )
          # cY = int((M["m01"] / M["m00"]) )
          c = c.astype("int")
          # print(i,"c",c)
          itext = text + str(i)
          cv2.drawContours(img, [c], -1, def_clr, 2)
          # cv2.putText(img, itext, (cX, cY),
          #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# from: https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 30
        min_angle_to_merge = 30
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < orientation < 135:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines, img):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        if lines is not None:
          for line_i in [l[0] for l in lines]:
                orientation = self.get_orientation(line_i)
                # if vertical
                if 45 < orientation < 135:
                    lines_y.append(line_i)
                else:
                    lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
                if len(i) > 0:
                    groups = self.merge_lines_pipeline_2(i)
                    merged_lines = []
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments1(group))

                    merged_lines_all.extend(merged_lines)

        return merged_lines_all

