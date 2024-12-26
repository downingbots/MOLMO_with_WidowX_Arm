from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import requests
from PIL import Image
import numpy as np
import re
import widowx_calibrate
import util_perfection
import copy
import cv2

class molmo_perfection():

  def __init__(self):
     # load the processor
     self.calib = widowx_calibrate.widowx_calibrate()
     self.perfect = util_perfection.perfection()

  def get_pixel_location(self, board_r, board_w):
      print("Board loc  ", board_r, board_w)
      g_loc = self.perfect.get_gripper_location_on_board(board_r, board_w)
      print("Gripper loc", g_loc)
      (px,py,zd) = self.calib.pixel_gripper_conversion(g_loc[0], g_loc[1], up_down="DOWN", dir="G2P")
      px = px*6 - 10 - (board_w-2) * 20
      py = py*4 - 10 + (board_r-2) * 27
      # px = px*6 + board_r * 20 
      # py = py*4 - 50 - board_w * 5
      print("Pixel loc  ", px, py)
      return round(px),round(py)
  
  def get_board_shape(self, r, w):
      return self.perfect.get_board_shape(r, w)

  def set_prompt(self, prompt):
      self.prompt = prompt

  def set_img(self, img_nm):
      self.img = Image.open(img_nm)

  def send_prompt(self, prompt, img = None):
      self.set_prompt(prompt)
      if img is not None:
        self.set_img(img)
  
      # print the generated text
      # print("generated_tokens:")
      # print(generated_tokens)
      image_w, image_h = self.img.size
      print("image_w, image_h = ", image_w, image_h)
      all_points = []
      pct_points = []
      print("pix points: ", all_points)
      print("pct points: ", pct_points)
      return(generated_text, all_points, pct_points)


#  def run_game(self):
if True:

      molmo_perf = molmo_perfection()
      molmo_perf.set_img('perfection_tt.png')
      click_image = np.array(molmo_perf.img)
      click_image = cv2.cvtColor(click_image, cv2.COLOR_RGB2BGR)
      # click_image = Image.fromarray(click_image)
      # click_image = Image.fromarray(np.array(click_image))

      molmo_perf.set_img('perfection_tt.png')
      prompt = "You are controlling a robot arm to play the Mattel Perfection Game. \
You are given an image of the board.  You identify the location of \
a yellow game piece in the picture and then select the matching  \
corresponding shape on a blue game board. \
 \
The blue board has 5 rows and 5 columns.  The image shows a blue board  \
with a grid that has rows starting on the upper left and ending on the lower left. \
In order of the front row to back row, from left to right, the shapes in the game  \
board are: \
Row 1: Kite, Pentagon, Tub, X, Equalateral Triangle, \
Row 2: Pizza Slice, Trapezoid, 6-Pointed Star, Rhombus, Octagon, \
Row 3: Parallelogram, Arch, Circle, Cross, S, \
Row 4: 5-Pointed Star, Hot Dog, Right Triangle, Y, Rectangle, \
Row 5: Semicircle, Square, Inverted S, Hexagon, Astrisk  \
 \
The yellow pieces that need to be matched to the shapes in the grid of the blue game \
board are located next to the blue game board.  Point to all the yellow pieces."

      gen_text, all_pt, yellow_piece_loc = molmo_perf.send_prompt(prompt)
      blue_shape = []
      blue_pix_loc = []
      blue_yellow_match = []
      for r in range(5):
        blue_shape.append([])
        blue_pix_loc.append([])
        blue_yellow_match.append([])
        for w in range(5):
          # for ypl in range(len(yellow_piece_loc)):
            blue_pix_loc[r].append(molmo_perf.get_pixel_location(r, w))
            # blue_pix_loc[r].append(molmo_perf.get_pixel_location(w, r))
            blue_shape[r].append(molmo_perf.get_board_shape(r, w))
            # yp_loc = [yellow_piece_loc[ypl][0] * 100, yellow_piece_loc[ypl][1] * 100]
            yp_loc = [0,0]
            prompt = "Will the yellow piece at " + str(yp_loc) + " fit in the blue hole located at " + str(blue_pix_loc[r][w]) + " that is shaped like a " + str(blue_shape[r][w]) + "? Answer with Yes or No only."
            center = (molmo_perf.get_pixel_location(r, w))
            radius = 10
            cv2.circle(click_image,center,radius,(255,255,0),2)
            continue
            # prompt: Will the yellow piece at [0.16  0.527] fit in the blue hole located at (168.62828445325727, 2272.4963336498245) that is shaped like a Kite? Answer with Yes or No only.

            gen_text, all_pt, pct_pt = molmo_perf.send_prompt(prompt)
            blue_yellow_match[r].append([])
            if gen_text.upper() == "YES":
              blue_yellow_match[r][w].append(yellow_piece_loc[ypl])
              print(blue_shape[r][w], r, w, " matches yellow piece ", yellow_piece_loc[ypl])
            elif gen_text.upper() != "NO":
              print("prompt:", prompt)
              print("answer:", gen_text)
      cv2.imshow(None,click_image)
      cv2.waitKey(0)



# pct points:  []
# Astrisk [4, 4] (1484, 594)  does not match yellow piece  [604.8, 893.1600000000001]
# Final blue-yellow matches: [[[[374, 677]], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[[307, 569], [374, 677], [459, 504], [470, 774], [586, 432], [605, 893]], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]

# later run
# Final blue-yellow matches: [[[[605, 893]], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[[307, 569], [374, 677], [459, 504], [470, 774], [586, 432], [605, 893]], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]

#Pizza Slice [1, 0] (648, 547)  matches yellow piece  [307, 569]
#generated_text: Yes
#Pizza Slice [1, 0] (648, 547)  matches yellow piece  [374, 677]
#generated_text: Yes
#Pizza Slice [1, 0] (648, 547)  matches yellow piece  [459, 504]
#generated_text: Yes
#Pizza Slice [1, 0] (648, 547)  matches yellow piece  [470, 774]
#generated_text: Yes
#Pizza Slice [1, 0] (648, 547)  matches yellow piece  [586, 432]
#generated_text: Yes
#Pizza Slice [1, 0] (648, 547)  matches yellow piece  [605, 893]
