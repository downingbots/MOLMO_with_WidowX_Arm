from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import requests
from PIL import Image
import numpy as np
import re
import widowx_calibrate
import util_perfection
import math

class molmo_perfection():

  def __init__(self):
     # load the processor
     self.processor = AutoProcessor.from_pretrained(
         'allenai/Molmo-7B-D-0924',
         trust_remote_code=True,
         torch_dtype='auto',
         device_map='auto'
     )

     # load the model
     try:
       self.model = AutoModelForCausalLM.from_pretrained(
         'allenai/Molmo-7B-D-0924',
         trust_remote_code=True,
         torch_dtype='auto',
         device_map='auto'
       )
     except Exception as e:
       print(e)
     self.calib = widowx_calibrate.widowx_calibrate()
     self.perfect = util_perfection.perfection()

  def get_pixel_location(self, board_r, board_w):
      print("Board loc  ", board_r, board_w)
      g_loc = self.perfect.get_gripper_location_on_board(board_r, board_w)
      print("Gripper loc", g_loc)
      (px,py,zd) = self.calib.pixel_gripper_conversion(g_loc[0], g_loc[1], up_down="DOWN", dir="G2P")
      px = px*6 - 10 - (board_w-2) * 20
      py = py*4 - 10 + (board_r-2) * 27
      w = 1920
      h = 1080
      # px = px*6 + board_r * 20
      # py = py*4 - 50
      px = round(px / w * 1000) / 10
      py = round(py / h * 1000) / 10
      print("Pixel loc  ", px, py)
      return px,py
  
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
      inputs = self.processor.process(
          images=[self.img],
          text=self.prompt
      )

      # move inputs to the correct device and make a batch of size 1
      inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
  
      # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
      molmo_output = self.model.generate_from_batch(
          inputs,
          GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
          tokenizer=self.processor.tokenizer
      )
      # print("MOLMO_OUTPUT")
      # print(molmo_output)
  
      # only get generated tokens; decode them to text
      generated_tokens = molmo_output[0,inputs['input_ids'].size(1):]
      generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
  
      # print the generated text
      # print("generated_tokens:")
      # print(generated_tokens)
      print("generated_text:")
      print(generated_text)
      image_w, image_h = self.img.size
      print("image_w, image_h = ", image_w, image_h)
      all_points = []
      pct_points = []
      for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', generated_text):
        try:
          point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
          pass
        else:
          point = np.array(point)
          pct_points.append(point)
          if np.max(point) > 100:
              # Treat as an invalid output
              continue
          point /= 100.0
          point = point * np.array([image_w, image_h])
          all_points.append(point)
      print("pix points: ", all_points)
      print("pct points: ", pct_points)
      return(generated_text, all_points, pct_points)


#  def run_game(self):
  def get_ypl1(self):
      prompt = "You are controlling a robot arm to play the Mattel Perfection Game. \
You are given an image of the board and yellow pieces.  You identify the location of \
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
Each yellow pieces has one of the shapes listed above and each piece has a centralized \
a vertical peg sticking up for the robot gripper to grasp. \
The yellow pieces need to be matched to the shapes in the grid of the blue game \
board. The yellow pieces are located next to the blue game board. \
Locations are a percentage of height and width of the image with the origin at the bottom left.  \
Point to all the yellow pieces."

      gen_text, all_pt, yellow_piece_loc = molmo_perf.send_prompt(prompt)
      print("###############")
      print("yellow_piece_loc = ", yellow_piece_loc)
      print("###############")
      return yellow_piece_loc

  def get_ypl2(self):
      # from coords.py:
      #   hot dog = 30,40
      #   circ = 24, 46
      #   trap = 16, 53
      #   tub = 19, 62
      #   Y = 25, 71
      #   rhom = 32, 82
      yellow_piece_loc =  [[0.16, 0.52], [0.195, 0.625], [0.235, 0.455], [0.245, 0.71 ], [0.305, 0.395], [0.31, 0.82]]
      y_shape = ['Trapezoid', 'Tub', 'Circle', 'Y', 'Hot Dog', 'Rhombus']
      b_shape =  ['Tub', 'Y', 'Hot Dog', 'Trapezoid', 'Circle', 'Rhombus']
      return yellow_piece_loc

  def get_blue_board_info1(self):
      blue_shape = []
      blue_pix_loc = []
      blue_yellow_match = []
      b_done_once = False
      b_approx_loc = []
      b_pct_loc = []
      b_shape  = []
      blue_grid_loc = [(0,2), (3,3), (3,1), (1,1), (2,2), (1,3)]
      for i in range(len(blue_grid_loc)):
        r,w = blue_grid_loc[i]
        b_approx_loc.append(molmo_perf.get_pixel_location(r, w))
        b_shape.append(molmo_perf.get_board_shape(r, w))
        prompt = "point to the blue hole located near " + str(b_approx_loc[i]) + " that is shaped like a " +str(b_shape[i])
        gen_text, all_pt, pct_pt = molmo_perf.send_prompt(prompt)
        b_pct_loc.append([round(pct_pt[0][0]*1000)/10, round(pct_pt[0][1]*1000)/10])

      print("###############")
      print("b_approx_loc = ", b_approx_loc)
      print("###############")
      print("b_pct_loc = ", b_pct_loc)
      print("###############")
      print("b_shape = ", b_shape)
      print("###############")
      return blue_grid_loc, b_approx_loc, b_pct_loc, b_shape

  def get_blue_board_info2(self):
      blue_grid_loc = [(0,2), (3,3), (3,1), (1,1), (2,2), (1,3)]
      b_approx_loc =  [(39.5, 70.0), (63.1, 58.6), (51.5, 45.3), (39.6, 57.7), (51.7, 58.0), (51.2, 71.0)]
      b_pct_loc =  [[37.9, 71.0], [64.0, 57.6], [52.0, 44.8], [39.6, 57.7], [51.5, 57.6], [50.5, 72.0]]
      b_shape =  ['Tub', 'Y', 'Hot Dog', 'Trapezoid', 'Circle', 'Rhombus']
      return blue_grid_loc, b_approx_loc, b_pct_loc, b_shape

if True:
      molmo_perf = molmo_perfection()
      molmo_perf.set_img('images/perfection_tt.png')
      yellow_piece_loc = molmo_perf.get_ypl2()
      blue_grid_loc, b_approx_loc, b_pct_loc, b_shape = molmo_perf.get_blue_board_info2()

      ypl_raw_score = []
      ypl_score = []
      for ypl in range(len(yellow_piece_loc)):
        ypl_score.append([])
        for i in range(len(blue_grid_loc)):
          ypl_score[ypl].append(0)
      for i in range(len(blue_grid_loc)):
        for j in range(len(blue_grid_loc)):
          if j == i:
            continue
          for ypl in range(len(yellow_piece_loc)):
            yp_loc = [round(yellow_piece_loc[ypl][0] * 1000)/10, round(yellow_piece_loc[ypl][1] * 1000)/10]
            prompt = "Does the yellow piece at " + str(yp_loc) + " better fits in the hole on blue board at " \
                     + str(b_pct_loc[i]) + " or " + str(b_pct_loc[j]) + " ?  Point to the blue board hole."
            gen_text, all_pt, pct_pt = molmo_perf.send_prompt(prompt)
            better_b_loc = [round(pct_pt[0][0]*1000)/10, round(pct_pt[0][1]*1000)/10]

            print("prompt: ", prompt)
            print("answer:", better_b_loc)
            idist = (math.sqrt(math.pow((b_pct_loc[i][0] - better_b_loc[0]),2) + 
                               math.pow((b_pct_loc[i][1] - better_b_loc[1]),2)))
            jdist = (math.sqrt(math.pow((b_pct_loc[j][0] - better_b_loc[0]),2) + 
                               math.pow((b_pct_loc[j][1] - better_b_loc[1]),2)))
            print("dist 1,2: ", idist, jdist)
            if idist <= jdist:
              ypl_score[ypl][i] += 1
              ypl_raw_score.append([ypl, i, j, i])
            else:
              ypl_score[ypl][j] += 1
              ypl_raw_score.append([ypl, i, j, j])

      print("###############")
      print("ypl_raw_score = ", ypl_raw_score)
      print("###############")
      print("ypl_score = ", ypl_score)
      print("###############")
      ypl_max_score = []
      for ypl in range(len(yellow_piece_loc)):
        ypl_max_score.append(max(ypl_score[ypl]))
      print("ypl_max_score = ", ypl_max_score)
      print("###############")
      for ypl in range(len(yellow_piece_loc)):
        yp_loc = [round(yellow_piece_loc[ypl][0] * 1000)/10, round(yellow_piece_loc[ypl][1] * 1000)/10]
        best_loc = b_pct_loc[ypl_max_score[ypl]]
        answer = "The yellow piece at " + str(yp_loc) + " best fits in the hole on blue board at " + str(best_loc)
        print(answer)
      print("###############")
      exit()

#########################################################################################################
      for r in range(5):
        # yp_loc = [yellow_piece_loc[ypl][0] * 100, yellow_piece_loc[ypl][1] * 100]
        # prompt = "The yellow pieces are one of the following shapes: \ < list from above> 
        #           Which shape is the yellow piece at " + str(yp_loc) + "?"
        # print("prompt:", prompt)
        # print("answer:", gen_text)

          # prompt = "Will the yellow piece at " + str(yp_loc) + " fit snugly in the blue hole located at " + str(b_loc) + " that is shaped like a " + str(b_shape) + "? Answer with Yes or No only."
          # prompt = "Will the yellow piece at " + str(yp_loc) + " fit snugly in the blue hole located at " + str(b_pct_pt[i]) + "? Answer with Yes or No only."
          # prompt = "Is the yellow piece at " + str(yp_loc) + " the same shape as the blue hole located at " + str(b_pct_pt[i]) + "? Answer with Yes or No only."
          # prompt = "Is the yellow piece at " + str(yp_loc) + " shaped like a " + str(b_shape) + "? Answer with Yes or No only."
          # prompt = "The blue holes are one of the following shapes: \ < list from above >
          #           Which shape is the blue hole located at " + str(b_pct_pt[i]) + "?"
          prompt = "Will the yellow piece at " + str(yp_loc) + " fit snugly in the blue hole located at " + str(b_pct_pt[i]) + "? Answer with Yes or No only."
          gen_text, all_pt, pct_pt = molmo_perf.send_prompt(prompt)
      if false:
        blue_shape.append([])
        blue_pix_loc.append([])
        blue_yellow_match.append([])
        for w in range(5):
          blue_pix_loc[r].append(molmo_perf.get_pixel_location(r, w))
          blue_shape[r].append(molmo_perf.get_board_shape(r, w))
          for ypl in range(len(yellow_piece_loc)):
            yp_loc = [yellow_piece_loc[ypl][0] * 100, yellow_piece_loc[ypl][1] * 100]
            # yp_loc = [round(yellow_piece_loc[ypl][0] * molmo_perf.img.size[0]), round(yellow_piece_loc[ypl][1] * molmo_perf.img.size[1])]
            prompt = "Will the yellow piece at " + str(yp_loc) + " fit in the blue hole located at " + str(blue_pix_loc[r][w]) + " that is shaped like a " + str(blue_shape[r][w]) + "? Answer with Yes or No only."

            print("prompt:", prompt)
            gen_text, all_pt, pct_pt = molmo_perf.send_prompt(prompt)
            blue_yellow_match[r].append([])
            if gen_text.strip().upper() == "YES":
              blue_yellow_match[r][w].append(yp_loc)
              print("##################")
              print(blue_shape[r][w], [r, w], blue_pix_loc[r][w], " matches yellow piece ", yp_loc)
              print("##################")
            elif gen_text.strip().upper() == "NO":
              print(blue_shape[r][w], [r, w], blue_pix_loc[r][w], " does not match yellow piece ", yp_loc)
            elif gen_text.strip().upper() != "NO":
              print("prompt:", prompt)
              print("answer:", gen_text)
        print("Final blue-yellow matches:", blue_yellow_match)



