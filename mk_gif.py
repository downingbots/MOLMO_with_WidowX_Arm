import rlds
from IPython import display
from PIL import Image
import numpy as np
import math
import time
import datetime
import json
import os
import copy

#########################################################################
# Functions extracted from DeepMind sample code:
# robotics_open_x_embodiment_and_rt_x_oss_Minimal_example_for_running_inference_using_RT_1_X_TF_using_tensorflow_datasets.ipynb
#########################################################################
def as_gif(images, rbt=False):
  # Render the images as the gif:
  if rbt:
    filenm = '/tmp/temprbt.gif'
  else:
    filenm = '/tmp/temp.gif'

  images[0].save(filenm, save_all=True, append_images=images[1:], duration=1000, loop=0)
  gif_bytes = open(filenm,'rb').read()
  return gif_bytes

def resize(image):
    image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
    image = tf.cast(image, tf.uint8)
    return image

###################################################3
def read_config():
    with open('widowx_config.json') as config_file:
      config_json = config_file.read()
    config = json.loads(config_json)
    return config

def get_initial_state(config):
    init_state = json.loads(config["initial_state"])
    print("init_state", init_state)
    return init_state

def sorted_directory_listing(directory):
    items = os.listdir(directory)
    sorted_items = sorted(items)
    return sorted_items

config = read_config()
gif_images = sorted_directory_listing( config['gif_dir'])
robot_images = []
for gi in gif_images:
  if not gi.startswith("test_image_t") or not gi.endswith(".png"):
    print(gi)
    continue
  filenm = config['gif_dir'] + "/" + gi
  im = Image.open(filenm)
  robot_image = Image.fromarray(np.array(im))
  robot_images.append(robot_image)

display.Image(as_gif(robot_images, True))
time.sleep(100)
