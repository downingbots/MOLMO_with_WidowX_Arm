#!/usr/bin/env python
import os
import array
import time
import copy
import struct
import datetime
import json
import camera_snapshot
import widowx
from threading import Thread
import serial
import math
import os
import random
from ctypes import c_uint8 as unsigned_byte

# joystick controls:
# Start: start/stop controlling/recording the robot arm.
# Mode: Toggle to record a Octo dataset for fine-tuning.
# Up cross: Goto centralized initial "pick" position.
# Left/Right cross: swivel pick position to left/right.
# Down cross: Return to "rest" position.
# Left joystick: control the X-Y point of the gripper.
# Right joystick up/down: control the Z point of the gripper.
# Y/A (up/down buttons): control the gripper open/close.
# X/B (left/right buttons): control the gripper wrist rotate/left right.
#
# Pick / Place / Push Random:
# Top Left button: move to random pick angle (within range) and place, raise a bit
# Top Right button: move to previous pick angle/rot and pick up an object (same rot)
# Top Left trigger: move to left of previous object position and push right
# Top right trigger: move to right of previous object position and push left
#
# from https://gist.github.com/rdb/8864666

class Joystick():
    '''
    An interface to a physical joystick available at /dev/input
    '''
    def __init__(self, dev_fn='/dev/input/js0', debug=False):
        self.axis_states = {}
        self.button_states = {}
        self.axis_map = []
        self.button_map = []
        self.jsdev = None
        self.dev_fn = dev_fn
        self.debug = debug
        self.pressed_button = None
        self.pressed_button_value = None
        self.js_event_timeout = False
        init = True
        
        # These constants were borrowed from linux/input.h
        self.axis_names = {
            0x00 : 'y',
            0x01 : 'x',
            0x02 : 'lz',
            # 0x03 : 'wrist angle',  # automatic up/down
            0x03 : 'w',
            0x04 : 'z',
            0x05 : 'rz',
            0x06 : 'throttle',
            0x07 : 'rudder',
            0x08 : 'wheel',
            0x09 : 'gas',
            0x0a : 'brake',
            0x10 : 'moveArm',
            0x11 : 'moveArm2',
            0x12 : 'hat1x',
            0x13 : 'hat1y',
            0x14 : 'hat2x',
            0x15 : 'hat2y',
            0x16 : 'hat3x',
            0x17 : 'hat3y',
            0x18 : 'pressure',
            0x19 : 'distance',
            0x1a : 'tilt_x',
            0x1b : 'tilt_y',
            0x1c : 'tool_width',
            0x20 : 'volume',
            0x28 : 'misc',
        }

        self.button_names = {
            0x120 : 'trigger',
            0x121 : 'thumb',
            0x122 : 'thumb2',
            0x123 : 'top',
            0x124 : 'top2',
            0x125 : 'pinkie',
            0x126 : 'base',
            0x127 : 'base2',
            0x128 : 'base3',
            0x129 : 'base4',  
            0x12a : 'base5',  
            0x12b : 'base6',  
           
            #PS3 six axis specific
            0x12c : 'triangle',
            0x12d : "circle",
            0x12e : "cross",
            0x12f : 'square',

            0x130 : 'wrist rotate right',
            0x131 : 'gripper open',
            0x132 : 'c',
            0x133 : 'gripper close',
            0x134 : 'wrist rotate left',
            0x135 : 'tz',
            # 0x136 : 'tl',
            # 0x137 : 'tr',
            # 0x136 : 'move to point',
            0x136 : 'auto-pick',
            # 0x137 : 'move from {1}',
            0x137 : 'auto-place',
            # Top left/right trigger: auto-push
            # 0x138 : 'tl2',
            0x138 : 'auto-push-right',
            # 0x139 : 'tr2',
            0x138 : 'auto-push-left',
            # 0x139 : 'tr2',
            0x13a : 'relaxServos',
            0x13b : 'start',
            # 0x13c : 'torqueServos',
            0x13c : 'delete-last-recording',
            # vibration
            # 0x13d : 'delete-last-recording',
            0x13e : 'thumbr',

            0x220 : 'dpad_up',
            0x221 : 'dpad_down',
            0x222 : 'dpad_left',
            0x223 : 'dpad_right',

            # XBox 360 controller uses these codes.
            0x2c0 : 'dpad_left',
            0x2c1 : 'dpad_right',
            0x2c2 : 'dpad_up',
            0x2c3 : 'dpad_down', }


    def js_init(self):
        from fcntl import ioctl
        '''
        call once to setup connection to /dev/input/js0 and map buttons
        '''

        # Open the joystick device.
        print('Opening %s...' % self.dev_fn)
        # blocking:
        self.jsdev = open(self.dev_fn, 'rb')
        # nonblocking:
        # self.jsdev = os.open(self.dev_fn, os.O_RDONLY|os.O_NONBLOCK)
    
        # Get the device name.
        buf = array.array('B', [0] * 64)
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf) # JSIOCGNAME(len)
        self.js_name = buf.tobytes().decode('utf-8')
        print('Device name: %s' % self.js_name)

        # Get number of axes and buttons.
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf) # JSIOCGAXES
        self.num_axes = buf[0]

        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf) # JSIOCGBUTTONS
        self.num_buttons = buf[0]

        # Get the axis map.
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf) # JSIOCGAXMAP

        for axis in buf[:self.num_axes]:
            axis_name = self.axis_names.get(axis, 'unknown(0x%02x)' % axis)
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0

        # Get the button map.
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf) # JSIOCGBTNMAP

        for btn in buf[:self.num_buttons]:
            btn_name = self.button_names.get(btn, 'unknown(0x%03x)' % btn)
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0

        os.set_blocking(self.jsdev.fileno(), False)
        return True

    def init(self):
        self.js_init()

    def show_map(self):
        '''
        list the buttons and axis found on this joystick
        '''
        print ('%d axes found: %s' % (self.num_axes, ', '.join(self.axis_map)))
        print ('%d buttons found: %s' % (self.num_buttons, ', '.join(self.button_map)))


    def poll(self):
        '''
        query the state of the joystick, returns button which was pressed, if any,
        and axis which was moved, if any. button_state will be None, 1, or 0 if no changes, 
        pressed, or released. axis_val will be a float from -1 to +1. button and axis will 
        be the string label determined by the axis map in init.
        '''
        self.pressed_button = None
        self.pressed_button_value = None
        button = None
        button_state = None
        axis = None
        axis_val = None
        accept_js_event = True

        start_tm = datetime.datetime.now()
        while True:
          # try:
          if True:
            evbuf = self.jsdev.read(8)
            if evbuf:
                tval, value, typev, number = struct.unpack('IhBB', evbuf)
    
                # if typev & 0x80:
                    #ignore initialization event
                    # return button, button_state, axis, axis_val
                if typev & 0x80:
                    continue
  
                if typev & 0x01:
                    button = self.button_map[number]
                    if button:
                        self.button_states[button] = value
                        # button_state = value
                        if value:
                            # print("%s pressed" % (button))
                            self.pressed_button = button
                            self.pressed_button_value = value
                        else:
                            # print("%s released" % (button))
                            self.pressed_button = None
                            self.pressed_button_value = None
    
    
                if typev & 0x02:
                    axis = self.axis_map[number]
                    if axis:
                        fvalue = value / 32767.0
                        self.axis_states[axis] = fvalue
                        axis_val = fvalue
          # except:
                accept_js_event = True
            # keep looping until js event or js event timeout
  
          # if no event, check for joystick event timeout
          if (self.pressed_button is None and self.pressed_button_value is None and 
                axis is None and axis_val is None):
                # no js action; keep usb busy if necessary.
                poll_tm = datetime.datetime.now()
                delta = poll_tm - start_tm
                if (delta.total_seconds() > 1.0):
                      print("delta secs:", delta.total_seconds())
                      # fall through to "update() to perform a "none" action
                      self.js_event_timeout = True
                      break
          # process legal joystick event
          else:
              # only consider expected js events
              if (self.pressed_button not in 
                  ['start', 'auto-pick', 'auto-push-left', 'auto-push-right',
                   'relaxServos', 'torqueServos', 'delete-last-recording', 'wrist rotate left',
                   'wrist rotate right', 'gripper close', 'gripper open'] and 
                  axis not in ['x', 'y', 'z', 'wrist angle', 'moveArm', 'moveArm2']):
                    # print(self.pressed_button, self.pressed_button_value, axis, axis_val)
                    continue
              if accept_js_event:
                # ignore the buffered events before the poll began
                break
              else:
                # process js action
                continue
        # end while True:
        # return button, button_state, axis, axis_val
        return self.pressed_button, self.pressed_button_value, axis, axis_val


class JoystickController(object):
    '''
    Joystick client using access to local physical input
    '''

    def __init__(self, poll_delay=0.0,
                 dev_fn='/dev/input/js0',
                 debug = False):

        # ARD & another snapshot commented below
        self.robot_camera = camera_snapshot.CameraSnapshot()
        self.poll_delay = poll_delay
        self.recording = False
        self.dev_fn = dev_fn
        self.js = None
        self.started = False
        self.debug = debug
        while not self.init_js():
          time.sleep(1)
        self.joystick_threshold = 10
        self.wrist_angle = 0
        # self.wrist_rot_vel = 0
        self.wrist_rotation_velocity = 0
        self.open_close = 1      # open when started
        self.running = False
        self.recording = False
        # self.move_mode = "Relative"
        self.move_mode = None
        self.action_val     = {}
        self.last_auto_pick = {}
        self.last_auto_push = {}
        self.prev_auto_angle = 0
        self.poll_cnt = 0
        self.last_episode_file = None
        #We expect that the framework for parts will start a new
        #thread for our update fn. We used to do that and it caused
        #two threads to be polling for js events.

        # do handshake with WidowX
        self.widowx = widowx.WidowX()
        print("WX")
        self.running = True
        self.started = True
        self.config = self.read_config()
        # 0.196 vs 0.785
        self.MAX_SWIVEL = math.pi / 16
        # self.MAX_SWIVEL = atan2(1.75, 1.75)
        # print("MAX_SWIVEL:", self.MAX_SWIVEL)
        # self.DELTA_ACTION = 1.75
        self.DELTA_ACTION = .1
        self.DELTA_SERVO = 20
        self.DELTA_GRIPPER = 10
        self.DELTA_ANGLE   = math.pi / 30
        self.GRIPPER_CLOSED = 0b10
        self.GRIPPER_OPEN   = 0b01
        self.gripper_fully_open_closed = None

    def init_js(self):
        '''
        attempt to init joystick
        '''
        try:
            self.js = Joystick(self.dev_fn, self.debug)
            self.js.init()
        except FileNotFoundError:
            print(self.dev_fn, "not found.")
            self.js = None
        return self.js is not None

    def read_config(self):
        with open('widowx_config.json') as config_file:
          config_json = config_file.read()
        config = json.loads(config_json)
        return config

    def get_initial_state(self):
        self.config = self.read_config()
        init_state = json.loads(self.config["initial_state"])
        print("init_state", init_state)
        return init_state

    # Dataset of trajectories where each step has the following fields:
    #  - observation:
    #      - image_{name1, name2, ...} # RGB image observations
    #      - depth_{name1, name2, ...} # depth image observations
    #      - proprio                   # 1-dimensional array of proprioceptive observations
    #      - timestep                  # timestep of each frame
    #  - task:
    #      - language_instruction      # language instruction, present if `language_key` is provided
    #    - action                        # action vector
    #    - dataset_name                  # name of the dataset

    # add timestep info

    def episode_start(self):
        return # ARD
        if not self.recording:
          return
        self.episode = {}

    # ~/google_rt/octo/tests/debug_dataset/bridge_dataset/1.0.0/features.json
    # step: <_VariantDataset element_spec= {
    # 'action': TensorSpec(shape=(7,), dtype=tf .float32, name=None),
    # 'discount': TensorSpec(shape=(), dtype=tf.float32, name=None),
    # 'is_first': TensorSpec(shape=(), dtype=tf.bool, name=None),
    # 'is_last': TensorSpec(shape=(), dtype=tf.bool, name=None),
    # 'is_terminal': TensorSpec(shape=(), dtype=tf.bool, name=None),
    # 'language_embedding': TensorSpec(shape=(512,), dtype= tf.float32, name=None),
    # 'language_instruction': TensorSpec(shape=(), dtype=tf.string, name=None),
    # 'observation':
    #    {'image_0': TensorSpec(shape=(256, 256, 3), dty pe=tf.uint8, name=None),
    #    'image_1': TensorSpec(shape=(256, 256, 3), dtype=tf.uin t8, name=None),
    #    'image_2': TensorSpec(shape=(256, 256, 3), dtype=tf.uint8, name= None),
    #    'image_3': TensorSpec(shape=(256, 256, 3), dtype=tf.uint8, name=None),
    #    'state': TensorSpec(shape=(7,), dtype=tf.float32, name=None)},
    # 'reward': TensorSpec(shape=(), dtype=tf.float32, name=None)}>
   
    #   traj_step ?
    def episode_step(self, first=False, last=False):
        return # ARD
        if not self.recording:
          return
        if len(self.action_val) == 0:
          return
        # im, im_file, im_time = self.robot_camera.snapshot(True)
        ep = {}
        # ep['action'] = copy.deepcopy(self.action_val)
        # "ACTION_DIM_LABELS = ['X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll', 'Grasp']\n",
        ep["action"] = {}
        ep["action"]["world_vector"] = [self.action_val['X'], self.action_val['Y'], self.action_val['Z']] 
        ep["action"]["rotation_delta"] = [self.action_val['Pitch'], self.action_val['Roll']]
        ep["action"]["gripper_closedness_action"] = [self.action_val['Grasp']]
        ep['is_first'] = first
        ep['is_last'] = last
        ep['is_terminal'] = False
        ep['language_embedding'] = 'language_instruction'
        ep['language_instruction'] = self.config['language_instruction'] 
        ep['observation'] = {}
        ep['observation']['image'] = copy.deepcopy(im)
        ep['observation']['state'] = copy.deepcopy(self.widowx.state)
        ep['reward'] = None
        self.episode.append(ep)
        # traj["task"]["timestep"] = len(self.episode)
        
    def episode_end(self):
        return # ARD
        if not self.recording:
          return
        if len(self.action_val) == 0:
          return
        if len(self.episode) == 0:
          return
        # Serializing json
        json_object = json.dumps(self.episode, indent=4)
        # Bridgedata filename:
        # "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
        filenm = self.config["dataset"] + '/' + "ep" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ".json"
        self.last_episode_file = filenm
        with open(filenm, "w") as outfile:
          outfile.write(json_object)

    def remove_episode(self):
        print("remove episode", self.last_episode_file)
        if (self.last_episode_file is not None):
          os.remove(self.last_episode_file)

    def drop_down(self):
        print("drop down ")
        [success,err_msg] = self.set_move_mode('Absolute')
        self.action(vz=0)

    def lift_up(self):
        print("lift up ")
        init_state = json.loads(self.config["initial_state"])
        self.set_move_mode('Absolute')
        self.action(vz = init_state['z'])

    def gripper(self, o_c):
        # print("gripper: ", o_c)
        if self.widowx.state['Gripper'] == self.GRIPPER_CLOSED:  
          return
        if not self.widowx.state['Gripper'] == self.GRIPPER_OPEN: 
          return
        elif self.move_mode == 'Relative': # use relative values
             self.action(goc=o_c)
        elif self.move_mode == 'Absolute': # use absolute values
             self.action(goc=o_c)
        self.open_close = self.widowx.state['Gripper']
        return [True,None] 

    def wrist_rotate(self,angle):
        print("wrist_rotate: ", angle)
        self.set_move_mode('Absolute')
        self.action(vr=angle)

    # swivel relative to current position
    def do_swivel(self,left_right):
        [success,err_msg] = self.set_move_mode('Absolute')
        if not success and err_msg == "RESTART_ERROR":
          return [success,err_msg]

        # curr_angle = math.atan2(self.widowx.state["y"], self.widowx.state["x"])
        # new_angle = curr_angle + angle
        self.action(swivel=left_right)

    def action(self, vx=None, vy=None, vz=None, vg=None, vr=None, goc=None, swivel=None):
        if self.move_mode == 'Absolute':
          # compute point action based on initial_state:
          # {\"x\":20, \"y\":0, \"z\":12, \"gamma\":-254, \"rot\":0, \"gripper\":1}"
          # then move x/y so that it matches the value under "gamma"
          # init_pose = self.get_initial_state()
          orig_pose = copy.deepcopy(self.widowx.state)
          prev_pose = copy.deepcopy(self.widowx.state)
          delta_action_performed = True
          while delta_action_performed: 
            delta_action_performed = False
            if vx is None and vy is None and swivel is not None: 
              # find
              x0 = orig_pose['X']
              y0 = orig_pose['Y']
              radius = math.sqrt(math.pow(x0,2) + math.pow(y0,2))
              # radius = round(round(radius*3) / 3.0, 4)
              curr_angle = math.atan2(self.widowx.state['Y'], self.widowx.state['X'])
              # swivel is desired angle (absolute, not relative)
              # compute a swivel detectable by WidowX.cpp
              print("radius, x3,y0, angle: ",radius,x0,y0, curr_angle)
              # x_angle = math.acos(x0/radius)
              if swivel == "LEFT":
                curr_angle += self.DELTA_ANGLE
              elif swivel == "RIGHT":
                curr_angle -= self.DELTA_ANGLE
              x = math.cos(curr_angle) * radius
              y = math.sin(curr_angle) * radius
              print("SWIVEL:",x,y)
            elif vx is not None or vy is not None:
              # x = self.widowx.state['X']
              x = orig_pose['X']
              if vx is not None and vx != x:
                if abs(vx - x) > self.DELTA_ACTION:
                  if vx > x:
                    x = x + self.DELTA_ACTION
                  else:
                    x = x - self.DELTA_ACTION
                  # delta_action_performed = True
                else:
                  x = vx
              # y = self.widowx.state['Y']
              y = orig_pose['Y']
              if vy is not None and vy != y:
                if abs(vy - y) > self.DELTA_ACTION:
                  if vy > y:
                    y = y + self.DELTA_ACTION
                  else:
                    y = y - self.DELTA_ACTION
                  # delta_action_performed = True
                else:
                  y = vy 
            else:
                # x = self.widowx.state['X']
                # y = self.widowx.state['Y']
                x = orig_pose['X']
                y = orig_pose['Y']

            # z = self.widowx.state['Z']
            z = orig_pose['Z']
            if vz is not None and vz != z:
                if abs(vz - z) > self.DELTA_ACTION:
                  if vz > z:
                    z = z + self.DELTA_ACTION
                  else:
                    z = z - self.DELTA_ACTION
                  # delta_action_performed = True
                else:
                  z = vz 
       
            # r = self.widowx.state['Rot']
#            r = orig_pose['Rot']
#            if vr is not None and vr != r:
#                if abs(vr - r) > self.DELTA_ACTION:
#                  if vr > r:
#                    r = r + self.DELTA_ACTION
#                  else:
#                    r = r - self.DELTA_ACTION
#                  # delta_action_performed = True
#                else:
#                  r = vr 
            r = vr 

            # gamma = self.widowx.state['Gamma']
            gamma = orig_pose['Gamma']
            if vg is not None and vg != gamma:
                gamma = vg
                # delta_action_performed = True

            # g = self.widowx.state['Gripper']
            g = orig_pose['Gripper']
            if goc is not None and goc != g:
                g = goc
                # delta_action_performed = True

            self.move(x, y, z, gamma, r, g) # absolute "To Point" movement
            self.action_val = {'mode':'absolute', 'X':x, 'Y':y, 'Z':z, 'Yaw':0, 'Pitch':gamma, 'Roll':r, 'Grasp':g}
            print("action:", self.action_val)
            # See if requested action actually happened
            if (self.widowx.state['X'] == prev_pose['X'] and
                self.widowx.state['Y'] == prev_pose['Y'] and
                self.widowx.state['Z'] == prev_pose['Z'] and
                self.widowx.state['Gamma'] == prev_pose['Gamma'] and
                self.widowx.state['Rot'] == prev_pose['Rot'] and
                self.widowx.state['Gripper'] == prev_pose['Gripper']):
                print("JS: ARM DIDN'T MOVE")
                # delta_action_performed = False
            else:
                # same_pos = "same pose: "
                delta_pos = "arm moved: "
                if (self.widowx.state['X'] != prev_pose['X']):
                  # same_pos += "x"
                  delta_pos += "x:" + str(self.widowx.state['X'] - prev_pose['X'])
                if (self.widowx.state['Y'] != prev_pose['Y']):
                  # same_pos += " y"
                  delta_pos += " y:" + str(self.widowx.state['Y'] - prev_pose['Y'])
                if (self.widowx.state['Z'] != prev_pose['Z']):
                  # same_pos += " z"
                  delta_pos += " z:" + str(self.widowx.state['Z'] - prev_pose['Z'])
                if (self.widowx.state['Gamma'] != prev_pose['Gamma']):
                  # same_pos += " g"
                  delta_pos += " g:" + str(self.widowx.state['Gamma'] - prev_pose['Gamma'])
                if (self.widowx.state['Rot'] != prev_pose['Rot']):
                  # same_pos += " r"
                  delta_pos += " r:" + str(self.widowx.state['Rot'] - prev_pose['Rot'])
                if (self.widowx.state['Gripper'] != prev_pose['Gripper']):
                  # same_pos += " oc"
                  delta_pos += " oc:" + str(self.widowx.state['Gripper'])
                # print(same_pos)
                print(delta_pos)
                print("servo:", self.widowx.current_position)
                print("widow state", self.widowx.state)
                print("prev  pose ", prev_pose)
                prev_pose = copy.deepcopy(self.widowx.state)
                delta_action_performed = False
            self.episode_step()
            delta_action_performed = False

        # "ACTION_DIM_LABELS = ['X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll', 'Grasp']\n",
        else:
          # None means no relative "By Point" movement
          if vx is None: vx = 0
          if vy is None: vy = 0
          if vz is None: vz = 0
          if vg is None: vg = 0
          if vr is None: vr = 0
          if goc is None: goc = 0
          # x,y,z are relative.  Need to normalize.
          # No longer need to reduce to byte commands for microprocessor
          # vx = min(max(-127, round(vx * 127.0 / 1.75)), 127)
          # vy = min(max(-127, round(vy * 127.0 / 1.75)), 127)
          # vz = min(max(-127, round(vz * 127.0 / 1.75)), 127)
          # vg = min(max(-255, round(vg * 255.0 / 1.4)), 255)
          # vr = min(max(-255, round(vr * 255.0 / 1.4)), 255)
          self.move(vx, vy, vz, vg, vr, goc)
          self.action_val = {'mode':'relative', 'X':vx, 'Y':vy, 'Z':vz, 'Yaw':0, 'Pitch':vg, 'Roll':vr, 'Grasp':goc}
          print("action:", self.action_val)
          self.episode_step()
        self.widowx.getState()

##########################
# Code shared with joystick & RT1 script (by copying)
##########################
    # a cross-over interface between joystick & widowx.py, deals with move_mode
    def move(self, vx, vy, vz, vg, vr, goc):
        initial_time = self.widowx.millis()
        if (vr and self.move_mode == 'Absolute'):
            self.widowx.moveServoWithSpeed(4, vr, initial_time)
        elif vr != None:
            print("move vr:", vr, self.widowx.current_angle[self.widowx.IDX_ROT])
            ax12pos = vr + 512
            self.widowx.moveServo2Position(self.widowx.IDX_ROT, ax12pos)  # servo id 5 / idx 4: rotate gripper to angle
        if goc != None and goc != 0:
            self.widowx.openCloseGrip(goc)
        if (vx or vy or vz or vg):
          if (self.move_mode == 'Relative'):
            fvx = min(max(-1.75, (float(vx) / 127.0 * 1.75)), 1.75)
            fvy = min(max(-1.75, (float(vy) / 127.0 * 1.75)), 1.75)
            fvz = min(max(-1.75, (float(vz) / 127.0 * 1.75)), 1.75)
            fvg = min(max(-1.4, (float(vg) / 255.0 * 1.4)), 1.4)
            self.widowx.movePointWithSpeed(fvx, fvy, fvz, fvg, initial_time)
            self.gripper_fully_open_closed = self.widowx.moveGrip(goc)
          elif (self.move_mode == 'Absolute'):
            self.widowx.moveArmGammaController(vx, vy, vz, vg)
            angle = (float(vr) / 256.0) * math.pi
            self.widowx.moveServoWithSpeed(self.widowx.IDX_ROT, angle, initial_time)
            self.gripper_fully_open_closed = self.widowx.openCloseGrip(goc)
          # elif (moveOption == self.USER_FRIENDLY):
            # used by joystick controller only
            # self.moveArmWithSpeed(vx, vy, vz, vg, initial_time)

    def set_move_mode(self, mode):
        # AKA:      absolute point         relative point
        if mode != 'Absolute' and mode != 'Relative':
          print("illegal move mode:", mode)
          return [True, None]
        if self.move_mode != mode:
          self.move_mode = mode
          return [True, None]
        return [True, None]
##########################

    def update(self):
        # for i in range(5):
        #   print("M2P")
        #   [success,err_msg] = self.widowx.moveRest()
        # return
        '''
        poll a joystick for input events

        button map name => PS3 button => function
        '''

#        while self.running and not self.init_js():
#            print("waiting for joystick to be online")
#            time.sleep(5)
#        print("update", self.running)
#        #wait for joystick to be online

        vx = vy = vz = vgamma = vq5 = 0.0
        vwa = vwr = 0
        button = axis = None
        while self.running:
            prev_button, prev_axis = button, axis
            # print("poll()")
            button, button_state, axis, axis_val = self.js.poll()
            self.poll_cnt += 1
            if button is not None:
              print("button:", button)
              print("button_state:", button_state)
            # Too many print outs of unimportant values. Comment out:
            # if axis is not None:
            #   print("axis:", axis)
            #   print("axis val:", axis_val)
        
            if button == 'start' and button_state == 1:
                # start / stop arm
                if not self.recording:
                    self.recording = True
                    print("recording started!")
                    self.widowx.moveRest()
                    self.episode_start()
                    self.episode_step()
                else:
                    self.episode_end()
                    self.recording = False
                    print("recording stopped.")
                    print("ready")
                    self.action()
                    # time.sleep(1)

            elif not self.started:
                print("robotic arm not started")

            elif axis == 'moveArm2':
                if axis_val == -1:
                  print("movePick")
                  self.widowx.moveArmPick()
                  self.gripper(self.GRIPPER_OPEN)
                elif axis_val == 1:
                  print("moveRest")
                  self.widowx.moveRest()
                  self.gripper(self.GRIPPER_OPEN)
            elif axis == 'moveArm':
                if axis_val == 1:
                  print("RotRight")
                  self.do_swivel("RIGHT")
                elif axis_val == -1:
                  print("RotLeft")  
                  self.do_swivel("LEFT")
            elif button == 'auto-pick' and button_state == 1:
                  # first, line up the gripper to right position
                  # or start with y = 0. Then do auto-pick, which
                  # does a straight-down pick. auto-place does a
                  # random placement.
                  #
                  # After the first drop, if the nextmovement is
                  # auto-pick, then start from y=0 and swivel to
                  # the last drop-off and then go straight-down.
                  #
                  # Use vibration button to erase unsuccessful actions.

                  print("auto-pick")  # TODO: pickNdrop Home/Trash
                  self.episode_start()
                  # move to previous drop off point to central location
                  if prev_button == 'auto-place':
                    self.set_move_mode('Absolute')
                    self.action(swivel = self.prev_place_angle) 
                  self.gripper(self.GRIPPER_OPEN)
                  self.drop_down() 
                  self.gripper(self.GRIPPER_CLOSED)
                  self.lift_up() 
            elif button == 'auto-place' and button_state == 1:
                  print("auto-place")
                  if prev_button == 'auto-pick':
                    self.set_move_mode('Absolute')
                    self.action(swivel = self.prev_place_angle) 
                    self.prev_auto_angle = self.widowx.state['Gripper']
                  self.drop_down() 
                  self.gripper(self.GRIPPER_OPEN)
                  self.lift_up() 
            elif button == 'auto-push-left' and button_state == 1:
                  print("auto-push-left")
                  self.set_move_mode('Absolute')
                  if prev_button == 'auto-push-right':
                    self.do_swivel(self.prev_auto_angle + math.pi / 16) 
                  else:
                    self.prev_auto_angle = - math.pi * 3 / 16 
                    self.do_swivel(self.prev_auto_angle)
                  self.wrist_rotate(128)
                  self.drop_down() 
                  self.do_swivel(self.prev_auto_angle - math.pi / 4) 
                  self.prev_auto_angle = self.do_swivel(self.prev_auto_angle - math/pi / 4) 
                  self.lift_up() 
            elif button == 'auto-push-right' and button_state == 1:
                  print("auto-push-left")
                  self.set_move_mode('Absolute')
                  if prev_button == 'auto-push-left':
                    self.do_swivel(self.prev_auto_angle - math.pi / 16)
                  else:
                    self.do_swivel(math.pi * 3 / 16)
                  self.wrist_rotate(128)
                  self.drop_down() 
                  self.do_swivel(self.prev_auto_angle + math.pi / 4) 
                  self.prev_auto_angle = swivel(self.prev_auto_angle - math/pi / 4)


            elif button == 'relaxServos' and button_state == 1:
                  print("relaxServos")
                  self.widowx.relaxServos()
                  self.episode_step()
            elif button == 'torqueServos' and button_state == 1:
                  print("torqueServos")
                  self.widowx.torqueServos()
                  self.episode_step()
            elif button == 'delete-last-recording' and button_state == 1:
                  print("Remove Previous Episode")
                  self.remove_episode()
            # bug: axis /button events that we don't care about end up being processed here.
            # bug: axis /button events that we don't care about should be ignored.
            elif axis is not None or button is not None:
                # print("Move Arm:", axis, button)
                # vwa = vwr = 0
                vx = vy = vz = vgamma = vg_rot = o_c = None
                if button == 'wrist rotate left' and button_state == 1:
                  if self.wrist_rotation_velocity >= 0:
                    self.wrist_rotation_velocity -= 20
                  elif -255 < self.wrist_rotation_velocity < 0:
                    self.wrist_rotation_velocity -= 20
                  print("wrist rotation velocity", self.wrist_rotation_velocity)
                  vg_rot = self.wrist_rotation_velocity
                elif button == 'wrist rotate right' and button_state == 1:
                  if self.wrist_rotation_velocity <= 0:
                    self.wrist_rotation_velocity += 20
                  elif 255 > self.wrist_rotation_velocity > 0:
                    self.wrist_rotation_velocity += 20
                  print("wrist rotation velocity", self.wrist_rotation_velocity)
                  vg_rot = self.wrist_rotation_velocity
                elif ((button == 'wrist rotate left' and button_state == 0)
                   or (button == 'wrist rotate right' and button_state == 0)):
                  pass
                else:
                  self.wrist_rotation_velocity = 0

                if button == 'gripper close' and button_state == 1:
                    self.open_close = self.GRIPPER_CLOSED
                    print("close gripper")
                elif button == 'gripper open' and button_state == 1:
                    self.open_close = self.GRIPPER_OPEN
                    print("open gripper")
                else:
                    self.open_close = 0

                xy_radius = math.sqrt(math.pow(self.js.axis_states['x'],2)+math.pow(self.js.axis_states['y'],2))
                if axis == 'x' and xy_radius > 0.99 and abs(axis_val) >= .7:
                    # vx speed values range from [-127,127]
                    vx = -1 * round(axis_val * 127)
                    # if axis_val > 0:  # vx is pointing right, go down by reducing vy.
                    #   vx *= -1 
                    # if abs(vx) < self.joystick_threshold:
                    #   vx = 0
                    print("xy_radius, axis_states:", xy_radius, self.js.axis_states)
                if axis == 'y' and xy_radius > 0.99 and abs(axis_val) >= .7:
                    # vy speed values range from [-127,127]
                    vy = -1 * round(axis_val * 127)
                    # if axis_val > 0:  # vy is pointing down, go down by reducing vy.
                    #   vy *= -1 
                    # if abs(vy) < self.joystick_threshold:
                    #   vy = 0
                    print("vy", vy)
                    print("xy_radius, axis_states:", xy_radius, self.js.axis_states)

                if axis == 'z' and abs(axis_val) > 0.99:
                    # One of the following is true:
                    # vz speed values range from [-127,127]
                    vz = round(-1 * axis_val * 127)
                    # if axis_val > 0:  # vz is pointing down, go down by reducing vz.
                    #    vz *= -1 
                    if abs(vz) < self.joystick_threshold:
                      vz = 0
                    print("vz", vz)

                # for now, preferred "wrist angle" is set to -254
                if axis == 'wrist angle':
                  # vwa = round(axis_val * 512)
                  # vwa = round(axis_val * 255)
                  # if abs(vwa) < self.joystick_threshold * 2:
                  # if abs(vwa) < self.joystick_threshold:
                  #   vwa = 0 
                  vwa = -254
                  print("wrist angle velocity", vwa)

                  # Gamma is the desired angle of the gripper. For example,
                  # gamma = pi/2 will make the arm vertical for Pick N Drop.
                  # range is -255, +255
                  vgamma = vwa
                  print("vgamma, vq5", vgamma, vq5)
                else:
                  # vgamma = 0
                  # vgamma = -254
                  pass

                # vg_rot = self.wrist_rotation_velocity

                if not (vx is None and vy is None and vz is None and
                    vgamma is None and vg_rot is None and self.open_close == 0):
                  # self.widowx.move(vx, vy, vz, vgamma, vg_rot, self.open_close)
                  # self.episode_step()
                  self.set_move_mode('Relative')
                  print("button",button, button_state, "axis", axis, axis_val )
                  print("vx",vx, "vy",vy, "vz",vz, "vg", vgamma, "vr", vg_rot, "oc",self.open_close)
                  self.action(vx, vy, vz, vgamma, vg_rot, self.open_close)
                elif False and self.js.js_event_timeout:
                  # No longer required b/c no more flakey old arbotix board
                  self.js.js_event_timeout = False
                  # make the same (do nothing); need to keep the serial
                  # link busy to prevent the usb from resetting.
                  poll_checks = 1
                  print("JS timeout")
                  if self.move_mode == 'Absolute': # use absolute values
                      if axis is None:
                        print("widowx.state", self.widowx.state)
                        self.action(self.widowx.state['X'], self.widowx.state['Y'], 
                           self.widowx.state['Z'], self.widowx.state['Gamma'], 
                           self.widowx.state['Rot'], self.widowx.state['Gripper'])
                  elif self.move_mode == 'Relative': # use relative values
                      # all Nones
                      # self.action(vx, vy, vz, vgamma, vg_rot, self.open_close)
                      if axis is None:
                        self.action()
                  else:
                      self.set_move_mode("Relative")
          
if __name__ == '__main__':
    jsc = JoystickController(0.0, '/dev/input/js0', False)
    jsc.update()
