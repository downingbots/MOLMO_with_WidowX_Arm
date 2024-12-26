##########################
# Code shared with joystick & RT1 script (by copying)
##########################
# a cross-over interface between joystick & widowx.py, deals with move_mode
import widowx
import json
import math
import copy
from util_radians import *

class widowx_client():

    def __init__(self):
        # do handshake with WidowX
        self.widowx = widowx.WidowX()
        print("WX")
        self.running = True
        self.started = True
        self.config = self.read_config()
        self.move_mode = 'Relative'
        # self.move_mode = 'Absolute'
        # 0.196 vs 0.785
        self.MAX_SWIVEL = math.pi / 16
        # self.MAX_SWIVEL = atan2(1.75, 1.75)
        # print("MAX_SWIVEL:", self.MAX_SWIVEL)
        # self.DELTA_ACTION = 1.75
        self.DELTA_ACTION = .1
        self.DELTA_SERVO = 20
        self.DELTA_GRIPPER = 10
        self.DELTA_ANGLE   = math.pi / 30
        self.DELTA_ROT_ANGLE = math.pi / 100
        self.GRIPPER_CLOSED = 0b10
        self.GRIPPER_OPEN   = 0b01
        self.GRIPPER_ROT_TO_CAMERA   = -175
        self.gripper_fully_open_closed = None
        self.gripper_pos_open = None   # servo position
        self.gripper_pos_closed = None   # servo position

    def read_config(self):
        with open('widowx_config.json') as config_file:
          config_json = config_file.read()
        self.config = json.loads(config_json)
        return self.config

    def moveOutCamera(self):
        self.widowx.moveRest()
        gamma = -(self.widowx.MX_MAX_POSITION_VALUE - 1)
        wristAngle = -(gamma / self.widowx.MX_MAX_POSITION_VALUE) * (math.pi / 2.0)
        self.widowx.moveServo2Angle(3, wristAngle)
        self.widowx.delay(300)

    def moveRest(self):
        self.widowx.moveRest()

    def moveArmPick(self):
        self.widowx.moveArmPick()

    def is_grasping(self):
        return self.widowx.is_grasping

    def set_move_mode(self, mode):
        # AKA:      absolute point         relative point
        if mode != 'Absolute' and mode != 'Relative':
          print("illegal move mode:", mode)
          return [True, None]
        if self.move_mode != mode:
          self.move_mode = mode
          return [True, None]
        return [True, None]

    def state(self):
        return self.widowx.state

    def gripper(self, o_c):
        print("gripper: ", o_c, self.widowx.state['Gripper'])
        if o_c not in [self.GRIPPER_OPEN, self.GRIPPER_CLOSED]:
          print("ERROR: use GRIPPER_OPEN or GRIPPER_CLOSED")
          return [False,None]
        if self.widowx.state['Gripper'] == o_c:
          pass
        elif self.move_mode == 'Relative': # use relative values
             self.action(goc=o_c)
        elif self.move_mode == 'Absolute': # use absolute values
             self.action(goc=o_c)
        self.open_close = self.widowx.state['Gripper']
        gripper_pos = self.widowx.getServoPosition(self.widowx.IDX_GRIPPER)
        if self.open_close in [self.widowx.GRIPPER_OPEN, self.widowx.GRIPPER_CLOSED]:
          return [True,gripper_pos] 
        return [False,gripper_pos] 

    def wrist_rotate(self,angle):
        orig_move_mode = self.move_mode
        print("wrist_rotate: ", angle)
        # self.set_move_mode('Absolute')
        self.set_move_mode('Relative')
        self.action(vr=angle)
        self.set_move_mode(orig_move_mode)

    # swivel relative to current position
    def do_swivel(self,left_right, delta=None):
        if left_right not in ["LEFT","RIGHT"]:
          print("ERROR: do_swivel", left_right)
          return
        # find
        self.widowx.getState()
        x0 = self.widowx.state['X']
        y0 = self.widowx.state['Y']
        radius = math.sqrt(math.pow(x0,2) + math.pow(y0,2))
        # radius = round(round(radius*3) / 3.0, 4)
        curr_angle = math.atan2(self.widowx.state['Y'], self.widowx.state['X'])
        # swivel is desired angle (absolute, not relative)
        # compute a swivel detectable by WidowX.cpp
        print("radius, x3,y0, angle: ",radius,x0,y0, curr_angle)
        # x_angle = math.acos(x0/radius)
        delta_angle = delta
        if delta is None:
          delta_angle = self.DELTA_ANGLE
        if left_right == "LEFT":
          curr_angle += delta_angle
        elif left_right == "RIGHT":
          curr_angle -= delta_angle
        print("new swivel angle, delta: ", curr_angle, delta_angle)
        print("Swivel :", curr_angle, self.widowx.getServoAngle(self.widowx.IDX_BASE))
        self.widowx.moveServo2Angle(self.widowx.IDX_BASE, curr_angle)  
        self.widowx.delay(300)
        self.widowx.getState()
        success = False
        exp_x = math.cos(curr_angle) * radius
        exp_y = math.sin(curr_angle) * radius
        if (abs(self.widowx.state['X'] - exp_x ) < self.DELTA_ACTION and
            abs(self.widowx.state['Y'] - exp_y ) < self.DELTA_ACTION):
          print("successful swivel:", exp_x, exp_y)
          success = True
        else:
          # try with move interface
          [success,err_msg] = self.set_move_mode('Absolute')
          success = self.action(swivel=curr_angle)
        if not success and err_msg == "RESTART_ERROR":
          print("DO_SWIVEL ERROR:", err_msg)
          return None
        if success:
          return curr_angle
        else:
          return None

    def action_achieved(self, goal_pose):
        # widow.py tries to get within .1. At this level, accept .1 tolerance. 
        if vx >= 24:
          DELTA_ACTION = 2*self.DELTA_ACTION
          MIN_Z_UP = 3.5  # lower than widowx.py to allow for boost failures
        else:
          DELTA_ACTION = self.DELTA_ACTION
          MIN_Z_UP = None
        if (abs(self.widowx.state['X'] - goal_pose['X']) < 2*DELTA_ACTION and
            abs(self.widowx.state['Y'] - goal_pose['Y']) < 2*DELTA_ACTION and
            # with Gravity, Z is harder to get accurate; increase tolerance
            ((MIN_Z_UP is None and abs(self.widowx.state['Z'] - goal_pose['Z']) < 2*DELTA_ACTION)
            or
             (MIN_Z_UP is not None and self.widowx.state['Z'] >= MIN_Z_UP)) and
            # rad_dif(self.widowx.state['Gamma'], goal_pose['Gamma']) < DELTA_ANGLE and
            # abs(self.widowx.state['Rot'] - goal_pose['Rot']) < self.DELTA_ACTION and
            abs(self.widowx.state['Rot'] - goal_pose['Rot']) < self.DELTA_ROT_ANGLE and
            (gripper_pos == self.widowx.getServoPosition(self.widowx.IDX_GRIPPER))):
                print("action_achieved: True ")
                return True
        else:
                print("action_achieved: False")
                return False

    def action(self, vx=None, vy=None, vz=None, vg=None, vr=None, goc=None, swivel=None):
        if self.move_mode == 'Absolute':
          # compute point action based on initial_state:
          # {\"x\":20, \"y\":0, \"z\":12, \"gamma\":-254, \"rot\":0, \"gripper\":1}"
          # then move x/y so that it matches the value under "gamma"
          # init_pose = self.get_initial_state()
          self.widowx.getState()
          VZ_DOWN = self.widowx.MIN_Z_UP
          orig_pose = copy.deepcopy(self.widowx.state)
          orig_vz = vz
          prev_pose = copy.deepcopy(self.widowx.state)
          print("prev pose deepcopy:", prev_pose)
          delta_dist = 0 
          prev_delta_dist = 1000000000
          delta_action_performed = True
          while delta_action_performed: 
            delta_action_performed = False
            doing_swivel = False
            if vx is None and vy is None and swivel is not None: 
              # find current angle
              x0 = orig_pose['X']
              y0 = orig_pose['Y']
              radius = math.sqrt(math.pow(x0,2) + math.pow(y0,2))
              curr_angle = math.atan2(self.widowx.state['Y'], self.widowx.state['X'])
              # swivel is desired angle (absolute, not relative)
              print("raddif curr_angle, swivel_angle", curr_angle, swivel)
              dif_angle = rad_dif(curr_angle, swivel)
              if dif_angle > math.pi:
                # put in +-1 pi format
                print("dif_angle", dif_angle)
                dif_angle -= math.pi
              # success = self.do_swivel(swivel) # gets recursive...
              self.widowx.moveServo2Angle(self.widowx.IDX_BASE, swivel)
              self.widowx.delay(300)
              self.widowx.getState()
              success = False
              exp_x = math.cos(swivel) * radius
              exp_y = math.sin(swivel) * radius
              if (abs(self.widowx.state['X'] - exp_x ) < self.DELTA_ACTION and
                  abs(self.widowx.state['Y'] - exp_y ) < self.DELTA_ACTION):
                print("successful swivel:", exp_x, exp_y)
                success = True
              if success:
                delta_action_performed = False
                continue
              if abs(dif_angle) >= self.DELTA_ANGLE * .9:
                x = math.cos(swivel) * radius
                y = math.sin(swivel) * radius
                print("DOING SWIVEL:",x,y, dif_angle)
                doing_swivel = True
              else:
                x = self.widowx.state['X']
                y = self.widowx.state['Y']
                print("SWIVEL: COMPLETE ", x, y)
            elif vx is not None or vy is not None:
              x = self.widowx.state['X']
              # x = orig_pose['X']
              if vx is not None and vx != x:
                x = vx
              y = self.widowx.state['Y']
              # y = orig_pose['Y']
              if vy is not None and vy != y:
                y = vy 
            else:
                # x = self.widowx.state['X']
                # y = self.widowx.state['Y']
                x = orig_pose['X']
                y = orig_pose['Y']

            z = self.widowx.state['Z']
            if vz is not None and vz != z:
              z = vz 

            # r = self.widowx.state['Rot']
            r = orig_pose['Rot']
            if vr is not None and vr != r:
              r = vr 
            elif vr is None:
              r = None   # don't wrist rotate 

            gamma = orig_pose['Gamma']
            if vg is not None and vg != gamma:
                gamma = vg
                # delta_action_performed = True

            g = orig_pose['Gripper']
            if goc is not None and goc != g:
                g = goc
                # delta_action_performed = True

            self.move(x, y, z, gamma, r, g) # absolute "To Point" movement
            self.action_val = {'mode':'Absolute', 'X':x, 'Y':y, 'Z':z, 'Yaw':0, 'Pitch':gamma, 'Roll':r, 'Grasp':g}
            self.widowx.getState()
            print("action:", self.action_val)
            print("curr state:", self.widowx.state)
            print("prev pose :", prev_pose)
            print("rad_interval Gamma:", rad_interval(self.widowx.state['Gamma'], prev_pose['Gamma']), self.DELTA_ANGLE)
            # See if requested action actually happened

            if x >= 24:
              DELTA_ACTION = 2*self.DELTA_ACTION
              MIN_Z_UP = 3.5   # Lower than widowx.py to allow for boost failures
            else:
              DELTA_ACTION = self.DELTA_ACTION
              MIN_Z_UP = None
            success = False
            if (abs(self.widowx.state['X'] - x) < DELTA_ACTION and
                abs(self.widowx.state['Y'] - y) < DELTA_ACTION and
                ((MIN_Z_UP is None and abs(self.widowx.state['Z'] - z ) < DELTA_ACTION)
                 or
                 (MIN_Z_UP is not None and self.widowx.state['Z'] >= MIN_Z_UP)) and
                # rad_interval(self.widowx.state['Gamma'], gamma) <  self.DELTA_ANGLE and
                (r is None or abs(self.widowx.state['Rot'] - r) < self.DELTA_ROT_ANGLE) and
                self.widowx.state['Gripper'] == g):
                # Arm moved to approximately desired position
                print("ARM MOVE: SUCCESS")
                success = True
            elif (self.widowx.state['X'] == prev_pose['X'] and
                self.widowx.state['Y'] == prev_pose['Y'] and
                self.widowx.state['Z'] == prev_pose['Z'] and
                self.widowx.state['Gamma'] == prev_pose['Gamma'] and
                self.widowx.state['Rot'] == prev_pose['Rot'] and
                self.widowx.state['Gripper'] == prev_pose['Gripper']):
                print("ARM DIDN'T MOVE")
                # delta_action_performed = False
            elif (abs(self.widowx.state['X'] - prev_pose['X']) < DELTA_ACTION and
                  abs(self.widowx.state['Y'] - prev_pose['Y']) < DELTA_ACTION and
                  abs(self.widowx.state['Z'] - prev_pose['Z']) < DELTA_ACTION and
                  rad_interval(self.widowx.state['Gamma'], prev_pose['Gamma']) < self.DELTA_ANGLE and
                self.widowx.state['Rot'] == prev_pose['Rot'] and
                self.widowx.state['Gripper'] == prev_pose['Gripper']):
                print("ARM DIDN'T MOVE MUCH")
                # delta_action_performed = False
            else:
                delta_pos = "arm missed goal by: "
                delta_pos += "x:" + str(self.widowx.state['X'] - x)
                delta_pos += " y:" + str(self.widowx.state['Y'] - y)
                delta_pos += " z:" + str(self.widowx.state['Z'] - z)
                delta_pos += " gam:" + str(self.widowx.state['Gamma'] - gamma) 
                if r is None:
                  delta_pos = " r:" + str(self.widowx.state['Rot'])
                else:
                  delta_pos += " r:" + str(self.widowx.state['Rot'] - r)
                delta_pos += " oc:" + str(self.widowx.state['Gripper'])
                print(delta_pos)
                delta_pos = "arm moved delta:    "
                delta_pos += "x:" + str(self.widowx.state['X'] - prev_pose['X'])
                delta_pos += " y:" + str(self.widowx.state['Y'] - prev_pose['Y'])
                delta_pos += " z:" + str(self.widowx.state['Z'] - prev_pose['Z'])
                delta_pos += " g:" + str(self.widowx.state['Gamma'] - prev_pose['Gamma'])
                delta_pos += " rot:" + str(self.widowx.state['Rot'] - prev_pose['Rot'])
                delta_pos += " oc:" + str(self.widowx.state['Gripper'])
                print(delta_pos)

                print("servo:", self.widowx.current_position)
                print("widow state", self.widowx.state)
                print("prev  pose ", prev_pose)
                prev_pose = copy.deepcopy(self.widowx.state)

                if r is not None and abs(self.widowx.state['Rot'] - r) > self.DELTA_ROT_ANGLE:
                  print("Wrist Rot not correct:", r, self.widowx.state['Rot'])
                  self.widowx.moveServo2Angle(self.widowx.IDX_ROT, r)  
                  self.widowx.delay(300)
                  self.widowx.getState()
                  delta_action_performed = True
                  continue

                if self.widowx.state['Gripper'] != g:
                  print("Gripper not correct:", g, self.widowx.state['Gripper'])
                  gripper_state, gripper_pos = self.widowx.openCloseGrip(goc)
                  if gripper_state == self.widowx.GRIPPER_OPEN:
                      self.gripper_fully_open_closed = gripper_state
                      self.gripper_pos_open = gripper_pos
                  elif gripper_state == self.widowx.GRIPPER_CLOSED:
                      self.gripper_fully_open_closed = gripper_state
                      self.gripper_pos_closed = gripper_pos
                  self.widowx.getState()
                  delta_action_performed = True
                  continue

                delta_dist = abs(self.widowx.state['X'] - x) + abs(self.widowx.state['Y'] - y) + abs(self.widowx.state['Z'] - z)
                print("delta_dist, prev:", delta_dist, prev_delta_dist)
                if delta_dist < prev_delta_dist and abs(z - self.widowx.state['Z']) > DELTA_ACTION:
                  delta_action_performed = True
                  prev_delta_dist = delta_dist 

                # adjust vg
                # compute X,Y dist from origin
                state_radius = math.sqrt(math.pow(self.widowx.state['X'],2) + math.pow(self.widowx.state['Y'],2))
                # determine requested x,y,z,gamma
                requested_radius = math.sqrt(math.pow(x,2) + math.pow(y,2))
                if requested_radius - state_radius > self.DELTA_ACTION:
                  # vg = rad_sum(gamma, self.DELTA_ANGLE) - 2*math.pi
                  vg = rad_dif(gamma, self.DELTA_ANGLE) - 2*math.pi
                  print("Radius too small; decrease vg:", vg)
                elif requested_radius - state_radius < -self.DELTA_ACTION:
                  # vg = rad_dif(gamma, self.DELTA_ANGLE) - 2*math.pi
                  vg = rad_sum(gamma, self.DELTA_ANGLE) - 2*math.pi
                  print("Radius too big; increase vg:", vg)
                elif (z - self.widowx.state['Z'] > self.DELTA_ACTION and
                      not (requested_radius > 24 and z <= VZ_DOWN and 
                      self.widowx.state['Z'] <= VZ_DOWN)):
                  # Note: permits gravity boost failure for near-ground pickup
                  print("Z too low; decrease vg:", vg)
                  # vg = rad_sum(gamma, self.DELTA_ANGLE) - 2*math.pi
                  vg = rad_dif(gamma, self.DELTA_ANGLE) - 2*math.pi
                elif z - self.widowx.state['Z'] < -self.DELTA_ACTION:
                  print("Z too high; increase vg:", vg)
                  # vg = rad_dif(gamma, self.DELTA_ANGLE) - 2*math.pi
                  vg = rad_sum(gamma, self.DELTA_ANGLE) - 2*math.pi
                else:
                  vg = gamma
                vx = x
                vy = y
                vz = z
                if vg != gamma:
                  # print("ACTION FAILED; TRY NEW VG:", vx, vy, vz, vg)
                  print("ACTION FAILED:", vx, vy, vz, vg)
                  delta_action_performed = False
            # self.episode_step()
            # delta_action_performed = False
            self.widowx.getState()

        # "ACTION_DIM_LABELS = ['X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll', 'Grasp']\n",
        else:
          # None means no relative "By Point" movement
          if vx is None: vx = 0
          if vy is None: vy = 0
          if vz is None: vz = 0
          if vg is None: vg = 0
          # if vr is None: vr = 0
          # if goc is None: goc = 0

          # x,y,z are relative.  Need to normalize.
          # No longer need to reduce to byte commands for microprocessor
          # vx = min(max(-127, round(vx * 127.0 / 1.75)), 127)
          # vy = min(max(-127, round(vy * 127.0 / 1.75)), 127)
          # vz = min(max(-127, round(vz * 127.0 / 1.75)), 127)
          # vg = min(max(-255, round(vg * 255.0 / 1.4)), 255)
          # vr = min(max(-255, round(vr * 255.0 / 1.4)), 255)
          if (abs(self.widowx.state['X'] - vx) < self.DELTA_ACTION and
              abs(self.widowx.state['Y'] - vy) < self.DELTA_ACTION and
              abs(self.widowx.state['Z'] - vz) < self.DELTA_ACTION and
              (vg is None or vg == 0) and 
              (vr is None or vr == 0) and 
              (goc is None or goc == 0 or goc == self.widowx.state['Gripper'])):
            print("ALREADY NEAR ABSOLUTE POSITION", vx, vy, vz)
            # success = False
            success = True
          else:
            self.move(vx, vy, vz, vg, vr, goc)
            self.action_val = {'mode':'relative', 'X':vx, 'Y':vy, 'Z':vz, 'Yaw':0, 'Pitch':vg, 'Roll':vr, 'Grasp':goc}
            print("action:", self.action_val)
            success = False
            if (abs(self.widowx.state['X'] - vx) < self.DELTA_ACTION and
                abs(self.widowx.state['Y'] - vy) < self.DELTA_ACTION and
                abs(self.widowx.state['Z'] - vz) < self.DELTA_ACTION and
                # rad_interval(self.widowx.state['Gamma'], vg) <  self.DELTA_ANGLE and
                abs(self.widowx.state['Rot'] - vr) < self.DELTA_ACTION and
                self.widowx.state['Gripper'] == goc):
                print("ARM MOVE: SUCCESS")
                success = True
          # self.episode_step()
        self.widowx.getState()
        return success

    ##########################
    # Code shared with joystick & RT1 script (by copying)
    ##########################
    # a cross-over interface between joystick & widowx.py, deals with move_mode
    def move(self, vx, vy, vz, vg, vr, goc):
        print("MOVE:", self.move_mode, vx, vy, vz, vg, vr, goc)
        initial_time = self.widowx.millis()
#        # vr and goc only move in "Relative" mode for gpt control
        # if (vr is not None and self.move_mode == 'Absolute'):
            # self.widowx.moveServoWithSpeed(self.widowx.IDX_ROT, vr, initial_time)
            # pass
        # 
        if vr is not None:
            print("move vr:", vr, self.widowx.getServoAngle(self.widowx.IDX_ROT))
            # servo id 5 / idx 4: rotate gripper to angle
            rotLim = (300/360)/2 * math.pi
            if vr < -rotLim:
              print("Wrist Rot: limited angle from/to", vr, -rotLim)
              vr = -rotLim
            if vr > rotLim:
              print("Wrist Rot: limited angle from/to", vr, rotLim)
              vr = rotLim
            self.widowx.moveServo2Angle(self.widowx.IDX_ROT, vr)  
            self.widowx.delay(300)
            self.widowx.getState()

            # angle = 0.00511826979472 * (position - (self.AX12_MAX/2))
            # 614 is 3.1415 / .00511827
            # (6.280626 - 3.1415)  / 0.00511826979472 = 613.3178 => too large (>512)
            # 6.280626 is too big => 300 deg is max
            ########
            # 511.5 * .00511 = 2.613765
            # 2.613765 / 3.1415 = .832
            # .83 * 360 = 298.80
            #
            #  ~2Pi!
            # AX12_MAX = 1023 
            # ax12pos = round(((deg_to_rad(vr) - math.pi) / 0.00511826979472) + (AX12_MAX/2))
            # ax12pos = round((vr / 0.00511826979472) + (AX12_MAX/2))
            # print("ax12pos, vr", ax12pos, vr)
            # self.widowx.moveServo2Position(self.widowx.IDX_ROT, ax12pos)  # servo id 5 / idx 4: rotate gripper to angle

        # if goc is not None and self.move_mode == 'Relative':
        if goc is not None:
            gripper_state, gripper_pos = self.widowx.openCloseGrip(goc)
            if gripper_state == self.widowx.GRIPPER_OPEN:
              self.gripper_fully_open_closed = gripper_state
              self.gripper_pos_open = gripper_pos
            elif gripper_state == self.widowx.GRIPPER_CLOSED:
              self.gripper_fully_open_closed = gripper_state
              self.gripper_pos_closed = gripper_pos
        if (vx or vy or vz or vg):
          if (self.move_mode == 'Relative'):
            fvx = min(max(-1.75, (float(vx) / 127.0 * 1.75)), 1.75)
            fvy = min(max(-1.75, (float(vy) / 127.0 * 1.75)), 1.75)
            fvz = min(max(-1.75, (float(vz) / 127.0 * 1.75)), 1.75)
            fvg = min(max(-1.4, (float(vg) / 255.0 * 1.4)), 1.4)
            self.widowx.movePointWithSpeed(fvx, fvy, fvz, fvg, initial_time)
          elif (self.move_mode == 'Absolute'):
            if (abs(self.widowx.state['X'] - vx) < self.DELTA_ACTION and
                abs(self.widowx.state['Y'] - vy) < self.DELTA_ACTION and
                abs(self.widowx.state['Z'] - vz) < self.DELTA_ACTION):
              print("MOVE: ALREADY NEAR ABSOLUTE POSITION", vx, vy, vz)
            else:
              print("moveArmGammaController Request:", vx, vy, vz, vg)
              print("curr state:", self.widowx.state)
              self.widowx.moveArmGammaController(vx, vy, vz, vg)
