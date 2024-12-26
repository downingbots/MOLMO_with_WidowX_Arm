import cv2
import copy
import math
import do_skills
import camera_snapshot
import widowx_client
import widowx_calibrate

class widowx_manual_control():
  def __init__(self, wdw_client, robot_cam, calib):
      self.robot_actions = ["FORWARD", "BACKWARD", "UP", "DOWN", "LEFT", "RIGHT", 
               "ROTATE_ARM_CLOCKWISE", "ROTATE_ARM_COUNTERCLOCKWISE", 
               "ROTATE_GRIPPER_CLOCKWISE", "ROTATE_GRIPPER_COUNTERCLOCKWISE", 
               "GRIPPER_OPEN", "GRIPPER_CLOSE", "PICK_POSE", "REST",
               "DONE"] 
      self.wrist_rotation_angle = 0
      self.camera_snapshot = robot_cam
      self.wdw = wdw_client
      self.calib = calib
      self.skills = do_skills.skills(wdw_client, self.calib)

  def set_click_image(self, click_lbl, click_img):
      self.click_label = click_lbl
      self.click_image = copy.deepcopy(click_img)
      cv2.imshow(self.click_label, self.click_image)
  
  # low level control
  def do_manual_action(self, do_loop=False):
      # print(filename)
      USE_FINE_DELTA = len(self.robot_actions) + 1 
      RETURN_TO_SKILLS_MENU = len(self.robot_actions) + 2 
      DELTA = 127
      FINE_DELTA = 63
      use_fine_delta = False
      while True:
        print("Select action:")
        for i in range(len(self.robot_actions)):
          if i < 9:
            print(str(i+1) + "  " + self.robot_actions[i])
          else:
            print(str(i+1) + " " + self.robot_actions[i])
        print(str(USE_FINE_DELTA) + " use fine delta toggle")
        print(str(RETURN_TO_SKILLS_MENU) + " return to skill menu")
        while True:
          print("")
          print("action:")
          action_number = input()
          try:
            if int(action_number) == USE_FINE_DELTA:
              if use_fine_delta:
                use_fine_delta = False
              else:
                use_fine_delta = True
              print("use_fine_delta = ", use_fine_delta)
            if int(action_number) == RETURN_TO_SKILLS_MENU:
              return
            if int(action_number) <= len(self.robot_actions):
              break
          except:
            pass
        robot_action = self.robot_actions[int(action_number)-1]
        # robot_action['terminate_episode'] = True
        # observation['action'] = robot_action
        ############################################################
        # Move the robot based on selected action and take snapshot
        ############################################################
        self.wdw.set_move_mode('Relative')
        if use_fine_delta:
          delta = FINE_DELTA
        else:
          delta = DELTA
         
        if robot_action == "FORWARD":
          self.wdw.action(vx=delta)
        elif robot_action == "BACKWARD":
          self.wdw.action(vx= -delta)
        elif robot_action == "UP":
          self.wdw.action(vz= delta)
        elif robot_action == "DOWN":
          self.wdw.action(vz= -delta)
        elif robot_action == "LEFT":
          self.wdw.action(vy= delta)
        elif robot_action == "RIGHT":
          self.wdw.action(vy= -delta)
        elif robot_action == "ROTATE_ARM_CLOCKWISE":
          self.wdw.do_swivel("RIGHT")
        elif robot_action == "ROTATE_ARM_COUNTERCLOCKWISE":
          self.wdw.do_swivel("LEFT")
        elif robot_action == "ROTATE_GRIPPER_CLOCKWISE":
          DELTA_ROT_ANGLE = math.pi / 100
          rotLim = (300/360)/2 * math.pi
          if self.wrist_rotation_angle < rotLim:
            self.wrist_rotation_angle += DELTA_ROT_ANGLE
          print("wrist rotation angle", self.wrist_rotation_angle)
          vg_rot = self.wrist_rotation_angle
          self.wdw.action(vr= vg_rot)
          # self.wdw.wrist_rotate(vg_rot)
        elif robot_action == "ROTATE_GRIPPER_COUNTERCLOCKWISE":
          DELTA_ROT_ANGLE = math.pi / 100
          rotLim = (300/360)/2 * math.pi
          if self.wrist_rotation_angle > -rotLim:
            self.wrist_rotation_angle -= DELTA_ROT_ANGLE
          print("wrist rotation angle", self.wrist_rotation_angle)
          vg_rot = self.wrist_rotation_angle
          self.wdw.action(vr= vg_rot)
          # self.wdw.wrist_rotate(vg_rot)
        elif robot_action == "GRIPPER_OPEN":
          self.wdw.gripper(self.wdw.GRIPPER_OPEN)
        elif robot_action == "GRIPPER_CLOSE": 
          self.wdw.gripper(self.wdw.GRIPPER_CLOSED)
        elif robot_action == "PICK_POSE":
          self.wdw.gripper(self.wdw.GRIPPER_OPEN)
          self.wdw.moveArmPick()
          self.wdw.wrist_rotate(angle = self.wdw.GRIPPER_ROT_TO_CAMERA)
        elif robot_action == "REST":
          self.wdw.gripper(self.wdw.GRIPPER_OPEN)
          self.wdw.moveRest()
          # self.wdw.wrist_rotate(angle = self.wdw.GRIPPER_ROT_TO_CAMERA)
        elif robot_action == "DONE":
          # self.wdw.gripper(self.wdw.GRIPPER_OPEN)
          # self.wdw.moveArmPick()
          # self.wdw.wrist_rotate(angle = self.wdw.GRIPPER_ROT_TO_CAMERA)
          return False
        if not do_loop:
          break
      return True

  # skill-level control with point-and-click locations
  def manual_pix_control(self, do_loop=False):
        self.wdw.moveOutCamera()
        self.click_row = None
        self.click_col = None
        while True:
          self.latest_image, self.latest_image_file, self.latest_image_time = self.camera_snapshot.snapshot(True)
          self.click_image = copy.deepcopy(self.latest_image)
          cv2.imshow("TableTop",self.click_image)
          while True:
            print("1 = Pick Up from tabletop")
            print("2 = Pick Up above tabletop")
            print("3 = Place at tabletop")
            print("4 = Place above tabletop")
            print("5 = Push")
            print("6 = Move Arm by small delta")
            print("7 = Move Arm out of picture")
            print("8 = Exit")
            print(" ")
            print("action:")
            num = input()
            try:
              num = int(num)
              if 1 <= num <= 8:
                break
            except:
              continue
          # self.wdw.moveArmPick()
          self.click_label = 'TableTop'

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
  
  # low level control
  def do_manual_action(self, do_loop=False):
      # print(filename)
      USE_FINE_DELTA = len(self.robot_actions) + 1 
      RETURN_TO_SKILLS_MENU = len(self.robot_actions) + 2 
      DELTA = 127
      FINE_DELTA = 63
      use_fine_delta = False
      while True:
        print("Select action:")
        for i in range(len(self.robot_actions)):
          if i < 9:
            print(str(i+1) + "  " + self.robot_actions[i])
          else:
            print(str(i+1) + " " + self.robot_actions[i])
        print(str(USE_FINE_DELTA) + " use fine delta toggle")
        print(str(RETURN_TO_SKILLS_MENU) + " return to skill menu")
        while True:
          print("")
          print("action:")
          action_number = input()
          try:
            if int(action_number) == USE_FINE_DELTA:
              if use_fine_delta:
                use_fine_delta = False
              else:
                use_fine_delta = True
              print("use_fine_delta = ", use_fine_delta)
            if int(action_number) == RETURN_TO_SKILLS_MENU:
              return
            if int(action_number) <= len(self.robot_actions):
              break
          except:
            pass
        robot_action = self.robot_actions[int(action_number)-1]
        # robot_action['terminate_episode'] = True
        # observation['action'] = robot_action
        ############################################################
        # Move the robot based on selected action and take snapshot
        ############################################################
        self.wdw.set_move_mode('Relative')
        if use_fine_delta:
          delta = FINE_DELTA
        else:
          delta = DELTA
         
        if robot_action == "FORWARD":
          self.wdw.action(vx=delta)
        elif robot_action == "BACKWARD":
          self.wdw.action(vx= -delta)
        elif robot_action == "UP":
          self.wdw.action(vz= delta)
        elif robot_action == "DOWN":
          self.wdw.action(vz= -delta)
        elif robot_action == "LEFT":
          self.wdw.action(vy= delta)
        elif robot_action == "RIGHT":
          self.wdw.action(vy= -delta)
        elif robot_action == "ROTATE_ARM_CLOCKWISE":
          self.wdw.do_swivel("RIGHT")
        elif robot_action == "ROTATE_ARM_COUNTERCLOCKWISE":
          self.wdw.do_swivel("LEFT")
        elif robot_action == "ROTATE_GRIPPER_CLOCKWISE":
          DELTA_ROT_ANGLE = math.pi / 100
          rotLim = (300/360)/2 * math.pi
          if self.wrist_rotation_angle < rotLim:
            self.wrist_rotation_angle += DELTA_ROT_ANGLE
          print("wrist rotation angle", self.wrist_rotation_angle)
          vg_rot = self.wrist_rotation_angle
          self.wdw.action(vr= vg_rot)
          # self.wdw.wrist_rotate(vg_rot)
        elif robot_action == "ROTATE_GRIPPER_COUNTERCLOCKWISE":
          DELTA_ROT_ANGLE = math.pi / 100
          rotLim = (300/360)/2 * math.pi
          if self.wrist_rotation_angle > -rotLim:
            self.wrist_rotation_angle -= DELTA_ROT_ANGLE
          print("wrist rotation angle", self.wrist_rotation_angle)
          vg_rot = self.wrist_rotation_angle
          self.wdw.action(vr= vg_rot)
          # self.wdw.wrist_rotate(vg_rot)
        elif robot_action == "GRIPPER_OPEN":
          self.wdw.gripper(self.wdw.GRIPPER_OPEN)
        elif robot_action == "GRIPPER_CLOSE": 
          self.wdw.gripper(self.wdw.GRIPPER_CLOSED)
        elif robot_action == "PICK_POSE":
          self.wdw.gripper(self.wdw.GRIPPER_OPEN)
          self.wdw.moveArmPick()
          self.wdw.wrist_rotate(angle = self.wdw.GRIPPER_ROT_TO_CAMERA)
        elif robot_action == "REST":
          self.wdw.gripper(self.wdw.GRIPPER_OPEN)
          self.wdw.moveRest()
          # self.wdw.wrist_rotate(angle = self.wdw.GRIPPER_ROT_TO_CAMERA)
        elif robot_action == "DONE":
          # self.wdw.gripper(self.wdw.GRIPPER_OPEN)
          # self.wdw.moveArmPick()
          # self.wdw.wrist_rotate(angle = self.wdw.GRIPPER_ROT_TO_CAMERA)
          return False
        if not do_loop:
          break
      return True

  # skill-level control with point-and-click locations
  def manual_pix_control(self, do_loop=False):
        self.wdw.moveOutCamera()
        self.click_row = None
        self.click_col = None
        while True:
          self.latest_image, self.latest_image_file, self.latest_image_time = self.camera_snapshot.snapshot(True)
          self.click_image = copy.deepcopy(self.latest_image)
          cv2.imshow("TableTop",self.click_image)
          while True:
            print("1 = Pick Up from tabletop")
            print("2 = Pick Up above tabletop")
            print("3 = Place at tabletop")
            print("4 = Place above tabletop")
            print("5 = Push")
            print("6 = Move Arm by small delta")
            print("7 = Move Arm out of picture")
            print("8 = Exit")
            print(" ")
            print("action:")
            num = input()
            try:
              num = int(num)
              if 1 <= num <= 8:
                break
            except:
              continue
          # self.wdw.moveArmPick()
          self.click_label = 'TableTop'
          if (num == 1):
            print("click on base of object to pick up")
            # setting mouse handler for the image
            # and calling the click_event() function
            cv2.setMouseCallback('TableTop', self.click_event)
            cv2.waitKey(0) 
            self.skills.pick(self.click_row, self.click_col, tt_loc=None)
          elif (num == 2):
            print("click on tabletop below object to pick up")
            cv2.setMouseCallback('TableTop', self.click_event)
            cv2.waitKey(0) 
            ttloc = [self.click_row, self.click_col]
            print("click on object to pick up")
            cv2.setMouseCallback('TableTop', self.click_event)
            cv2.waitKey(0) 
            self.skills.pick(self.click_row, self.click_col, tt_loc=ttloc)
          elif (num == 3): # Place at tabletop
            print("click on tabletop location to place")
            # setting mouse handler for the image
            # and calling the click_event() function
            cv2.setMouseCallback('TableTop', self.click_event)
            cv2.waitKey(0)
            self.skills.place(self.click_row, self.click_col, tt_loc=None)
            self.wdw.moveOutCamera()
          elif (num == 4): # Place above tabletop
            print("click on tabletop location directly below place location")
            cv2.setMouseCallback('TableTop', self.click_event)
            cv2.waitKey(0)
            ttloc = [self.click_row, self.click_col]
            print("click on place location")
            cv2.setMouseCallback('TableTop', self.click_event)
            cv2.waitKey(0)
            self.skills.place(self.click_row, self.click_col, tt_loc=ttloc)
          elif (num == 5): # Push
            print("click on tabletop location for gripper to start push")
            cv2.setMouseCallback('TableTop', self.click_event)
            cv2.waitKey(0)
            start_x = self.click_row
            start_y = self.click_col
            print("click on tabletop location for gripper to stop push")
            cv2.setMouseCallback('TableTop', self.click_event)
            cv2.waitKey(0)
            end_x = self.click_row
            end_y = self.click_col
            self.skills.push(start_x, start_y, end_x, end_y, end_up=True)
          elif (num == 6): 
            self.do_manual_action(True)
          elif (num == 7): 
            self.wdw.moveOutCamera()
          elif (num == 8): 
            exit()
          if not do_loop:
            break
      
if False:
  calib = widowx_calibrate()
  wmc = widowx_manual_control(calib.wdw, calib.robot_camera, calib)
  while True:
    wmc.manual_pix_control()

