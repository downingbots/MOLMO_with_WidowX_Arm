#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright 2017 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################


#*******************************************************************************
#***********************     SyncRead and SyncWrite Example      ***********************
#  Required Environment to run this example :
#    - Protocol 2.0 supported DYNAMIXEL(X, P, PRO/PRO(A), MX 2.0 series)
#    - DYNAMIXEL Starter Set (U2D2, U2D2 PHB, 12V SMPS)
#  How to use the example :
#    - Select the DYNAMIXEL in use at the self.MY_DXL in the example code. 
#    - Build and Run from proper architecture subdirectory.
#    - For ARM based SBCs such as Raspberry Pi, use linux_sbc subdirectory to build and run.
#    - https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/
#  Author: Ryu Woon Jung (Leon)
#  Maintainer : Zerom, Will Son
# *******************************************************************************

import os
from dynamixel_sdk import *                    # Use Dynamixel SDK library

class DXL_Servos():

   def DXL_init_port(self):
       index = 0
       dxl_goal_position = [self.dxl_params['DXL_MINIMUM_POSITION_VALUE'], self.dxl_params['DXL_MAXIMUM_POSITION_VALUE']]         # Goal position
      
       # Initialize PortHandler instance
       # Set the port path
       # Get methods and members of PortHandlerLinux or PortHandlerWindows
       self.portHandler = PortHandler(self.dxl_params['DEVICENAME'])
       
       # Initialize PacketHandler instance
       # Set the protocol version
       # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
       self.packetHandler = PacketHandler(self.dxl_params['PROTOCOL_VERSION'])
       # self.packetHandler2 = PacketHandler(2.0)
       
       # Initialize GroupSyncWrite instance
       self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.dxl_params['AX_GOAL_POSITION_L'], 4)

       # Initialize GroupBulkRead instance for Present Position
       self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)
       # self.bulk_param_write_size = 0
       self.bulk_param_read_size  = 0
       self.bulk_param_goal_position = []
       self.DXL_MOVING_STATUS_THRESHOLD = 20
       
       # Open port
       if self.portHandler.openPort():
           print("Succeeded to open the port")
       
       # Set port baudrate
       if self.portHandler.setBaudRate(self.dxl_params['BAUDRATE']):
           print("Succeeded to change the baudrate")
       else:
           quit()

   def __init__(self):
       #********* DYNAMIXEL Model definition *********
       #***** (Use only one definition at a time) *****
       self.dxl_params = {}
       self.dxl_params['MY_DXL'] = 'MX_SERIES'    # MX series with 2.0 firmware update.
       # The following are Protocol Version 2.0 POSITION defines. Don't use. Use AX values instead.
       self.dxl_params['ADDR_TORQUE_ENABLE'] = 562       # Control table address is different in DYNAMIXEL model
       self.dxl_params['ADDR_GOAL_POSITION'] = 596
       self.dxl_params['LEN_GOAL_POSITION']  = 4
       self.dxl_params['ADDR_PRESENT_POSITION']       = 611
       self.dxl_params['LEN_PRESENT_POSITION']        = 4
       self.dxl_params['DXL_MINIMUM_POSITION_VALUE']  = -150000   # Refer to the Minimum Position Limit of product eManual
       self.dxl_params['DXL_MAXIMUM_POSITION_VALUE']  = 150000    # Refer to the Maximum Position Limit of product eManual

       ####
       # DYNAMIXEL Protocol Version (1.0 / 2.0)
       # https://emanual.robotis.com/docs/en/dxl/protocol1/
       self.dxl_params['PROTOCOL_VERSION']            = 1.0
       self.dxl_params['BAUDRATE']                    = 1000000
       
       # Use the actual port assigned to the U2D2.
       # ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
       self.dxl_params['DEVICENAME']                   = '/dev/ttyUSB0'
       self.dxl_params['DEVICENAME1']                  = '/dev/ttyUSB1'
       self.dxl_params['DEVICENAME2']                  = '/dev/ttyUSB2'
       self.dxl_params['DEVICENAME3']                  = '/dev/ttyUSB3'
       self.dxl_params['DXL_MOVING_STATUS_THRESHOLD'] = 20             # Dynamixel moving status threshold
       
       self.TORQUE_ENABLE              = 1               # Value for enabling the torque
       self.TORQUE_DISABLE             = 0               # Value for disabling the torque
       
       ################
       # address definitions for Protocol 1.0
       self.dxl_params['AX_MODEL_NUMBER_L'] =           0
       self.dxl_params['AX_MODEL_NUMBER_H'] =           1
       self.dxl_params['AX_VERSION'] =                  2
       self.dxl_params['AX_ID'] =                       3
       self.dxl_params['AX_BAUD_RATE'] =                4
       self.dxl_params['AX_RETURN_DELAY_TIME'] =        5
       self.dxl_params['AX_CW_ANGLE_LIMIT_L'] =         6
       self.dxl_params['AX_CW_ANGLE_LIMIT_H'] =         7
       self.dxl_params['AX_CCW_ANGLE_LIMIT_L'] =        8
       self.dxl_params['AX_CCW_ANGLE_LIMIT_H'] =        9
       self.dxl_params['AX_SYSTEM_DATA2'] =             10
       self.dxl_params['AX_LIMIT_TEMPERATURE'] =        11
       self.dxl_params['AX_DOWN_LIMIT_VOLTAGE'] =       12
       self.dxl_params['AX_UP_LIMIT_VOLTAGE'] =         13
       self.dxl_params['AX_MAX_TORQUE_L'] =             14
       self.dxl_params['AX_MAX_TORQUE_H'] =             15
       self.dxl_params['AX_RETURN_LEVEL'] =             16
       self.dxl_params['AX_ALARM_LED'] =                17
       self.dxl_params['AX_ALARM_SHUTDOWN'] =           18
       self.dxl_params['AX_OPERATING_MODE'] =           19
       self.dxl_params['AX_DOWN_CALIBRATION_L'] =       20
       self.dxl_params['AX_DOWN_CALIBRATION_H'] =       21
       self.dxl_params['AX_UP_CALIBRATION_L'] =         22
       self.dxl_params['AX_UP_CALIBRATION_H'] =         23
       ### RAM AREA ###
       self.dxl_params['AX_TORQUE_ENABLE'] =            24
       self.dxl_params['AX_LED'] =                      25
       self.dxl_params['AX_CW_COMPLIANCE_MARGIN'] =     26
       self.dxl_params['AX_CCW_COMPLIANCE_MARGIN'] =    27
       self.dxl_params['AX_CW_COMPLIANCE_SLOPE'] =      28
       self.dxl_params['AX_CCW_COMPLIANCE_SLOPE'] =     29
       self.dxl_params['AX_GOAL_POSITION_L'] =          30
       self.dxl_params['AX_GOAL_POSITION_H'] =          31
       self.dxl_params['AX_GOAL_SPEED_L'] =             32
       self.dxl_params['AX_GOAL_SPEED_H'] =             33
       self.dxl_params['AX_TORQUE_LIMIT_L'] =           34
       self.dxl_params['AX_TORQUE_LIMIT_H'] =           35
       self.dxl_params['AX_PRESENT_POSITION_L'] =       36
       self.dxl_params['AX_PRESENT_POSITION_H'] =       37
       self.dxl_params['AX_PRESENT_SPEED_L'] =          38
       self.dxl_params['AX_PRESENT_SPEED_H'] =          39
       self.dxl_params['AX_PRESENT_LOAD_L'] =           40
       self.dxl_params['AX_PRESENT_LOAD_H'] =           41
       self.dxl_params['AX_PRESENT_VOLTAGE'] =          42
       self.dxl_params['AX_PRESENT_TEMPERATURE'] =      43
       self.dxl_params['AX_REGISTERED_INSTRUCTION'] =   44
       self.dxl_params['AX_PAUSE_TIME'] =               45
       self.dxl_params['AX_MOVING']  =                  46
       self.dxl_params['AX_LOCK']    =                  47
       self.dxl_params['AX_PUNCH_L'] =                  48
       self.dxl_params['AX_PUNCH_H'] =                  49

       self.DXL_init_port()
 
 
   def DXL_SetRegister(self, id, address, val, n_bytes=1):
       try:
         addr = self.dxl_params[address] 
       except:
         print("Bad Dynamixel Address:", address)
         quit()

       if (val > 256 or address.endswith('_L') or n_bytes == 4):
           dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, id, addr, int(val))
       else:
           dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, id, addr, int(val))
       if dxl_comm_result != COMM_SUCCESS:
           print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
       elif dxl_error != 0:
           print("%s" % self.packetHandler.getRxPacketError(dxl_error))
           print("id, value:", id, val)
           if (address.endswith('_L')):
             print("id, addr:", id, address)
           else:
             print("id, not addr:", id, address)
           x = 1/0
       else:
           # print("Dynamixel#%d %s set to %d" % (id, address, val))
           pass

   def DXL_GetRegister(self, id, address):
       try:
         addr = self.dxl_params[address] 
       except:
         print("Bad Dynamixel Address:", address)
         quit()
       dxl_comm_result = COMM_RX_CORRUPT
       while dxl_comm_result != COMM_SUCCESS:
         if address.endswith('_L'):
            dxl_baudnum_read, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, id, addr)
            # print("read2Byte:", dxl_baudnum_read)
         else:
            dxl_baudnum_read, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, id, addr)
         if dxl_comm_result == COMM_SUCCESS:
           # print("%d %d %s" % (id, dxl_baudnum_read, self.packetHandler.getTxRxResult(dxl_comm_result)))
           break
         elif dxl_error != 0:
           print("%d %s" % (id, self.packetHandler.getRxPacketError(dxl_error)))
       return dxl_baudnum_read


   def DXL_SetPosition(self, id, pos):
       self.DXL_SetRegister(id, 'AX_GOAL_POSITION_L', pos)

   def DXL_GetPosition(self, id):
       return self.DXL_GetRegister(id, 'AX_PRESENT_POSITION_L')

   def DXL_TorqueOn(self, id):
       self.DXL_SetRegister(id, 'AX_TORQUE_ENABLE', self.TORQUE_ENABLE)

   def DXL_Relax(self, id):
       self.DXL_SetRegister(id, 'AX_TORQUE_ENABLE', self.TORQUE_DISABLE)

   def DXL_GetVoltage(self, id):
       return self.DXL_GetRegister(id, 'AX_PRESENT_VOLTAGE')

   def DXL_GetTemperature(self, id):
       return self.DXL_GetRegister(id, 'AX_PRESENT_TEMPERATURE')

   def DXL_SetLED(self, id, on_off):
       self.DXL_SetRegister(id, 'AX_LED', on_off)

   # Close port
   def DXL_closePort(self):
      self.portHandler.closePort()

   def DXL_torque(self, id, enable_disable):
      if (enable_disable not in [self.TORQUE_ENABLE, self.TORQUE_DISABLE]):
          print("illegal torque value:", enable_disable)
 
      if enable_disable == self.TORQUE_ENABLE:
          self.DXL_TorqueOn(id)
      else:
          self.DXL_Relax(id)

   ############################################################################
   # Note: Bulk Sync Read is not in Protocol 1.0, but bulk group Read IS supported for MX servos.
   def DXL_BulkGetPosition(self, until_id):
          # quit()
          if self.bulk_param_read_size < until_id:
              for i in range(self.bulk_param_read_size, until_id):
                  id = i + 1
                  # Add parameter storage for Dynamixel id present position
                  dxl_addparam_result = self.groupBulkRead.addParam(id, self.dxl_params['AX_PRESENT_POSITION_L'], 2)
                  # dxl_addparam_result = self.groupBulkRead.addParam(id, self.dxl_params['AX_PRESENT_POSITION_L'], 4)
                  # dxl_addparam_result = self.groupBulkRead.addParam(id, self.dxl_params['ADDR_PRESENT_POSITION'], 4)
                  if dxl_addparam_result != True:
                      print("[ID:%03d] groupBulkRead addparam failed" % DXL1_ID)
                      quit()
              self.bulk_param_read_size = until_id

          # Bulkread present position 
          # print("groupBulkRead.txRxPacket")
          dxl_comm_result = self.groupBulkRead.txRxPacket()
          if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            # while dxl_comm_result != COMM_SUCCESS:
            #   dxl_comm_result = self.packetHandler.rxPacket(self.portHandler)
            #   print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
          # else:
            # print("COMM_SUCCESS")


          # Check if groupbulkread data of Dynamixels are available
          for i in range(until_id):
            id = i + 1
            # Check if groupbulkread data of Dynamixel#1 is available
            dxl_getdata_result = False
            # while not dxl_getdata_result:
            # dxl_getdata_result = self.groupBulkRead.isAvailable(id, self.dxl_params['ADDR_PRESENT_POSITION'], 4)
            dxl_getdata_result = self.groupBulkRead.isAvailable(id, self.dxl_params['AX_PRESENT_POSITION_L'], 2)
            # dxl_getdata_result = self.groupBulkRead.isAvailable(id, self.dxl_params['AX_PRESENT_POSITION_L'], 4)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupBulkRead isAvailable failed" % id)
                quit()

          # Get Dynamixel present position value
          dxl_present_position = []
          for id in range(until_id):
            dxl_present_position.append(0)

          for i in range(until_id):
            id = i + 1
            # Get present position value
            dxl_present_position[i] = self.groupBulkRead.getData(id, self.dxl_params['AX_PRESENT_POSITION_L'], 2)
            # dxl_present_position[i] = self.groupBulkRead.getData(id, self.dxl_params['ADDR_PRESENT_POSITION'], 4)
            # print("[ID:%03d] Present Position : %d" % (id, dxl_present_position[i]))
            # compare to goal position from write (in widowx.py)
            # if not (abs(dxl_goal_position[i] - dxl_present_position[i]) > DXL_MOVING_STATUS_THRESHOLD):
            #     break
          return dxl_present_position

   ############################################################################
   # Note: Bulk Group Write is not in Protocol 1.0, but bulk Sync Read IS supported for MX servos.
   def DXL_SyncSetPosition(self, until_id, dxl_goal_position):
      # Allocate goal position value into byte array
      # if self.bulk_param_write_size < until_id:  # comment out: needs to be run every call
      #   for i in range(self.bulk_param_write_size, until_id):
      for i in range(until_id):
          id = i + 1
          # Allocate goal position value into byte array
          self.bulk_param_goal_position.append(0)
          self.bulk_param_goal_position[i] = [DXL_LOBYTE(DXL_LOWORD(dxl_goal_position[i])), DXL_HIBYTE(DXL_LOWORD(dxl_goal_position[i])), DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[i])), DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[i]))]
          # Add Dynamixel#1 goal position value to the Syncwrite parameter storage
          dxl_addparam_result = self.groupSyncWrite.addParam(id, self.bulk_param_goal_position[i])
          if dxl_addparam_result != True:
              print("[ID:%03d] groupSyncWrite addparam failed" % id)
              quit()
        # self.bulk_param_write_size = until_id

      # Syncwrite goal position and LED value
      # print("groupSyncWrite.txPacket()")
      dxl_comm_result = self.groupSyncWrite.txPacket()
      if dxl_comm_result != COMM_SUCCESS:
          print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
          # no params added...
      # else:
          # print("COMM_SUCCESS")

      # Clear bulkwrite parameter storage
      self.groupSyncWrite.clearParam()
      return dxl_comm_result

   def DXL_Ping(self, id):
      # Protocol 1 has no broadcast
      dxl_result = COMM_RX_CORRUPT 
      while dxl_result != COMM_SUCCESS:
        dxl_model_number, dxl_result, dxl_error = self.packetHandler.ping(self.portHandler, id)
        # print(id,dxl_model_number, dxl_result, dxl_error)
      print("[ID:%03d] ping Succeeded. Dynamixel model number : %d" % (id, dxl_model_number))
      return dxl_model_number


   # TODO: use protocol 2 to reboot MX servos
   # note: setting LEDs and other ADDRs can also sent in bulk 

