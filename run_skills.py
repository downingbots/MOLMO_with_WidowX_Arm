import cv2
import do_skills
import widowx_client
import widowx_calibrate
import widowx_manual_control

calib = widowx_calibrate.widowx_calibrate()
wmc = widowx_manual_control.widowx_manual_control(calib.wdw, calib.robot_camera)
wmc.manual_pix_control(True)

