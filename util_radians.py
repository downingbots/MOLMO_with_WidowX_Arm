import numpy as np

#########################
# Radian Function
#########################
#
# The following Radian functions are being used to keep all
# angles and differences in angles in the code positive.
def rad_pos(x):
  if x < 0:
    return 2 * np.pi + x
  else:
    return x
  
def rad_arctan2(dx, dy):
  return rad_pos(np.arctan2(dx, dy))

def deg_to_rad(degrees):
    radians = degrees / 180 * np.pi
    return radians

def rad_to_deg(radians):
    # convert to degree
    degree = abs(radians * (180 / np.pi))
    return degree

# convert from 0 to 2pi range into -pi to +pi range
def conv_to_pos_neg_rads(radians):
    rads = radians
    rads -= np.pi
    return rads

RADIAN_RIGHT_ANGLE = 1.5708

# takes a radian input, returns diff from right angle in degrees
def dif_from_90_degrees(rad_angle):
    # indicate direction of right angle 
    while True:
      if rad_angle >= 0 and rad_angle <= np.pi:
        dir = "N"
        angle = rad_dif(rad_angle,RADIAN_RIGHT_ANGLE)
        dif_from_90_deg = rad_to_deg((angle))
        break
      elif rad_angle > np.pi and rad_angle <= 2*np.pi:
        dir = "S"
        angle = rad_dif(rad_angle, (np.pi+RADIAN_RIGHT_ANGLE))
        dif_from_90_deg = rad_to_deg((angle))
        break
      elif rad_angle > 2*np.pi:
        rad_angle = rad_angle - 2*np.pi
      elif rad_angle < 0:
        rad_angle = rad_angle + 2*np.pi
    if dif_from_90_deg > 180 and dif_from_90_deg < 360:
      dif_from_90_deg = (dif_from_90_deg - 360)
    elif dif_from_90_deg < 0 and dif_from_90_deg < -360:
      dif_from_90_deg = dif_from_90_deg + 360
    return dif_from_90_deg, dir

################################
# rad_sum2, rad_dif2 does a sum/dif for LEFT and dif/sum for RIGHT.
# Simplifies code logic as many angle calculations are the opposite for L/R.
def rad_sum2(action, x, y):
  if action == "LEFT":
    return rad_sum(x, y)
  elif action == "RIGHT":
    return (2*np.pi - rad_sum(x, y))

def rad_dif2(action, x, y):
  if action == "LEFT":
    return rad_dif(x, y)
  elif action == "RIGHT":
    return (2*np.pi - rad_dif(x, y))

def rad_sum(x, y):
  x = rad_pos(x)
  y = rad_pos(y)
  r_sum = x + y
  if r_sum > 2*np.pi:
    r_sum = r_sum % (2*np.pi)
  else:
    while r_sum < 0:
      r_sum = r_sum + (2*np.pi)
  return r_sum
  # return min(x + y, abs(x + y - 2*np.pi))

def rad_dif(x, y):
  x = rad_pos(x)
  y = rad_pos(y)
  r_dif = x - y
  if r_dif > 2*np.pi:
    r_dif = r_dif % (2*np.pi)
  else:
    while r_dif < 0:
      r_dif = r_dif + (2*np.pi)
  return r_dif
  # return min((2*np.pi) - abs(x - y), abs(x - y))

# absolute value of shortest distance in either direction 
# returns positive radians that are less than pi
def rad_interval(x, y):
  d1 = rad_dif(x, y)
  d2 = rad_dif(y, x)
  if d1 > np.pi:
    d1 = d1 - np.pi
  if d2 > np.pi:
    d2 = d2 - np.pi
  return min(d1,d2)

def rad_isosceles_triangle(pt1, angle_pt, pt2):
  dist1 = np.sqrt((pt1[0] - angle_pt[0])**2 + (pt1[1] - angle_pt[1])**2)
  dist2 = np.sqrt((pt2[0] - angle_pt[0])**2 + (pt2[1] - angle_pt[1])**2)
  dist3 = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
  if abs(dist1 - dist2) > .01*min(dist1,dist2):
    # print("ERROR: isosceles triangle does not have equal sides:",dist1, dist2, pt1, angle_pt, pt2)
    return None
  # two ways to compute the same angle.
  rad_angle = np.arccos((dist1**2 + dist2**2 - dist3**2) / (2.0 * dist1 * dist2))
  # drawing a line from the angle to the opposite side forms a 90deg triangle.
  # compute the height of the angle. Get angle from arctan2. Double the angle.
  h = np.sqrt(dist1**2 - (dist3/2)**2)
  a = np.arctan2(dist3/2, h)
  h2 = np.sqrt(dist2**2 - (dist3/2)**2)
  a2= np.arctan2(dist3/2, h2)
  print("p1,rl,p2:",pt1, angle_pt,pt2)
  print("arcos, tan angles:", rad_angle, (a, a2), dist1,dist2,dist3)
  return 2*a

