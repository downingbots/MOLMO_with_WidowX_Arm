from PIL import Image
import cv2 
import numpy as np
   
# function to display the coordinates of 
# of the points clicked on the image  
def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 
  
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (x,y), font, 1, 
                    (255, 255, 0), 2) 
        cv2.imshow('image', img) 
  
# driver function 
if __name__=="__main__": 
  
    # reading the image 
    print("enter image filename:")
    filenm = input()
    # filenm = '/home/ros/downingbots_gpt4o/calibration_images/image_empty_bycolor.png'
    if filenm.endswith('png'):
      img = Image.open(filenm)
      img = Image.fromarray(np.array(img))
      img_jpg = "/tmp/coord_img.jpg"
      img.save(img_jpg)
      img = cv2.imread(img_jpg, 1) 
    elif filenm.endswith('jpg'):
      img = cv2.imread(filenm, 1) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
    # displaying the image 
    cv2.imshow('image', img) 
  
    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('image', click_event) 
  
    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 
  
    # close the window 
    cv2.destroyAllWindows() 
