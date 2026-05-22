#___________________________________________________________________________________________________________________________
#import modules
import numpy as np
import cv2
import picamera
import time
import spidev
#___________________________________________________________________________________________________________________________
#initialise 
spi=spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 125000
time.sleep(0.0005)
mcs = cv2.VideoCapture(0)
#___________________________________________________________________________________________________________________________
# HSV Color Bound
boundaries_Red    = [([0, 150, 150], [5, 250, 250])]
boundaries_Green  = [([40, 50, 50], [90, 250, 250])]
boundaries_Blue   = [([94, 50, 50], [126, 250, 250])]
boundaries_Yellow = [([15, 100, 100], [40, 250, 250])]
boundaries_Orange = [([10, 100, 100], [14, 250, 250])]


while(True):
    avf , frame = mcs.read()
    bl_frame= cv2.GaussianBlur(frame,(5,5),0)
#___________________________________________________________________________________________________________________________
    # Processing Area
    hsv = cv2.cvtColor(bl_frame, cv2.COLOR_BGR2HSV)
    pf = hsv
    for (lcb,ucb) in boundaries_Blue:
        lcb = np.array(lcb, dtype = "uint8")
        ucb = np.array(ucb, dtype = "uint8")
        mask = cv2.inRange(hsv,lcb,ucb)
#___________________________________________________________________________________________________________________________
    _ctr,_hi = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    _ctr=sorted(_ctr,key=lambda x:cv2.contourArea(x),reverse=True)
    for cnt in _ctr:
       (x,y,w,h)=cv2.boundingRect(cnt)
       #cv2.rectangle(bl_frame,(x,y),(x+w,y+h),(100,250,10),2)
       cv2.line(bl_frame,(x,y+(h/2)),(x+w,y+(h/2)),(100,5,210),2)
       cv2.line(bl_frame,(x+(w/2),y),(x+(w/2),y+h),(100,5,210),2)
      # rasp=spi.xfer2(0xAA)
       break
#___________________________________________________________________________________________________________________________
       #print(cnt)
    cv2.imshow("out" ,bl_frame)
    cv2.imshow("mask", mask)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

mcs.release()
cv2.destroyAllWindows()
