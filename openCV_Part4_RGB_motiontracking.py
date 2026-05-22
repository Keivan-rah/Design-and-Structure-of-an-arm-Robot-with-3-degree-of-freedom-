import numpy as np
import cv2
import serial


myDataStream_overBluetoothSerial = serial.Serial('COM30',115200, timeout=.1)
myCameraSream = cv2.VideoCapture(1)
boundaries_Red    = [([0, 0, 0], [250, 50, 50])]
boundaries_Green  = [([0, 0, 0], [50, 250, 50])]
boundaries_Blue   = [([0, 0, 0], [50, 50, 250])]
boundaries_Yellow = [([209, 229, 0], [255, 255, 160])]
boundaries_Orange = [([0, 0, 0], [250, 200, 0])]




while(True):
    _ , inputFrame = myCameraSream.read()
    #-----------------------------------------------------------
    # Processing Area
    #----------------------------
    # Bluring Imgae
    #bluredFrame = cv2.GaussianBlur(inputFrame,(5,5),0)
    bluredFrame = cv2.medianBlur(inputFrame,5)
    #bluredFrame = cv2.bilateralFilter(inputFrame,9,75,75)

    #---------------------------
    # Change color channel and find specific color
    rgbColorFrame = cv2.cvtColor(bluredFrame , cv2.COLOR_BGR2RGB)
    coloredFrame_RGB = rgbColorFrame
    for (lower_colorbound, upper_colorbound) in boundaries_Yellow:

        lower_colorbound = np.array(lower_colorbound, dtype = "uint8")
        upper_colorbound = np.array(upper_colorbound, dtype = "uint8")
        color_mask = cv2.inRange(rgbColorFrame, 
                                 lower_colorbound, 
                                 upper_colorbound)
        output_colorextraction = cv2.bitwise_and(rgbColorFrame, 
                                                 rgbColorFrame, 
                                                 mask = color_mask)
        coloredFrame_RGB = output_colorextraction
    #-----------------------------------------------------------
    # Convert to normalized Grayscale
    channel_R, channel_G, channel_B = cv2.split(coloredFrame_RGB)
    grayScaledFrame_normal = (channel_R + channel_G + channel_B) / 255
    grayScaledFrame_Standard = cv2.cvtColor(coloredFrame_RGB , cv2.COLOR_RGB2GRAY)
    #------------------------------
    # implement thresholding filters
    _, thresholdedFrame = cv2.threshold(grayScaledFrame_Standard, 127, 255, cv2.THRESH_BINARY)
    #_, thresholdedFrame = cv2.threshold(grayScaledFrame_normal, 127, 255, cv2.THRESH_TRUNC)
    #_, thresholdedFrame = cv2.threshold(grayScaledFrame_normal, 127, 255, cv2.THRESH_TOZERO)
    #thresholdedFrame = cv2.adaptiveThreshold(grayScaledFrame_normal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #thresholdedFrame = cv2.adaptiveThreshold(grayScaledFrame_normal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    #------------------------------
    # Frame dilation
    kernel_dilation = np.ones((3,3),np.uint8)
    dilatedFrame = cv2.dilate(thresholdedFrame,kernel_dilation,iterations = 3)
    
    # Create conture
    
    contours, _ = cv2.findContours(dilatedFrame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(np.shape(contours))
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        if(radius > 20):
            cv2.circle(inputFrame,
                        (int(cX),int(cY)),
                        int(radius),
                        (0,0,255),
                            2)   
              
            outData = str(int(cX)) +";" + str(int(cY))  + '\n'
            print(outData)
            myDataStream_overBluetoothSerial.write(outData.encode())                             
    cv2.imshow('Output Stream' , inputFrame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break

myDataStream_overBluetoothSerial.close()
myCameraSream.release()
cv2.destroyAllWindows()
