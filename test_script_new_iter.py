import cv2 as cv
import numpy as np
import time
print(cv.__version__)

font = cv.FONT_HERSHEY_SIMPLEX

# for pc display
dispW=int(1920 * 0.5) # WIDTH OF OUTPUT IMAGE
dispH=int(1080 * 0.5) # HEIGHT OF ROTATED OUTPUT IMAGE
flip=6

# for vr display
#dispW=int(1080 * 0.5) # WIDTH OF ROTATED OUTPUT IMAGE
#dispH=int(1920 * 0.5) # HEIGHT OF ROTATED OUTPUT IMAGE
#flip=7 

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=1848, format=NV12, framerate=28/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! vaapipostproc ! video/x-raw, denoise = 5 ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv.VideoCapture(camSet)

#    return cv.resize(img, None, fx=zoom_factor, fy=zoom_factor)
while True:
    
    start = time.time()
    ret, frame = cam.read()
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    cv.imshow('picam',frame)
    key = cv.waitKey(1)
    if key==ord('q'):
        break
cam.release()
cv.destroyAllWindows()


