import cv2
import numpy as np
import time
print(cv2.__version__)

font = cv2.FONT_HERSHEY_SIMPLEX

# for pc display
#dispW=640
#dispH=480
dispW=1280
dispH=720
flip=2

# for vr display
#dispW=2160 
#dispH=1080 
#flip=3  

# gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "video/x-raw(memory:NVMM),width=3264,height=2464,framerate=$FRAMERATE/1" ! nvvidconv flip-method=3 ! nvoverlaysink
#camSet='nvarguscamerasrc sensor-id=0 ! "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=$30/1" ! nvvidconv ! nvoverlaysink'
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camSet)

#def zoom(img, zoom_factor):
#    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
zoom_factor = 1.0
frame_count = 0
while True:
    start = time.time()
    ret, frame = cam.read()
    ##########################camera output position correction###################
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    #print(frame_w,',',frame_h)
    split_len = frame_w//2
    camera_A = frame[ : , split_len : ]
    camera_B = frame[ : , : split_len ]
    #new_frame = zoom(new_frame, 2) 
    ###############################zoom in,out OLD####################################
    #if zoom_state == 1:
    #   crop_frame_A = camera_A[int(0.125*frame_w):int(0.375*frame_w), int(0.25*frame_h):int(0.75*frame_h)]
    #   crop_frame_B = camera_B[int(0.125*frame_w):int(0.375*frame_w), int(0.25*frame_h):int(0.75*frame_h)] 
    #   #resize_frame_A = cv2.resize(crop_frame_A,[0.5*frame_w,0.5*frame_h],interpolation = cv2.INTER_AREA )
    #   #resize_frame_B = cv2.resize(crop_frame_B,[0.5*frame_w,0.5*frame_h],interpolation = cv2.INTER_AREA )
    #   camera_A = cv2.resize(crop_frame_A,(int(0.5*frame_w),frame_h),interpolation = cv2.INTER_AREA )
    #   camera_B = cv2.resize(crop_frame_B,(int(0.5*frame_w),frame_h),interpolation = cv2.INTER_AREA )
    #   #new_frame = zoom(new_frame, 0.5) 
    #new_frame = np.concatenate((camera_A,camera_B),axis=1)
    ##############################################################################
    ###############################zoom in,out OLD####################################
    zoom_factor = round(zoom_factor,1)
    crop_w = (0.5 * frame_w) // zoom_factor
    crop_h = frame_h // zoom_factor
    crop_frame_A = camera_A[int(frame_w // 4 - crop_w // 2):int(frame_w // 4 + crop_w // 2), int(frame_h // 2 - crop_h // 2):int(frame_h // 2 + crop_h // 2)]
    crop_frame_B = camera_B[int(frame_w // 4 - crop_w // 2):int(frame_w // 4 + crop_w // 2), int(frame_h // 2 - crop_h // 2):int(frame_h // 2 + crop_h // 2)]
    camera_A = cv2.resize(crop_frame_A,(int(0.5*frame_w),frame_h),interpolation = cv2.INTER_AREA )
    camera_B = cv2.resize(crop_frame_B,(int(0.5*frame_w),frame_h),interpolation = cv2.INTER_AREA )
    new_frame = np.concatenate((camera_A,camera_B),axis=1)
    #new_frame = np.concatenate((crop_frame_A,crop_frame_B),axis=1)
    ##############################################################################
    end = time.time()
    ##########################fps calculation and display as overlay old##############
    time_elapsed = end - start
    fps = 1 // time_elapsed
    cv2.putText(new_frame,'fps:' + str(fps),(20,30),font,1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(new_frame,'fps:' + str(fps),(split_len+20,30),font,1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(new_frame,'X' + str(zoom_factor),(220,30),font,1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(new_frame,'X' + str(zoom_factor),(split_len+220,30),font,1,(0,0,255),2,cv2.LINE_AA)
    ##############################################################################
    cv2.imshow('picam',new_frame)
    #cv2.imshow('picam',frame)

    key = cv2.waitKey(1)
    #if cv2.waitKey(1)==ord('m'):#zoom in:
    #   zoom_state = 1
    #if cv2.waitKey(1)==ord('n'):#zoom in
    #   zoom_state = 0
    #if cv2.waitKey(1)==ord('q'):
    #    break
    
    if key==ord('m'):#zoom in:
       if zoom_factor >= 8:
         zoom_factor = 8.0
       else:  
         zoom_factor = zoom_factor + 0.2
    if key==ord('n'):#zoom in
       if zoom_factor <= 1:
         zoom_factor = 1.0
       else:  
          zoom_factor = zoom_factor - 0.2
    if key==ord('c'):
        if ret:
          cv2.imwrite(time.strftime("%Y%m%d-%H%M%S")+'_processed_frame.jpg',new_frame)
          cv2.imwrite(time.strftime("%Y%m%d-%H%M%S")+'_original_frame.jpg',frame)
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()


