import cv2 as cv
import numpy as np
import time
print(cv.__version__)

font = cv.FONT_HERSHEY_SIMPLEX

# for pc display
#dispW=640
#dispH=480
#dispW=1280
#dispH=720
#flip=2

# for vr display
dispH=2160 
dispW=1080 
flip=3  
 
# gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "video/x-raw(memory:NVMM),width=3264,height=2464,framerate=$FRAMERATE/1" ! nvvidconv flip-method=3 ! nvoverlaysink
#camSet='nvarguscamerasrc sensor-id=0 ! "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=$30/1" ! nvvidconv ! nvoverlaysink'
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#need to denoise
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=1848, format=NV12, framerate=28/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! vaapipostproc ! video/x-raw, denoise = 5 ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv.VideoCapture(camSet)

#def zoom(img, zoom_factor):
#    return cv.resize(img, None, fx=zoom_factor, fy=zoom_factor)
zoom_factor = 1.0
frame_count = 0
mode_select = 0
mode_name = 'Norm'
while True:
    #print(frame_count)
    #frame_count+= 1
    start = time.time()
    ret, frame = cam.read()
    #frame = cv.bilateralFilter(frame,3,50,50)
    #frame = cv.fastNlMeansDenoisingColored(frame,None,3,3,7,21)
    ##########################camera output position correction####################
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    split_len = frame_h//2
    camera_A = frame[split_len : ,  : ]
    camera_A_h = camera_A.shape[0]
    camera_A_w = camera_A.shape[1]
    camera_B = frame[ : split_len , :  ]
    camera_B_h = camera_B.shape[0]
    camera_B_w = camera_B.shape[1]
    #cv.putText(camera_A,'+',(camera_A_w//2 ,camera_A_h//2),font,1,(0,0,255),2,cv.LINE_AA)
    #cv.putText(camera_B,'+',(camera_B_w//2 ,camera_B_h//2),font,1,(0,0,255),2,cv.LINE_AA)
    ###############################edge extraction####################################
    gray_camera_A = cv.cvtColor(camera_A, cv.COLOR_BGR2GRAY)
    gray_camera_B = cv.cvtColor(camera_B, cv.COLOR_BGR2GRAY)
    gray_blur_A = cv.GaussianBlur(gray_camera_A,(3,3),0)
    gray_blur_B = cv.GaussianBlur(gray_camera_B,(3,3),0)
    edges_camera_A = cv.Canny(gray_blur_A,80,200)
    edges_camera_B = cv.Canny(gray_blur_B,80,200)
    edges_camera_A = cv.cvtColor(edges_camera_A, cv.COLOR_GRAY2BGR)
    edges_camera_B = cv.cvtColor(edges_camera_B, cv.COLOR_GRAY2BGR)
    #----------------------------------edge overlay------------------------------------
    if mode_select== 2: #edgemode
      mode_name = 'edge'
      camera_A_edges = cv.addWeighted(camera_A, 1, edges_camera_A, 0.4, 1.2 )
      camera_B_edges = cv.addWeighted(camera_B, 1, edges_camera_B, 0.4, 1.2 )
      camera_A = camera_A_edges
      camera_B = camera_B_edges
      #camera_A = cv.cvtColor(edges_camera_A, cv.COLOR_GRAY2BGR)
      #camera_B = cv.cvtColor(edges_camera_B, cv.COLOR_GRAY2BGR)
    ###############################creating overlay################################
    overlay_camera_A = cv.resize(camera_A,(int(camera_A_w//2.5) , int(camera_A_h//2.5)),interpolation = cv.INTER_LINEAR )
    overlay_camera_A_h = overlay_camera_A.shape[0]
    overlay_camera_A_w = overlay_camera_A.shape[1]
    overlay_camera_B = cv.resize(camera_B,(int(camera_B_w//2.5) , int(camera_B_h//2.5)),interpolation = cv.INTER_LINEAR )
    overlay_camera_B_h = overlay_camera_B.shape[0]
    overlay_camera_B_w = overlay_camera_B.shape[1]
    ###############################zoom in,out ####################################
    zoom_factor = round(zoom_factor,1)
    crop_A_w = camera_A_w // zoom_factor
    crop_B_w = camera_B_w // zoom_factor
    crop_A_h = camera_A_h // zoom_factor
    crop_B_h = camera_B_h // zoom_factor
    crop_frame_A = camera_A[int(camera_A_w // 2 - crop_A_w // 2):int(camera_A_w // 2 + crop_A_w // 2), int(camera_A_h // 2 - crop_A_h // 2):int(camera_A_h // 2 + crop_A_h // 2)]
    crop_frame_B = camera_B[int(camera_B_w // 2 - crop_B_w // 2):int(camera_B_w // 2 + crop_B_w // 2), int(camera_B_h // 2 - crop_B_h // 2):int(camera_B_h // 2 + crop_B_h // 2)]
    camera_A = cv.resize(crop_frame_A,(frame_w,int(0.5*frame_h)),interpolation = cv.INTER_LANCZOS4 )
    #camera_A = cv.fastNlMeansDenoisingColored(camera_A,None,3,3,7,21)
    camera_B = cv.resize(crop_frame_B,(frame_w,int(0.5*frame_h)),interpolation = cv.INTER_LANCZOS4 )
    #camera_B = cv.fastNlMeansDenoisingColored(camera_B,None,3,3,7,21)
    #---------------------------------------------add overlay---------------------------
    if mode_select == 1:# overlay mode
      camera_A[camera_A_h - overlay_camera_A_h : camera_A_h,camera_A_w - overlay_camera_A_w : camera_A_w]= overlay_camera_A
      camera_B[camera_B_h - overlay_camera_B_h : camera_B_h,camera_B_w - overlay_camera_B_w : camera_B_w]= overlay_camera_B
      mode_name = 'Ovly'
    #-----------------------------------------------------------------------------------
    new_frame = np.concatenate((camera_A,camera_B),axis=0)
    #new_frame_noisy = np.concatenate((camera_A,camera_B),axis=1)
    #new_frame = cv.fastNlMeansDenoisingColored(new_frame_noisy,None,3,3,7,21)
    ##############################################################################
    end = time.time()
    ##########################fps calculation and display as overlay old##############
    time_elapsed = end - start
    fps = 1 // time_elapsed
    cv.putText(new_frame,'fps:' + str(fps),(300,30),font,1,(255,0,0),2,cv.LINE_AA)
    cv.putText(new_frame,'fps:' + str(fps),(split_len+300,30),font,1,(255,0,0),2,cv.LINE_AA)
    cv.putText(new_frame,'X' + str(zoom_factor),(450,30),font,1,(0,0,255),2,cv.LINE_AA)
    cv.putText(new_frame,'X' + str(zoom_factor),(split_len+450,30),font,1,(0,0,255),2,cv.LINE_AA)
    cv.putText(new_frame, mode_name,(550,30),font,1,(0,255,0),2,cv.LINE_AA)
    cv.putText(new_frame, mode_name,(split_len+550,30),font,1,(0,255,0),2,cv.LINE_AA)
    ##############################################################################
    cv.imshow('picam',new_frame)
    #cv.imshow('picam',frame)

    key = cv.waitKey(1)
    #if cv.waitKey(1)==ord('m'):#zoom in:
    #   zoom_state = 1
    #if cv.waitKey(1)==ord('n'):#zoom in
    #   zoom_state = 0
    #if cv.waitKey(1)==ord('q'):
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
    if key==ord('x'):#mode change:
       if mode_select == 2:
         mode_select = 0
       else:  
         mode_select +=1 
    if key==ord('q'):
        break
cam.release()
cv.destroyAllWindows()


