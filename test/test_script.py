import cv2 as cv
import numpy as np
import time
print(cv.__version__)

font = cv.FONT_HERSHEY_SIMPLEX

#-----------------------------------------------------
# for pc display
#-----------------------------------------------------
#dispW=1280
#dispH=720
#dispW=int(1920 * 0.5) # WIDTH OF OUTPUT IMAGE
#dispH=int(1080 * 0.5) # HEIGHT OF ROTATED OUTPUT IMAGE
#flip=2 # vertically up oriented, imgB+imgA
#flip=0 # vertically down oriented, imgB+imgA

#dispW=int(1080 * 0.5) # WIDTH OF ROTATED OUTPUT IMAGE
#dispH=int(1920 * 0.5) # HEIGHT OF ROTATED OUTPUT IMAGE
#flip=3 # horizontally left oriented, imgB+imgA
#flip=1 # horizontally right oriented, imgB+imgA

#------------------------------------------------------
# for vr display
#------------------------------------------------------
#dispW=int(1080 * 0.5) # WIDTH OF ROTATED OUTPUT IMAGE
#dispH=int(1920 * 0.5) # HEIGHT OF ROTATED OUTPUT IMAGE
dispW=1080 # WIDTH OF ROTATED OUTPUT IMAGE
dispH=1920 # HEIGHT OF ROTATED OUTPUT IMAGE
flip=3
 
# gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "video/x-raw(memory:NVMM),width=3264,height=2464,framerate=$FRAMERATE/1" ! nvvidconv flip-method=3 ! nvoverlaysink
#camSet='nvarguscamerasrc sensor-id=0 ! "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=$30/1" ! nvvidconv ! nvoverlaysink'
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#need to denoise
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=1848, format=NV12, framerate=28/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! vaapipostproc ! video/x-raw, denoise = 5 ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv.VideoCapture(camSet)

########################brightness and contrast function##############################
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
######################################################################################

zoom_factor = 1.0
frame_count = 0
mode_select = 0
#mode_name = 'Norm'
b = 0
c = 0
while True:
    mode_name = 'Norm'
    start = time.time()
    ret, frame = cam.read()
    ##########################camera output position correction####################### 
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    if frame_w > frame_h:
      #print('horizontal')
      split_len = frame_w//2
      camera_A = frame[ : , split_len : ]
      camera_A = cv.flip(camera_A, 0)
      camera_A_h = camera_A.shape[0]
      camera_A_w = camera_A.shape[1]
      camera_B = frame[ : , : split_len ]
      camera_B = cv.flip(camera_B, 0)
      camera_B_h = camera_B.shape[0]
      camera_B_w = camera_B.shape[1]
    else:
      #print('vertical')
      split_len = frame_h//2
      camera_A = frame[ split_len : , : ]
      camera_A = cv.flip(camera_A, 1)
      camera_A = cv.flip(camera_A, 0)
      camera_A_h = camera_A.shape[0]
      camera_A_w = camera_A.shape[1]
      camera_B = frame[ : split_len , : ]
      camera_B = cv.flip(camera_B, 1)
      camera_B = cv.flip(camera_B, 0)
      camera_B_h = camera_B.shape[0]
      camera_B_w = camera_B.shape[1]

    ###############################edge extraction####################################
    #----------------------------------edge overlay-----------------------------------
    if mode_select== 2: #edgemode
      mode_name = 'edge'
      gray_camera_A = cv.cvtColor(camera_A, cv.COLOR_BGR2GRAY)
      gray_camera_A = apply_brightness_contrast(gray_camera_A,0,32)
      gray_camera_B = cv.cvtColor(camera_B, cv.COLOR_BGR2GRAY)
      gray_camera_B = apply_brightness_contrast(gray_camera_B,0,32)
      gray_blur_A = cv.GaussianBlur(gray_camera_A,(3,3),0)
      gray_blur_B = cv.GaussianBlur(gray_camera_B,(3,3),0)
      edges_camera_A = cv.Canny(gray_blur_A,80,200)
      edges_camera_B = cv.Canny(gray_blur_B,80,200)
      edges_camera_A = cv.cvtColor(edges_camera_A, cv.COLOR_GRAY2BGR)
      edges_camera_B = cv.cvtColor(edges_camera_B, cv.COLOR_GRAY2BGR)
      camera_A_edges = cv.addWeighted(camera_A, 1, edges_camera_A, 0.4, 1.2 )
      camera_B_edges = cv.addWeighted(camera_B, 1, edges_camera_B, 0.4, 1.2 )
      camera_A = camera_A_edges
      camera_B = camera_B_edges
    ###############################creating overlay###################################
    overlay_camera_A = cv.resize(camera_A,(int(camera_A_w//2.5) , int(camera_A_h//2.5)),interpolation = cv.INTER_LINEAR )
    overlay_camera_A_h = overlay_camera_A.shape[0]
    overlay_camera_A_w = overlay_camera_A.shape[1]
    overlay_camera_B = cv.resize(camera_B,(int(camera_B_w//2.5) , int(camera_B_h//2.5)),interpolation = cv.INTER_LINEAR )
    overlay_camera_B_h = overlay_camera_B.shape[0]
    overlay_camera_B_w = overlay_camera_B.shape[1]
    ###############################zoom in,out #######################################
    zoom_factor = round(zoom_factor,1)
    crop_A_w = camera_A_w // zoom_factor
    crop_B_w = camera_B_w // zoom_factor
    crop_A_h = camera_A_h // zoom_factor
    crop_B_h = camera_B_h // zoom_factor
    crop_frame_A = camera_A[int(camera_A_w // 2 - crop_A_w // 2):int(camera_A_w // 2 + crop_A_w // 2), int(camera_A_h // 2 - crop_A_h // 2):int(camera_A_h // 2 + crop_A_h // 2)]
    crop_frame_B = camera_B[int(camera_B_w // 2 - crop_B_w // 2):int(camera_B_w // 2 + crop_B_w // 2), int(camera_B_h // 2 - crop_B_h // 2):int(camera_B_h // 2 + crop_B_h // 2)]
    camera_A = cv.resize(crop_frame_A,(camera_A_w,camera_A_h),interpolation = cv.INTER_LANCZOS4 )
    camera_B = cv.resize(crop_frame_B,(camera_B_w,camera_B_h),interpolation = cv.INTER_LANCZOS4 )
    #---------------------------------------------add overlay---------------------------
    if mode_select == 1:# overlay mode
      #camera_A[camera_A_h - overlay_camera_A_h : camera_A_h,camera_A_w - overlay_camera_A_w : camera_A_w]= overlay_camera_A
      #camera_B[camera_B_h - overlay_camera_B_h : camera_B_h,camera_B_w - overlay_camera_B_w : camera_B_w]= overlay_camera_B
      camera_A[ 0 : overlay_camera_A_h , 0  : overlay_camera_A_w ]= overlay_camera_A
      camera_B[ 0 : overlay_camera_B_h , 0  : overlay_camera_B_w ]= overlay_camera_B
      mode_name = 'Ovly'
    #-----------------------------------------------------------------------------------
     
    #############################radial distortion######################################
    ####################################################################################

    if frame_w > frame_h:
      new_frame = np.concatenate((camera_A,camera_B),axis=1)
    else:
      new_frame = np.concatenate((camera_A,camera_B),axis=0)
    ####################################################################################
    ##################################glare removal#####################################
    if mode_select == 3:# no glare mode
      mode_name = 'NoGlr'
      new_frame_hsv = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)
      h, s, v = cv.split(new_frame_hsv)
      ret,v = cv.threshold(v,200,255,cv.THRESH_TRUNC)
      new_frame = cv.merge((h,s,v))
      new_frame = cv.cvtColor(new_frame, cv.COLOR_HSV2BGR)
      #-------------
      #new_frame_blur = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
      #new_frame_blur = cv.medianBlur(new_frame_gray,5)
      #ret,th1 = cv.threshold(new_frame_blur,180,255,cv.THRESH_BINARY)
      #new_frame = cv.inpaint(new_frame,th1,9,cv.INPAINT_NS)
    ####################################################################################
    
    ###############################brightness and contrast control######################
   
    new_frame = apply_brightness_contrast(new_frame,b,c)
    
    ####################################################################################
    end = time.time()
    ##########################fps calculation and display as overlay old################
    time_elapsed = end - start
    fps = 1 // time_elapsed
    cv.putText(new_frame,'fps:' + str(fps),(300,30),font,1,(255,0,0),1,cv.LINE_AA)
    cv.putText(new_frame,'fps:' + str(fps),(split_len+300,30),font,1,(255,0,0),1,cv.LINE_AA)
    cv.putText(new_frame,'X' + str(zoom_factor),(450,30),font,1,(0,0,255),1,cv.LINE_AA)
    cv.putText(new_frame,'X' + str(zoom_factor),(split_len+450,30),font,1,(0,0,255),1,cv.LINE_AA)
    cv.putText(new_frame,'brightness:' + str(b),(400,650),font,1,(255,0,0),1,cv.LINE_AA)
    cv.putText(new_frame,'brightness:' + str(b),(split_len+400,650),font,1,(255,0,0),1,cv.LINE_AA)
    cv.putText(new_frame,'Contrast:' + str(c),(400,700),font,1,(0,0,255),1,cv.LINE_AA)
    cv.putText(new_frame,'Contrast:' + str(c),(split_len+400,700),font,1,(0,0,255),1,cv.LINE_AA)
    cv.putText(new_frame, mode_name,(550,30),font,1,(0,255,0),1,cv.LINE_AA)
    cv.putText(new_frame, mode_name,(split_len+550,30),font,1,(0,255,0),1,cv.LINE_AA)
    ####################################################################################
    cv.imshow('picam',new_frame)
    #cv.imshow('picam',frame)

    key = cv.waitKey(1)
    # brightness and contrast control
    if key==ord('w'):
      b+=10
    if key==ord('s'):
      b-=10 
    if key==ord('a'):
      c+=10 
    if key==ord('d'):
      c-=10 
    
    if key==ord('m'):#zoom in:
      if zoom_factor >= 8.0:
        zoom_factor = 8.0
      else:  
        zoom_factor = zoom_factor + 0.2
    if key==ord('n'):#zoom in
      if zoom_factor <= 1.0:
        zoom_factor = 1.0
      else:  
        zoom_factor = zoom_factor - 0.2
    if key==ord('x'):#mode change:+
      if mode_select == 3:
        mode_select = 0
      else:  
        mode_select +=1 #
    if key==ord('m') and key==ord('m'):#zoom in:
      print('success')
    if key==ord('q'):
      break
cam.release()
cv.destroyAllWindows()


