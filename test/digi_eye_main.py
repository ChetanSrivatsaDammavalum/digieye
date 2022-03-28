import cv2 as cv
import numpy as np
import time
print(cv.__version__)

font = cv.FONT_HERSHEY_SIMPLEX

#****************NEED TO IMPLEMENT THREADDING AND MULTI PROCESSING****************

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
 
#------------------------------------------------------
# Create Image stream pipeline
#------------------------------------------------------

#need to denoise
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=1848, format=NV12, framerate=28/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv.VideoCapture(camSet)

#------------------------------------------------------
# Image Overlaying 
#------------------------------------------------------

def apply_img_overlay(cam_img, scale_factor, flip):
  #Function to overlay windowed mini-image on input image

  h, w = cam_img.shape[:2]
  overlay_img = cv.resize(cam_img,(int(w//scale_factor) , int(h//scale_factor)),interpolation = cv.INTER_LINEAR )
  overlay_h, overlay_w = overlay_img.shape[:2]
  if flip == 1:
    cam_img[ 0 : overlay__h , 0  : overlay_w ]= overlay_img
  else:
    cam_img[ h - overlay_h : h , 0  : overlay_w ]= overlay_img
  
  return cam_img

#------------------------------------------------------
# Image de-glareing
#------------------------------------------------------

def remove_image_glare(cam_img):
  #Function to remove glare spots from image

  cam_img_hsv = cv.cvtColor(cam_img, cv.COLOR_BGR2HSV)
  h, s, v = cv.split(cam_img_hsv)
  ret,v = cv.threshold(v,210,255,cv.THRESH_TRUNC)
  img_glr = cv.merge((h,s,v))
  img_glr = cv.cvtColor(img_glr, cv.COLOR_HSV2BGR)
  
  return img_glr

#------------------------------------------------------
# Image Distortion 
#------------------------------------------------------

def apply_radial_distortion(cam_img, camera_mtx, distortion_mtx):
  #Function to introduce barrel distortion to the input image  

  h, w = cam_img.shape[:2]
  camera_mtx[0,2] = w // 2
  camera_mtx[1,2] = h // 2
  newcameramtx, roi=cv.getOptimalNewCameraMatrix(camera_mtx,distortion_mtx,(w,h),1,(w,h))
  img_dist = cv.undistort(cam_img, camera_mtx, distortion_mtx, None, newcameramtx)
  
  return img_dist

#------------------------------------------------------
# Image Brightness and Contrast control
#------------------------------------------------------

def apply_brightness_contrast(cam_img, brightness = 0, contrast = 0):
  #Function to change the brightness and/or contrast of input image
  #---------Need to set limits and increamentsteps------- 
       
  if brightness != 0:
    if brightness > 0:
        shadow = brightness
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + brightness
    alpha_b = (highlight - shadow)/255
    gamma_b = shadow
    
    img_bnc = cv.addWeighted(cam_img, alpha_b, cam_img, 0, gamma_b)
  else:
    img_bnc = cam_img.copy()
  
  if contrast != 0:
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    
    img_bnc = cv.addWeighted(img_bnc, alpha_c, img_bnc, 0, gamma_c)

  return img_bnc

#------------------------------------------------------
# Parameters
#------------------------------------------------------

zoom_factor = 1.0
frame_count = 0
mode_select = 0
overlay_scale = 2.5
#mode_name = 'Norm'
b = 0
c = 0
# distortion parameters
mtx = np.array([[ 1.70000000e+06, 0.00000000e+00, 0.00000000e+00 ],[ 0.00000000e+00, 1.70000000e+06, 0.00000000e+00 ],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 1.70000000e+06 , 1.00000000e+05 , 0.00000000e+00, 0.00000000e+00, 1.00000000e+04 ]])


while True:
    mode_name = 'Norm'
    start = time.time()
    ret, frame = cam.read()

    #-----------------------------------------Pre-Processing---------------------------------------
    
    # Camera image position and orientation correction 
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    if frame_w > frame_h:   #horizontal
      split_len = frame_w//2
      camera_A = frame[ : , split_len : ]
      camera_A_h = camera_A.shape[0]
      camera_A_w = camera_A.shape[1]
      camera_B = frame[ : , : split_len ]
      camera_B_h = camera_B.shape[0]
      camera_B_w = camera_B.shape[1]
    else:                   #vertical
      split_len = frame_h//2
      camera_A = frame[ split_len : , : ]
      camera_A = cv.flip(camera_A, -1)
      camera_A_h = camera_A.shape[0]
      camera_A_w = camera_A.shape[1]
      camera_B = frame[ : split_len , : ]
      camera_B = cv.flip(camera_B, -1)
      camera_B_h = camera_B.shape[0]
      camera_B_w = camera_B.shape[1]

    # Image zoom in/out
    zoom_factor = round(zoom_factor,1)
    crop_A_w = camera_A_w // zoom_factor
    crop_B_w = camera_B_w // zoom_factor
    crop_A_h = camera_A_h // zoom_factor
    crop_B_h = camera_B_h // zoom_factor
    crop_frame_A = camera_A[int(camera_A_w // 2 - crop_A_w // 2):int(camera_A_w // 2 + crop_A_w // 2), int(camera_A_h // 2 - crop_A_h // 2):int(camera_A_h // 2 + crop_A_h // 2)]
    crop_frame_B = camera_B[int(camera_B_w // 2 - crop_B_w // 2):int(camera_B_w // 2 + crop_B_w // 2), int(camera_B_h // 2 - crop_B_h // 2):int(camera_B_h // 2 + crop_B_h // 2)]
    camera_A = cv.resize(crop_frame_A,(camera_A_w,camera_A_h),interpolation = cv.INTER_LANCZOS4 )
    camera_B = cv.resize(crop_frame_B,(camera_B_w,camera_B_h),interpolation = cv.INTER_LANCZOS4 )

    #-----------------------------------------Modes------------------------------------------------

    # Overlay mode
    if mode_select == 1:
      mode_name = 'Ovly'
      camera_A = apply_img_overlay(camera_A, overlay_scale, flip)
      camera_B = apply_img_overlay(camera_B, overlay_scale, flip)
      
    # Edge highlight mode
    #*********need to streamline by creating function*********
    if mode_select== 2:
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
     
    # No Glare mode
    if mode_select == 3:# no glare mode
      mode_name = 'NoGlr'
      new_frame = remove_image_glare(new_frame)
    
    #-----------------------------------------Post-Processing--------------------------------------

    # Apply radial distortion on image
    camera_A = apply_radial_distortion(camera_A, mtx, dist)
    camera_B = apply_radial_distortion(camera_B, mtx, dist)
    
    # Concatenate left and right inages as single image output
    if frame_w > frame_h:
      new_frame = np.concatenate((camera_A,camera_B),axis=1)
    else:
      new_frame = np.concatenate((camera_A,camera_B),axis=0)
      
    # Apply Brightness and Contrast values 
    new_frame = apply_brightness_contrast(new_frame,b,c)
    
    # fps calculation
    end = time.time()
    time_elapsed = end - start
    fps = 1 // time_elapsed

    # Display status texts
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
    
    #-------------------------------------------User Input-----------------------------------------
    #**********need to optimize***********
    key = cv.waitKey(1)
    # brightness and contrast controls
    if key==ord('w'):
      b+=10
    if key==ord('s'):
      b-=10 
    if key==ord('a'):
      c+=10 
    if key==ord('d'):
      c-=10 
    
    # Mode controls
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


