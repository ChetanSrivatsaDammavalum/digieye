import cv2 as cv
import numpy as np
import time
from threading import Thread
#print(cv.__version__)

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
#dispH=1920 # HEIGHT OF ROTATED OUTPUT IMAGE
dispH=2160 # HEIGHT OF ROTATED OUTPUT IMAGE
flip=3
 
   
#------------------------------------------------------
# Image Zooming
#------------------------------------------------------

def apply_image_zoom(cam_img, zoom_factor):
  #Function to  zoom in/out of input image
  h, w = cam_img.shape[:2]
  crop_img_w = w // zoom_factor
  crop_img_h = h // zoom_factor
  crop_img = cam_img[int(w // 2 - crop_img_w // 2):int(w // 2 + crop_img_w // 2), int(h // 2 - crop_img_h // 2):int(h // 2 + crop_img_h // 2)]
  img_zoom = cv.resize(crop_img,(w,h),interpolation = cv.INTER_LANCZOS4 )

  return img_zoom

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
# Image Edge Highlighting
#------------------------------------------------------ 

def apply_edge_highlight(cam_img):
  #Function to extract edges from input image and highlight the edges
  gray_img = cv.cvtColor(cam_img, cv.COLOR_BGR2GRAY)
  gray_img = apply_brightness_contrast(gray_img,0,32)
  gray_img = cv.GaussianBlur(gray_img,(3,3),0) 
  img_edges = cv.Canny(gray_img,80,200)
  img_edges = cv.cvtColor(img_edges, cv.COLOR_GRAY2BGR)
  img_edges = cv.addWeighted(cam_img, 1, img_edges, 0.4, 1.2 )

  return img_edges

#------------------------------------------------------
# Image de-glareing
#------------------------------------------------------

def remove_image_glare(cam_img):
  #Function to remove glare spots from image

  cam_img_hsv = cv.cvtColor(cam_img, cv.COLOR_BGR2HSV)
  h, s, v = cv.split(cam_img_hsv)
  ret,v = cv.threshold(v,190,255,cv.THRESH_TRUNC)
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
# Add Status text
#------------------------------------------------------

def put_text_image(cam_img, flip, fps, zoom_factor, mode_name, brightness, contrast, flag):
  #Add text to images
  if flip == 0:
    cv.putText(cam_img,'fps:' + str(fps),(550,30),font,1,(255,0,0),1,cv.LINE_AA)
    cv.putText(cam_img,'X' + str(zoom_factor),(700,30),font,1,(0,0,255),1,cv.LINE_AA)
    cv.putText(cam_img, mode_name,(800,30),font,1,(0,255,0),1,cv.LINE_AA)
    if flag==1:
      cv.putText(cam_img,'brightness:' + str(brightness),(400,700),font,1,(255,0,0),1,cv.LINE_AA)
    if flag==2:
      cv.putText(cam_img,'Contrast:' + str(contrast),(400,700),font,1,(0,0,255),1,cv.LINE_AA)
  else:
    h, w = cam_img.shape[:2]
    center = (w/2, h/2)
    rotate_matrix = cv.getRotationMatrix2D(center=center, angle=90, scale=1)
    cam_img = cv.warpAffine(src=cam_img, M=rotate_matrix, dsize=(h, w))

    #cam_img = cv.flip(cam_img, -1)
    #cam_img = cv.rotate(cam_img, cv.ROTATE_90CLOCKWISE)
    cv.putText(cam_img,'fps:' + str(fps),(550,30),font,1,(255,0,0),1,cv.LINE_AA)
    cv.putText(cam_img,'X' + str(zoom_factor),(700,30),font,1,(0,0,255),1,cv.LINE_AA)
    cv.putText(cam_img, mode_name,(800,30),font,1,(0,255,0),1,cv.LINE_AA)
    if flag==1:
      cv.putText(cam_img,'brightness:' + str(brightness),(400,700),font,1,(255,0,0),1,cv.LINE_AA)
    if flag==2:
      cv.putText(cam_img,'Contrast:' + str(contrast),(400,700),font,1,(0,0,255),1,cv.LINE_AA)
    rotate_matrix = cv.getRotationMatrix2D(center=center, angle=90, scale=1)
    cam_img = cv.warpAffine(src=cam_img, M=rotate_matrix, dsize=(w, h))
    #cam_img = cv.flip(cam_img, -1)
    #cam_img = cv.rotate(cam_img, cv.ROTATE_90COUNTERCLOCKWISE)

  return cam_img

#------------------------------------------------------
# Parameters
#------------------------------------------------------
fps = 28
zoom_factor = 1.0  
frame_count = 0
mode_select = 0
overlay_scale = 2.3
mode_name = 'Norm'
flipped = 0
b_c_flag = 0
b = 0
c = 0
# distortion parameters
mtx = np.array([[ 1.70000000e+06, 0.00000000e+00, 0.00000000e+00 ],[ 0.00000000e+00, 1.70000000e+06, 0.00000000e+00 ],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 1.70000000e+06 , 1.00000000e+05 , 0.00000000e+00, 0.00000000e+00, 1.00000000e+04 ]])

#------------------------------------------------------
# Create Image stream pipeline
#------------------------------------------------------

#need to denoise
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=1848, format=NV12, framerate=28/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv.VideoCapture(camSet)

##########################Multi Threading#########################
class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src):
        self.stream = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
 
while True:
    #mode_name = 'Norm'
    start = time.time()
    #video_getter = VideoGet(camSet).start()
    ret, frame = cam.read()
    #frame = video_getter.frame

    #-----------------------------------------Pre-Processing---------------------------------------
    
    # Camera image position-orientation correction and text addition 
    frame_h, frame_w = frame.shape[:2]
    if frame_w > frame_h:   #horizontal
      split_len = frame_w//2
      flipped = 0
      camera_A = frame[ : , split_len : ]
      camera_A_h, camera_A_w = camera_A.shape[:2]
      camera_B = frame[ : , : split_len ]
      camera_B_h, camera_B_w = camera_B.shape[:2]
    else:                   #vertical
      split_len = frame_h//2
      flipped = 1
      camera_A = frame[ split_len : , : ]
      camera_A = cv.flip(camera_A, -1)
      camera_A_h, camera_A_w = camera_A.shape[:2]
      camera_B = frame[ : split_len , : ]
      camera_B = cv.flip(camera_B, -1)
      camera_B_h, camera_B_w = camera_B.shape[:2]

    # Image zoom in/out
    zoom_factor = round(zoom_factor,1)
    camera_A = apply_image_zoom(camera_A, zoom_factor)
    camera_B = apply_image_zoom(camera_B, zoom_factor)

    #---------------------------------------------------------------------------------------------
    # Display status texts
    #*************Need to change to a GUI package*************************************************
    camera_A = put_text_image(camera_A,flipped,fps, zoom_factor, mode_name, b, c, b_c_flag)
    camera_B = put_text_image(camera_B,flipped,fps, zoom_factor, mode_name, b, c, b_c_flag)

    #-----------------------------------------Modes------------------------------------------------

    # Overlay mode
    if mode_select == 1:
      mode_name = 'Ovly'
      camera_A = apply_img_overlay(camera_A, overlay_scale, flip)
      camera_B = apply_img_overlay(camera_B, overlay_scale, flip)
      
    # Edge highlight mode
    if mode_select== 2:
      mode_name = 'edge'
      camera_A = apply_edge_highlight(camera_A)
      camera_B = apply_edge_highlight(camera_B)
     
    # No Glare mode
    if mode_select == 3:# no glare mode
      mode_name = 'NoGlr'
      camera_A = remove_image_glare(camera_A)
      camera_B = remove_image_glare(camera_B)
    #-----------------------------------------Post-Processing--------------------------------------

    # Apply Brightness and Contrast values 
    camera_A = apply_brightness_contrast(camera_A,b,c)
    camera_B = apply_brightness_contrast(camera_B,b,c)
    
    # Apply radial distortion on image
    camera_A = apply_radial_distortion(camera_A, mtx, dist)
    camera_B = apply_radial_distortion(camera_B, mtx, dist)
      
    # fps calculation
    end = time.time()
    time_elapsed = end - start
    fps = 1 // time_elapsed

    # Concatenate left and right inages as single image output
    if frame_w > frame_h:
      new_frame = np.concatenate((camera_A,camera_B),axis=1)
    else:
      new_frame = np.concatenate((camera_A,camera_B),axis=0)
    
    cv.imshow('picam',new_frame)
    
    #-------------------------------------------User Input-----------------------------------------
    key = cv.waitKey(10)
    
    # Mode and function control
    #if key==ord('m') or key==ord('n') or key==ord('b'):#function input
    if key==ord('m') or key==ord('n') or key==ord('b') or key==ord('c') or key==ord('x'):#function input

      #if key==ord('m') and key==ord('b'):#brightness flag
      #  b_c_flag = 1
      #  print('brightness mode')
      
      #elif key==ord('n') and key==ord('b'):#contrast flag
      #  b_c_flag = 2
      #  print('contrast mode')

      if key==ord('b'):#brightness flag
        b_c_flag = 1
        print('brightness mode')
      
      elif key==ord('c'):#contrast flag
        b_c_flag = 2
        print('contrast mode')
    

      else:
        if key==ord('m'):#zoom-in/brightness-contrast increment
          if b_c_flag != 0:
            if b_c_flag == 1:
              b+=10
              print('brightness increase')
            else:
              c+=10
              print('Contrast increase')
          elif zoom_factor >= 8.0:
            zoom_factor = 8.0
            print('zoom increase')
          else:  
            zoom_factor = zoom_factor + 0.2
            print('zoom increase')

        elif key==ord('n'):#zoom-out/brightness-contrast decrement
          if b_c_flag != 0:
            if b_c_flag == 1:
              b-=10
              print('brightness decrease')
            else:
              c-=10
              print('Contrast decrease')
          elif zoom_factor <= 1.0:
            zoom_factor = 1.0
            print('zoom decrease')
          else:  
            zoom_factor = zoom_factor - 0.2
            print('zoom decrease')

        else: #mode change  
        #if key==ord('x'):
          if b_c_flag != 0:
            b_c_flag = 0
            print('exit brightness and contrast setting')
          elif mode_select == 3:
            mode_select = 0
            print('change mode')
          else:  
            mode_select +=1 
            print('change mode')
    
    # Application termination
    if key==ord('q'): #or video_getter.stopped:#exit application
      #video_getter.stop()
      break
cam.release()
cv.destroyAllWindows()


