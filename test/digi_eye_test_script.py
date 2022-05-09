import cv2 as cv
import numpy as np
from threading import Thread
#from multiprocessing import Process, Queue
import multiprocessing as mp
import time
import math
import easyocr
import PIL 
from PIL import ImageDraw, Image
print(PIL.__version__)
print(cv.__version__)


font = cv.FONT_HERSHEY_SIMPLEX

reader = easyocr.Reader(['de','en'])

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
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=1848, format=NV12, framerate=28/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videomedian ! videoconvert ! videoscale ! video/x-raw,format=I420 ! videomedian filtersize=9 ! videoconvert ! video/x-raw, format=BGR ! appsink'
#camSet='v4l2src device=/dev/video1 ! videoconvert ! videoscale ! video/x-raw,format=I420 ! videomedian filtersize=9 ! videoconvert ! video/x-raw,format=BGR ! queue ! videoconvert ! appsink'
#camSet='v4l2src name=cam_src ! videoconvert ! videoscale ! video/x-raw,format=I420 ! videomedian filtersize=9 ! videoconvert ! video/x-raw,format=BGR ! queue ! videoconvert ! appsink'
#cam= cv.VideoCapture(camSet)

#device=/dev/video0

#gst-launch-1.0 v4l2src name=cam_src ! videoconvert ! videoscale ! video/x-raw,format=I420 ! videomedian filtersize=9 ! videoconvert ! video/x-raw,format=RGB !  queue ! videoconvert ! ximagesink name=img_origin


#-----------------------------------------------------
#Capturing Camera stream
#-----------------------------------------------------
# sequential method
#cam = cv.VideoCapture(0)
# get height, width and frame count of the video
#width, height = (int(cam.get(cv.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv.CAP_PROP_FRAME_HEIGHT)))
#fps = int(cam.get(cv.CAP_PROP_FPS))

# threading method
class VideoStreamWidget(object):
  def __init__(self, src=0):
    self.capture = cv.VideoCapture(src)
    self.status, self.frame = self.capture.read()
    # Start the thread to read frames from the video stream
    self.thread = Thread(target=self.update, args=())
    self.thread.daemon = True
    self.thread.start()
    

  def update(self):
    # Read the next frame from the stream in a different thread
    while True:
      if self.capture.isOpened():
        self.status, self.frame = self.capture.read()
        time.sleep(.01)


  def show_frame(self, img, key):
    # Display frames in main program
    # cv.imshow('frame', self.frame)
    cv.imshow('frame', img)
    # key = cv.waitKey(1)
    if key == ord('q'):
      self.capture.release()
      #if self.thread.is_alive():
      #self.thread.join()
      time.sleep(1)
      cv.destroyAllWindows()
      #self.thread.join()
      exit(1)

#-----------------------------------------------------
# Multiprocessing queue sample
#-----------------------------------------------------
# import scipy
# 
# def process_images(q):
#     while not q.empty():
#         im = q.get()
#         # Do stuff with item from queue
# 
# 
# def read_images(q, files):
#     for f in files:
#         q.put(scipy.misc.imread(f))
# 
# if __name__ == '__main__':
#     q = Queue()
# 
#     producer = Process(target=read_images, args=(q, files))
#     producer.start()
#     consumer = Process(target=process_images, args=(q, ))
#     consumer.start()
#
#------------------------------------------------------
# Image Overlaying 
#------------------------------------------------------

def apply_img_overlay(cam_img, cam_img_src, scale_factor, pos):
  #Function to overlay windowed mini-image on input image

  h, w = cam_img.shape[:2]
  h_src, w_src = cam_img_src.shape[:2]
  overlay_img = cv.resize(cam_img_src,(int(w_src//scale_factor) , int(h_src//scale_factor)),interpolation = cv.INTER_LINEAR )
  overlay_h, overlay_w = overlay_img.shape[:2]
  if pos == 0:
    cam_img[ 0 : overlay_h , 0  : overlay_w ]= overlay_img
  elif pos == 1:
    cam_img[ h - overlay_h : h , 0  : overlay_w ]= overlay_img
  elif pos == 2:
    cam_img[ h - overlay_h : h , w - overlay_w  : w ]= overlay_img
  else:
    cam_img[ 0 : overlay_h , w - overlay_w  : w ]= overlay_img
  
  return cam_img

#------------------------------------------------------
# Image de-glareing
#------------------------------------------------------

def remove_image_glare(cam_img):
  #Function to remove glare spots from image

  cam_img_hsv = cv.cvtColor(cam_img, cv.COLOR_BGR2HSV)
  h, s, v = cv.split(cam_img_hsv)
  ret,v = cv.threshold(v,180,255,cv.THRESH_TRUNC)
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
# Bubble magnification
#------------------------------------------------------

def bubble(cam_img, radius= 400, scale =2, amount =-2):
  # grab the dimensions of the image
  h, w = cam_img.shape[:2]
  center_y = h//2
  center_x = w//2

  # set up the x and y maps as float32
  flex_x = np.zeros((h, w), np.float32)
  flex_y = np.zeros((h, w), np.float32)

  # create map with the barrel pincushion distortion formula
  for y in range(h):
      delta_y = scale * (y - center_y)
      for x in range(w):
          # determine if pixel is within an ellipse
          delta_x = scale * (x - center_x)
          distance = delta_x * delta_x + delta_y * delta_y
          if distance >= (radius * radius):
              flex_x[y, x] = x
              flex_y[y, x] = y
          else:
              factor = 1.0
              if distance > 0.0:
                  # factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), -amount)
                  factor = math.pow( math.sqrt(distance) / radius , -amount/2 )
              flex_x[y, x] = factor * delta_x / scale + center_x
              flex_y[y, x] = factor * delta_y / scale + center_y

  # do the remap  this is where the magic happens
  return cv.remap(cam_img, flex_x, flex_y, cv.INTER_LANCZOS4) 

#------------------------------------------------------
# OCR
#------------------------------------------------------

def draw_boxes(image, bounds, color='blue', width=1):
    draw = ImageDraw.Draw(Image.fromarray(image))
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        print(p0, p1 ,p2 ,p3)
        p21, p22 = np.array(p2)
        p31, p32 = np.array(p3)
        # print(p11,p22,p21,p22)
        # draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        # draw.line([ *p1, *p2 ], fill=color, width=width)
        cv.line(image,(int(p21),int(p22)),(int(p31),int(p32)),(255,0,0),2)
    return image 

def ocr_img(cam_img):
  bounds = reader.readtext(cam_img)
  if bounds:
    return draw_boxes(cam_img, bounds)
    # return bounds
  else:
    return cam_img
#------------------------------------------------------
# Parameters
#------------------------------------------------------

zoom_factor = 1.0
frame_count = 0
mode_select = 0
overlay_scale = 2.5
mode_name = 'Norm'
b = 0
c = 0
# distortion parameters
mtx = np.array([[ 1.70000000e+06, 0.00000000e+00, 0.00000000e+00 ],[ 0.00000000e+00, 1.70000000e+06, 0.00000000e+00 ],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 1.70000000e+06 , 1.00000000e+05 , 0.00000000e+00, 0.00000000e+00, 1.00000000e+04 ]])
pos = 0
if __name__ == '__main__':
  video_stream_widget = VideoStreamWidget()
  pool = mp.Pool(2)
  while True:
      #--------------------
      #normal sequential capture method
      start = time.time()
      #ret, frame = cam.read()
      #if not ret:
      #  time.sleep(1)
      #  continue
      #---------------------
      img = video_stream_widget.frame

      #-----------------------------------------Pre-Processing---------------------------------------
      
      # Camera image position and orientation correction 
      frame_h = img.shape[0]
      frame_w = img.shape[1]
      camera_A = img
      camera_B = img
      camera_A_h = camera_A.shape[0]
      camera_A_w = camera_A.shape[1]
      camera_B_h = camera_B.shape[0]
      camera_B_w = camera_B.shape[1]   
      
      camera_A_src = img
      camera_B_src = img

      split_len = frame_w
      

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
        camera_A = apply_img_overlay(camera_A, camera_A_src , overlay_scale, pos)
        camera_B = apply_img_overlay(camera_B, camera_B_src , overlay_scale, pos)
        
      # Edge highlight mode
      #*********need to streamline by creating function*********
      elif mode_select== 2:
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
      elif mode_select == 3:# no glare mode
        mode_name = 'NoGlr'
        camera_A = remove_image_glare(camera_A)
        camera_B = remove_image_glare(camera_B)
      
      # BW high mode image
      elif mode_select == 4:
        mode_name = 'BW'
        camera_A = cv.cvtColor(camera_A, cv.COLOR_BGR2GRAY)
        camera_B = cv.cvtColor(camera_B, cv.COLOR_BGR2GRAY)
      # BW high contrast mode image
      elif mode_select == 5:
        mode_name = 'BW high contrast'
        camera_A = cv.cvtColor(camera_A, cv.COLOR_BGR2GRAY)
        camera_B = cv.cvtColor(camera_B, cv.COLOR_BGR2GRAY)
        camera_A = apply_brightness_contrast(camera_A,10,32)
        ret,v = cv.threshold(camera_A,155,255,cv.THRESH_BINARY)
        camera_A = v
        camera_B = apply_brightness_contrast(camera_B,10,32)
        ret,v = cv.threshold(camera_B,155,255,cv.THRESH_BINARY)
        camera_B = v
      
      elif mode_select == 6:
        mode_name = 'WB high contrast'
        camera_A = cv.cvtColor(camera_A, cv.COLOR_BGR2GRAY)
        camera_B = cv.cvtColor(camera_B, cv.COLOR_BGR2GRAY)
        camera_A = apply_brightness_contrast(camera_A,10,32)
        ret,v = cv.threshold(camera_A,155,255,cv.THRESH_BINARY)
        camera_A = 255 - v
        camera_B = apply_brightness_contrast(camera_B,10,32)
        ret,v = cv.threshold(camera_B,155,255,cv.THRESH_BINARY)
        camera_B = 255 - v
      
      elif mode_select == 7:
        mode_name = 'bubble'
        # camera_A = bubble(camera_A, 400, 2, -2)
        # camera_B = bubble(camera_B, 400, 2, -2) 
        img_pair = [camera_A, camera_B]
        #pool = mp.Pool(2)
        # img_p = pool.map(ocr_img, img_pair)
        bubble_pair = pool.map(bubble, img_pair)
        #pool.close()
        #pool.join()
    
        camera_A = bubble_pair[0]
        camera_B = bubble_pair[1]

      elif mode_select == 8:
        mode_name = 'ocr'
        # start pocess
        camera_A = ocr_img(camera_A)
        camera_B = ocr_img(camera_B)

        # camera_A_bounds = ocr_img(camera_A)
        # if camera_A_bounds != 0:
        #   camera_A = draw_boxes(camera_A, camera_A_bounds)       
        # camera_B_bounds = ocr_img(camera_B)
        # if camera_B_bounds != 0:
        #   camera_B = draw_boxes(camera_B, camera_B_bounds)      

      else:
        mode_select = 0

      #-----------------------------------------Post-Processing--------------------------------------
      
      # fps calculation
      end = time.time()
      time_elapsed = end - start
      fps = 1 // time_elapsed
      
      # Display status texts
      cv.putText(camera_A,'fps:' + str(fps),(300,30),font,1,(255,0,0),1,cv.LINE_AA)
      cv.putText(camera_B,'fps:' + str(fps),(300,30),font,1,(255,0,0),1,cv.LINE_AA)
      cv.putText(camera_A,'X' + str(zoom_factor),(450,30),font,1,(0,0,255),1,cv.LINE_AA)
      cv.putText(camera_B,'X' + str(zoom_factor),(450,30),font,1,(0,0,255),1,cv.LINE_AA)
      cv.putText(camera_A,'brightness:' + str(b),(400,650),font,1,(255,0,0),1,cv.LINE_AA)
      cv.putText(camera_B,'brightness:' + str(b),(400,650),font,1,(255,0,0),1,cv.LINE_AA)
      cv.putText(camera_A,'Contrast:' + str(c),(400,700),font,1,(0,0,255),1,cv.LINE_AA)
      cv.putText(camera_B,'Contrast:' + str(c),(400,700),font,1,(0,0,255),1,cv.LINE_AA)
      cv.putText(camera_A, mode_name,(550,30),font,1,(0,255,0),1,cv.LINE_AA)
      cv.putText(camera_B, mode_name,(550,30),font,1,(0,255,0),1,cv.LINE_AA)
      
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
      
      # Display status texts
      #cv.putText(new_frame,'fps:' + str(fps),(300,30),font,1,(255,0,0),1,cv.LINE_AA)
      #cv.putText(new_frame,'fps:' + str(fps),(split_len+300,30),font,1,(255,0,0),1,cv.LINE_AA)
      #cv.putText(new_frame,'X' + str(zoom_factor),(450,30),font,1,(0,0,255),1,cv.LINE_AA)
      #cv.putText(new_frame,'X' + str(zoom_factor),(split_len+450,30),font,1,(0,0,255),1,cv.LINE_AA)
      #cv.putText(new_frame,'brightness:' + str(b),(400,650),font,1,(255,0,0),1,cv.LINE_AA)
      #cv.putText(new_frame,'brightness:' + str(b),(split_len+400,650),font,1,(255,0,0),1,cv.LINE_AA)
      #cv.putText(new_frame,'Contrast:' + str(c),(400,700),font,1,(0,0,255),1,cv.LINE_AA)
      #cv.putText(new_frame,'Contrast:' + str(c),(split_len+400,700),font,1,(0,0,255),1,cv.LINE_AA)
      #cv.putText(new_frame, mode_name,(550,30),font,1,(0,255,0),1,cv.LINE_AA)
      #cv.putText(new_frame, mode_name,(split_len+550,30),font,1,(0,255,0),1,cv.LINE_AA)
      
      ####################################################################################
      #cv.imshow('picam',new_frame)
      
      #-------------------------------------------User Input-----------------------------------------
      #**********need to optimize***********
      key = cv.waitKey(1)
      if key == ord('q'):
        pool.close()
        #pool.terminate()
        pool.join()
        time.sleep(1)
      video_stream_widget.show_frame(new_frame,key)
        
        
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
        if mode_select == 7:
          mode_select = 0
        else:  
          mode_select +=1 #
          if mode_select == 8:
            print('ocr mode')
      # if key==ord('m') and key==ord('m'):#zoom in:
        # print('success')
      # if key==ord('q'):
      #   break
  # cam.release()
  # cv.destroyAllWindows()
    

