import cv2 as cv
import numpy as np
from threading import Thread
#from multiprocessing import Queue
import multiprocessing as mp
import queue
import time
import math
import easyocr
import PIL 
from PIL import ImageDraw, Image
#import asyncio
print(PIL.__version__)
print(cv.__version__)

font = cv.FONT_HERSHEY_SIMPLEX

reader = easyocr.Reader(['de','en'])

#------------------------------------------------------
# Video stream thread
#------------------------------------------------------
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
    cv.imshow('frame', img)
    if key == ord('q'):
      self.capture.release()
      time.sleep(1)
      cv.destroyAllWindows()
      exit(1)

#------------------------------------------------------
# Image magnification
#------------------------------------------------------

def zoom_image(cam_img, zoom_factor):
  h, w = cam_img.shape[:2]
  crop_w = w//zoom_factor
  crop_h = h//zoom_factor
  crop_img = cam_img[int(w // 2 - crop_w // 2):int( w // 2 + crop_w // 2), int(h // 2 - crop_h // 2):int(h // 2 + crop_h // 2)]
  return cv.resize(crop_img,(w,h),interpolation = cv.INTER_LANCZOS4 )

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
# Inter pupil distance
#------------------------------------------------------
def ipd_img(cam_img, img_pos, h, w, crop_size, ipd):
  center_w = w // 2
  if img_pos == 'left':
    center_w = center_w - ipd 
    cam_img = cam_img[int( center_w - crop_size // 2):int( center_w + crop_size // 2), int(h // 2 - crop_size // 2):int(h // 2 + crop_size // 2)]
  if img_pos == 'right':
    center_w = center_w + ipd
    cam_img = cam_img[int( center_w - crop_size // 2):int( center_w + crop_size // 2), int(h // 2 - crop_size // 2):int(h // 2 + crop_size // 2)]
  return cam_img   


#------------------------------------------------------
# Image edge highlight
#------------------------------------------------------

# def edge_image(cam_img):
def edge_image(cam_img):
  edge_img = cv.cvtColor(cam_img, cv.COLOR_BGR2GRAY)
  edge_img = apply_brightness_contrast(edge_img,0,32)
  edge_img = cv.GaussianBlur(edge_img,(3,3),0)
  edge_img = cv.Canny(edge_img,80,200)
  edge_img = cv.cvtColor(edge_img, cv.COLOR_GRAY2BGR)
  return cv.addWeighted(cam_img, 1, edge_img, 0.4, 1.2 )

#------------------------------------------------------
# Bubble magnification
#------------------------------------------------------

def bubble(cam_img, radius= 0.5, scale =1 , amount =8):
  # grab the dimensions of the image
  h, w = cam_img.shape[:2]
  center_y = h//2
  center_x = w//2

  radius = int(radius*h)
  scale_a = 0.6
  scale_b = 2

  # set up the x and y maps as float32
  flex_x = np.zeros((h, w), np.float32)
  flex_y = np.zeros((h, w), np.float32)

  # create map with the barrel pincushion distortion formula
  for y in range(h):
      delta_y = (y - center_y)
      for x in range(w):
          # determine if pixel is within an ellipse
          delta_x = (x - center_x)
          distance = delta_x * delta_x + delta_y * delta_y
          if distance >= (radius * radius):
              flex_x[y, x] = x
              flex_y[y, x] = y
          else:
              factor = 1.0
              if distance > 0.0:
              #if distance > (0.2*h):
                  #distance = distance * scale_a * scale_a
                  factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
                  #factor = math.pow( math.sqrt(distance) / radius , -amount/2 )
                  #scale = scale_a 
              #else: 
                  #distance = distance * scale_b * scale_b
                  #factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
                  #scale = scale_b
              flex_x[y, x] = factor * delta_x / scale + center_x
              flex_y[y, x] = factor * delta_y / scale + center_y

  cam_img = cv.line(cam_img,(int(0.3*w),int(0.45*h)),(int(0.7*w),int(0.45*h)),(0,255,0),2)
  cam_img = cv.line(cam_img,(int(0.3*w),int(0.55*h)),(int(0.7*w),int(0.55*h)),(0,255,0),2)
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
        cv.line(image,(int(p21),int(p22)),(int(p31),int(p32)),(255,0,0),2)
    return image 

# *************888change with queues***********
def ocr_img(cam_img):
  bounds = reader.readtext(cam_img)
  #q.put('Process Done')
  return bounds
  
#------------------------------------------------------
# Image gray
#------------------------------------------------------

def bw_img(cam_img):
  cam_img = cv.cvtColor(cam_img, cv.COLOR_BGR2GRAY) 
  return cv.cvtColor(cam_img,cv.COLOR_GRAY2BGR)

#------------------------------------------------------
# Image black-white high contrast
#------------------------------------------------------

def bwhc_img(cam_img):
  cam_img = cv.cvtColor(cam_img, cv.COLOR_BGR2GRAY)
  cam_img = apply_brightness_contrast(cam_img,10,32)
  ret,v = cv.threshold(cam_img,155,255,cv.THRESH_BINARY)
  return cv.cvtColor(v, cv.COLOR_GRAY2BGR)

#------------------------------------------------------
# Image white-black high contrast
#------------------------------------------------------

def wbhc_img(cam_img):
  cam_img = cv.cvtColor(cam_img, cv.COLOR_BGR2GRAY)
  cam_img = apply_brightness_contrast(cam_img,10,32)
  ret,v = cv.threshold(cam_img,155,255,cv.THRESH_BINARY)
  v = cv.cvtColor(v, cv.COLOR_GRAY2BGR)
  return 255 - v  

#------------------------------------------------------
# Image overlaying
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

def apply_zoom_overlay(cam_img, zoom_factor):
  h = cam_img.shape[0]
  w = cam_img.shape[1]
  overlay_img = cam_img[int(h * 0.3) : int(h * 0.7) , 0  : w ]
  overlay_img = zoom_image(overlay_img, zoom_factor)
  cam_img[int(h * 0.3) : int(h * 0.7) , 0  : w] = overlay_img
  return cam_img

#------------------------------------------------------
# Overlay text on image
#------------------------------------------------------

# function to draw text detection region
def draw_border(cam_img, point1, point2, point3, point4, line_length, type):

    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4    
    if type == 'corner':
      cv.line(cam_img, (x1, y1), (x1 , y1 + line_length), (255, 0, 255), 2)  #-- top-left
      cv.line(cam_img, (x1, y1), (x1 + line_length , y1), (255, 0, 255), 2)

      cv.line(cam_img, (x2, y2), (x2 , y2 - line_length), (255, 0, 255), 2)  #-- bottom-left
      cv.line(cam_img, (x2, y2), (x2 + line_length , y2), (255, 0, 255), 2)

      cv.line(cam_img, (x3, y3), (x3 - line_length, y3), (255, 0, 255), 2)  #-- top-right
      cv.line(cam_img, (x3, y3), (x3, y3 + line_length), (255, 0, 255), 2)

      cv.line(cam_img, (x4, y4), (x4 , y4 - line_length), (255, 0, 255), 2)  #-- bottom-right
      cv.line(cam_img, (x4, y4), (x4 - line_length , y4), (255, 0, 255), 2)
    if type == 'edge':
      cv.line(cam_img, (x1, y1), (x1 - line_length, y1 ), (255, 0, 255), 2)  #-- top-edge
      cv.line(cam_img, (x1, y1), (x1 + line_length , y1), (255, 0, 255), 2)

      cv.line(cam_img, (x2, y2), (x2 - line_length , y2 ), (255, 0, 255), 2)  #-- bottom-edge
      cv.line(cam_img, (x2, y2), (x2 + line_length , y2), (255, 0, 255), 2)

      cv.line(cam_img, (x3, y3), (x3, y3 - line_length), (255, 0, 255), 2)  #-- right-edge
      cv.line(cam_img, (x3, y3), (x3, y3 + line_length), (255, 0, 255), 2)

      cv.line(cam_img, (x4, y4), (x4 , y4 - line_length), (255, 0, 255), 2)  #-- left-edge
      cv.line(cam_img, (x4, y4), (x4 , y4 + line_length), (255, 0, 255), 2)

    return cam_img


# set as seperate function call, create blank image 
def puttext_img(cam_img, frame_h, frame_w, mode_name, sub_mode, fps, zoom_factor, brightness, contrast, ipd):
  
  text_img = np.zeros((frame_h,frame_w),np.uint8)
  text_img = cv.cvtColor(text_img,cv.COLOR_GRAY2BGR)

  text_img = cv.rectangle(text_img,(int(frame_w * 0.3),int(frame_h * 0.05)),(int(frame_w * 0.45),int(frame_h * 0.0)),(0,255,255),-1)
  text_img = cv.putText(text_img,'fps:' + str(fps),(int(frame_w * 0.3),int(frame_h * 0.05)),font,1,(255,0,0),1,cv.LINE_AA)

  text_img = cv.rectangle(text_img,(int(frame_w * 0.5),int(frame_h * 0.05)),(int(frame_w * 0.65),int(frame_h * 0.0)),(255,255,0),-1)
  text_img = cv.putText(text_img,'X' + str(zoom_factor),(int(frame_w * 0.5),int(frame_h * 0.05)),font,1,(0,0,255),1,cv.LINE_AA)

  text_img = cv.rectangle(text_img,(int(frame_w * 0.65),int(frame_h * 0.05)),(int(frame_w * 0.8),int(frame_h * 0.0)),(255,0,255),-1)
  text_img = cv.putText(text_img,mode_name,(int(frame_w * 0.65),int(frame_h * 0.05)),font,1,(0,255,0),1,cv.LINE_AA)

  text_img = cv.rectangle(text_img,(int(frame_w * 0.65),int(frame_h * 0.1)),(int(frame_w * 0.8),int(frame_h * 0.05)),(255,0,255),-1)
  text_img = cv.putText(text_img,sub_mode,(int(frame_w * 0.65),int(frame_h * 0.1)),font,1,(0,255,0),1,cv.LINE_AA)

  if mode_name == 'Reading1':
    text_img = draw_border(text_img, (int(frame_w * 0.5),int(frame_h * 0.1)), (int(frame_w * 0.5),int(frame_h * 0.9)), (int(frame_w * 0.9),int(frame_h * 0.5)), (int(frame_w * 0.1),int(frame_h * 0.5)), int(frame_w * 0.05), 'edge')
  
  if mode_name == 'Reading2':
    text_img = draw_border(text_img, (int(frame_w * 0.1),int(frame_h * 0.1)), (int(frame_w * 0.1),int(frame_h * 0.9)), (int(frame_w * 0.9),int(frame_h * 0.1)), (int(frame_w * 0.9),int(frame_h * 0.9)), int(frame_w * 0.05), 'corner')
  
  if mode_name == 'Setting':
    if sub_mode == 'Brightness':
      text_img = cv.rectangle(text_img,(int(frame_w * 0.3),int(frame_h * 0.7)),(int(frame_w * 0.6),int(frame_h * 0.6)),(0,255,255),-1)
      text_img = cv.putText(text_img,'brightness:' + str(brightness),(int(frame_w * 0.3),int(frame_h * 0.7)),font,1,(255,0,0),1,cv.LINE_AA)      
    if sub_mode == 'Contrast':
      text_img = cv.rectangle(text_img,(int(frame_w * 0.3),int(frame_h * 0.7)),(int(frame_w * 0.6),int(frame_h * 0.6)),(0,255,255),-1)
      text_img = cv.putText(text_img,'Contrast:' + str(contrast),(int(frame_w * 0.3),int(frame_h * 0.7)),font,1,(0,0,255),1,cv.LINE_AA)
    if sub_mode == 'IPD':
      text_img = cv.rectangle(text_img,(int(frame_w * 0.3),int(frame_h * 0.7)),(int(frame_w * 0.6),int(frame_h * 0.6)),(0,255,255),-1)
      text_img = cv.putText(text_img,'IPD:' + str(ipd),(int(frame_w * 0.3),int(frame_h * 0.7)),font,1,(0,0,255),1,cv.LINE_AA)
  cam_img = cv.addWeighted(cam_img, 0.8, text_img, 1.5 ,1)
  return  cam_img
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
# Parameters
#------------------------------------------------------

frame_count = 0
# Settings defaults
contrast = 50
brightness = 50
ipd = 0
zoom_factor = 1.0
overlay_scale = 3.2

# mode and sub mode flags
mode_select = 0
scenic_select = 0
reading_select = 0
setting_select = 0
mode_name = 'Norm'

# distortion parameters
mtx = np.array([[ 1.70000000e+06, 0.00000000e+00, 0.00000000e+00 ],[ 0.00000000e+00, 1.70000000e+06, 0.00000000e+00 ],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 1.70000000e+06 , 1.00000000e+05 , 0.00000000e+00, 0.00000000e+00, 1.00000000e+04 ]])
pos = 0

if __name__ == '__main__':

  video_stream_widget = VideoStreamWidget()

  mp.set_start_method('spawn')
  pool = mp.Pool(2)
  
  box_line = [[],[]]
  reading_first = 1

  while True:

      #normal sequential capture method
      start = time.time()
      img = video_stream_widget.frame

      #-----------------------------------------Pre-Processing---------------------------------------
      
      # Camera image position and orientation correction 
      #******************This is for PC camera, Need to change for Jetson****************************
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
  
      # move no glr functions here   
      camera_A = remove_image_glare(camera_A)
      camera_B = remove_image_glare(camera_B)

      # Brightness and contrast functions
      camera_A = apply_brightness_contrast(camera_A,brightness,contrast)
      camera_B = apply_brightness_contrast(camera_B,brightness,contrast)

      # create function to introduce IPD and vary values
      img_pos_A = 'left'
      img_pos_B = 'right'

      # introduce ipd function for Jetson camera  
      #camera_A = ipd_img(camera_A, img_pos_A, frame_h, frame_w, frame_h, ipd)
      #camera_B = ipd_img(camera_B, img_pos_B, frame_h, frame_w, frame_h, ipd)
      
      #-----------------------------------------Modes------------------------------------------------

      # Scenic mode
      if mode_select == 0:
        mode_name = 'Scenic'

        if scenic_select == 1:# Edge mode
          sub_mode = 'Edge'
          camera_A = edge_image(camera_A)
          camera_B = edge_image(camera_B)

        elif scenic_select == 2:#bubble mode
          sub_mode = 'bubble'
          img_pair = [camera_A, camera_B]
          bubble_pair = pool.map(bubble, img_pair)
          camera_A = bubble_pair[0]
          camera_B = bubble_pair[1]

        else:
          sub_mode = 'Normal' 
      
        camera_A = zoom_image(camera_A,zoom_factor)
        camera_B = zoom_image(camera_B,zoom_factor)

      # Reading mode - 1
      elif mode_select == 1:
        mode_name = 'Reading1'
        img_pair = [camera_A, camera_B]
        box_line = pool.map(ocr_img, img_pair)

        if reading_select == 1:#BW mode
          sub_mode = 'BW'
          camera_A = bw_img(camera_A)
          camera_B = bw_img(camera_B)

        elif reading_select == 2:#BWHC mode
          sub_mode = 'BWHC'
          camera_A = bwhc_img(camera_A)
          camera_B = bwhc_img(camera_B)

        elif reading_select == 3:#WBHC mode
          sub_mode = 'WBHC'
          camera_A = wbhc_img(camera_A)
          camera_B = wbhc_img(camera_B)

        else:
          sub_mode = 'Normal' 

        camera_A = draw_boxes(camera_A,box_line[0])
        camera_B = draw_boxes(camera_B,box_line[1])

        camera_A_src = draw_boxes(camera_A_src,box_line[0])
        camera_B_src = draw_boxes(camera_B_src,box_line[1])
 
        camera_A = zoom_image(camera_A,zoom_factor)
        camera_B = zoom_image(camera_B,zoom_factor)

        camera_A = apply_img_overlay(camera_A, camera_A_src , overlay_scale, pos)
        camera_B = apply_img_overlay(camera_B, camera_B_src , overlay_scale, pos)
      
      # Reading mode - 2
      elif mode_select == 2:
        mode_name = 'Reading2'
        img_pair = [camera_A, camera_B]
        box_line = pool.map(ocr_img, img_pair)
        
        if reading_select == 1:#BW mode
          sub_mode = 'BW'
          camera_A = bw_img(camera_A)
          camera_B = bw_img(camera_B)

        elif reading_select == 2:#BWHC mode
          sub_mode = 'BWHC'
          camera_A = bwhc_img(camera_A)
          camera_B = bwhc_img(camera_B)

        elif reading_select == 3:#WBHC mode
          sub_mode = 'WBHC'
          camera_A = wbhc_img(camera_A)
          camera_B = wbhc_img(camera_B)

        else:
          sub_mode = 'Normal' 

        camera_A = draw_boxes(camera_A,box_line[0])
        camera_B = draw_boxes(camera_B,box_line[1])

        camera_A = apply_zoom_overlay(camera_A, zoom_factor)
        camera_B = apply_zoom_overlay(camera_B, zoom_factor)
      
      # Setting mode
      elif mode_select == 3:
        mode_name = 'Setting'

        if setting_select == 0:#Brightness mode
          sub_mode = 'Brightness'
          
        elif setting_select == 1:#Contrast mode
          sub_mode = 'Contrast'

        else: #IPD mode
          sub_mode = 'IPD'
      
      else:
        mode_select = 0

      #-----------------------------------------Post-Processing--------------------------------------
      
      # fps calculation
      end = time.time()
      time_elapsed = end - start
      fps = 1 // time_elapsed
      
      # Display status texts    
      camera_A = puttext_img(camera_A, frame_h, frame_w, mode_name, sub_mode, fps, zoom_factor, brightness, contrast, ipd)
      camera_B = puttext_img(camera_B, frame_h, frame_w, mode_name, sub_mode, fps, zoom_factor, brightness, contrast, ipd)
      
      # Apply radial distortion on image
      camera_A = apply_radial_distortion(camera_A, mtx, dist)
      camera_B = apply_radial_distortion(camera_B, mtx, dist)
      
      # Concatenate left and right inages as single image output
      if frame_w > frame_h:
        new_frame = np.concatenate((camera_A,camera_B),axis=1)
      else:
        new_frame = np.concatenate((camera_A,camera_B),axis=0)
      
      #-------------------------------------------User Input-----------------------------------------
      #**********need to optimize***********
      key = cv.waitKey(1)
      if key==ord('x'):#mode change:--> Scenic, Reading, Setting
        if mode_select == 3:
          mode_select = 0
        else:  
          #if mode_select == 1:
          #  reading_first = 1
          #  while not q.empty():
          #   print(q.get())
          mode_select +=1 

      if key==ord('c'):
        if mode_select == 0:#scenic mode change:--> Normal, Edge, Bubble
          setting_select = 0
          if scenic_select == 3: 
            scenic_select = 0
          else:  
            scenic_select +=1 
        if mode_select == 1:#reading mode 1 change:--> Normal, BW, BWHC, WBHC
          scenic_select = 0
          if reading_select == 4: 
            reading_select = 0
          else:  
            reading_select +=1 
        if mode_select == 2:#reading mode 2 change:--> Normal, BW, BWHC, WBHC
          #reading_select = 0
          if reading_select == 4: 
            reading_select = 0
          else:  
            reading_select +=1 
        if mode_select == 3:#setting mode change:--> Brightness, Contrast, ipd
          reading_select = 0
          if setting_select == 2: 
            setting_select = 0
          else:  
            setting_select +=1 
      
      if key==ord('m'):
        if mode_select == 3:
          if setting_select == 0:
            brightness+=1
          if setting_select == 1:
            contrast+=1
          if setting_select == 2:
            ipd+=1
        else:#zoom in:
          if zoom_factor >= 8.0:
            zoom_factor = 8.0
          else:  
            zoom_factor = zoom_factor + 0.2

      if key==ord('n'):
        if mode_select == 3:
          if setting_select == 0:
            brightness-=1
          if setting_select == 1:
            contrast-=1
          if setting_select == 2:
            ipd-=1
        else:#zoom out
          if zoom_factor <= 1.0:
            zoom_factor = 1.0
          else:  
            zoom_factor = zoom_factor - 0.2
      
      if key == ord('q'):
        pool.close()
        pool.join()
        #q.join()
        time.sleep(1)
      video_stream_widget.show_frame(new_frame,key)
        

