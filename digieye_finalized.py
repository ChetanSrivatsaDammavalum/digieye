#!/usr/bin/env python3
# Program to test gstreamer pipeline on simple opencv video capture with no threading
import sys
import os
import cv2 as cv
import time
import numpy as np
import multiprocessing as mp
import math
from threading import Thread
import RPi.GPIO as GPIO

but_pin_13 = 13  # Board pin 13
but_pin_15 = 15  # Board pin 15
but_pin_19 = 19  # Board pin 19
but_pin_21 = 21  # Board pin 21
but_pin_23 = 23  # Board pin 23

GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme

flip=1
dispW=1080  
dispH=2160 
#camSet='nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
camSet='nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3264, height=1848, format=NV12, framerate=28/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'


font = cv.FONT_HERSHEY_SIMPLEX

#------------------------------------------------------
# Threading class for video input and GPIO input
#------------------------------------------------------
#---------------------------------Threading GPIO feed--------------------------------

class Button(object):
#class Button(threading.Thread):
  def __init__(self, channel_mode, channel_submode, channel_plus, channel_minus, channel_shutdown):
    self.channel_mode = channel_mode
    self.channel_submode = channel_submode
    self.channel_plus = channel_plus
    self.channel_minus = channel_minus
    self.channel_shutdown = channel_shutdown

    #self._pressed = False
    self.pressed_mode = False
    self.pressed_submode = False
    self.pressed_plus = False
    self.pressed_minus = False
    self.pressed_shutdown = False

    #GPIO.setup(self.channel, GPIO.IN)
    GPIO.setup(self.channel_mode, GPIO.IN)
    GPIO.setup(self.channel_submode, GPIO.IN)
    GPIO.setup(self.channel_plus, GPIO.IN)
    GPIO.setup(self.channel_minus, GPIO.IN)
    GPIO.setup(self.channel_shutdown, GPIO.IN)

    # Start the thread to read GPIO input from the user
    self.thread = Thread(target=self.run, args=())
    self.thread.daemon = True
    self.thread.start()

  def run(self):
    while 1:
      #current = GPIO.input(self.channel)
      current_mode = GPIO.input(self.channel_mode)
      current_submode = GPIO.input(self.channel_submode)
      current_plus = GPIO.input(self.channel_plus)
      current_minus = GPIO.input(self.channel_minus)
      current_shutdown = GPIO.input(self.channel_shutdown)
      time.sleep(0.02)

      if current_mode == 0 :
        self.pressed_mode = True
      else:
        self.pressed_mode = False

      if current_submode == 0 :
        self.pressed_submode = True
      else:
        self.pressed_submode = False
     
      if current_plus == 0 :
        self.pressed_plus = True
      else:
        self.pressed_plus = False
     
      if current_minus == 0 :
        self.pressed_minus = True
      else:
        self.pressed_minus = False

      if current_shutdown == 1 :
        self.pressed_shutdown = True
      else:
        self.pressed_shutdown = False
      
  def gpio_close(self,key):
    if key == ord('q') :
      GPIO.cleanup()

#---------------------------------Threading Video feed--------------------------------

class VideoStreamWidget(object):
  def __init__(self, src):
    self.src = src
    self.capture = cv.VideoCapture(src, cv.CAP_GSTREAMER)
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
    winname = "digieye"
    cv.namedWindow(winname)        # Create a named window
    cv.moveWindow(winname, 0,0)    # Move it to (0,0)
    cv.imshow(winname, img)
    #cv.imshow('frame', img)
    if key == ord('q') :
      self.capture.release()
      GPIO.cleanup()
      time.sleep(1)
      cv.destroyAllWindows()
      exit(1)

    if key == 1 : 
      self.capture.release()
      GPIO.cleanup()
      time.sleep(1)
      cv.destroyAllWindows()
      os.system('shutdown -h now')
      exit(1)

#------------------------------------------------------
# Image magnification
#------------------------------------------------------
#<--------------------------------------------------------------------cuda added
def zoom_image(cam_img, zoom_factor):
  h, w = cam_img.shape[:2]
  crop_w = w//zoom_factor
  crop_h = h//zoom_factor
  crop_img = cam_img[int(w // 2 - crop_w // 2):int( w // 2 + crop_w // 2), int(h // 2 - crop_h // 2):int(h // 2 + crop_h // 2)]
  return cv.resize(crop_img,(w,h),interpolation = cv.INTER_LINEAR )
  #return cv.resize(crop_img,(w,h),interpolation = cv.INTER_LANCZOS4 )
  # return cv2.cuda.resize(crop_img,(w,h),interpolation = cv.INTER_LANCZOS4)

#------------------------------------------------------
# Image de-glareing
#------------------------------------------------------
#************************************************************Need to improve- try cuda
def remove_image_glare(cam_img):
  #Function to remove glare spots from image

  cam_img_hsv = cv.cvtColor(cam_img, cv.COLOR_BGR2HSV)
  h, s, v = cv.split(cam_img_hsv)
  ret,v = cv.threshold(v,180,255,cv.THRESH_TRUNC)
  #v = cv.equalizeHist(v)
  img_glr = cv.merge((h,s,v))
  img_glr = cv.cvtColor(img_glr, cv.COLOR_HSV2BGR)
  return img_glr

def run_histogram_equalization(cam_img):
  # convert from RGB color-space to YCrCb
  ycrcb_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2YCrCb)

  # equalize the histogram of the Y channel
  ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])

  # convert back to RGB color-space from YCrCb
  # cam_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
  return cv2.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)

#------------------------------------------------------
# Image Brightness and Contrast control
#------------------------------------------------------

def apply_brightness_contrast(cam_img, brightness, contrast, gamma_table):
  #Function to change the brightness and/or contrast of input image
  #---------Need to set limits and increamentsteps------- 
  #brightness = map(brightness, 0, 100, -255, 255)
  #contrast = map(contrast, 0, 100, -127, 127)
  brightness = (brightness - 50) * 2
  contrast = (contrast*1.31) - 1.31 

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

  #+*******************************************************************************introduce gamma correction - check luminesence of image and use appropiate gamma  
  #return img_bnc
  return adjust_gamma(img_bnc, gamma_table[4])

#*****************************************************************************Need to improve with cuda

def gamma_table(gamma):
  gamma_table = []
  for k in range(len(gamma)):
    invGamma = 1.0 / gamma[k]
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_table.append(table)
  return gamma_table

def adjust_gamma(cam_img, table):
	return cv.LUT(cam_img, table)

#******************************************************************************Need to improve with cuda
def noise_reduction(cam_img, diameter=3, sigmaColor=21, sigmaSpace=5):
  return cv.bilateralFilter(cam_img, diameter, sigmaColor, sigmaSpace) 

#------------------------------------------------------
# Image edge highlight
#------------------------------------------------------

# def edge_image(cam_img):
def edge_image(cam_img):
  edge_img = cv.cvtColor(cam_img, cv.COLOR_BGR2GRAY)
  edge_img = apply_brightness_contrast(edge_img,0,32,gamma_table)
  edge_img = cv.GaussianBlur(edge_img,(3,3),0)
  edge_img = cv.Canny(edge_img,80,200)
  edge_img = cv.cvtColor(edge_img, cv.COLOR_GRAY2BGR)
  return cv.addWeighted(cam_img, 1, edge_img, 0.4, 1.2 )

#------------------------------------------------------
# Bubble magnification
#------------------------------------------------------

#****************************************************************************CHECK--break function--one to generate map, one to remap
def bubble_map(cam_img, radius = 0.8, scale =0.6, amount =0.8):#for sin fn
#def bubble(cam_img, radius = 0.4, scale =0.4, amount =0.8):#for sin fn
  # grab the dimensions of the image
  h, w = cam_img.shape[:2]
  center_y = h//2
  center_x = w//2

  radius = int(radius * h)

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
                  factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
                  #factor = math.pow( math.sqrt(distance) / radius , amount)
              flex_x[y, x] = factor * delta_x / scale + center_x
              flex_y[y, x] = factor * delta_y / scale + center_y
  
  # change maps to CUDA maps <----------------
  #flex_x = cv.cuda_GpuMat(flex_x )
  #flex_y = cv.cuda_GpuMat(flex_y)

  return flex_x, flex_y

def bubble(cam_img, flex_x, flex_y):#for sin fn

  # grab the dimensions of the image
  h, w = cam_img.shape[:2]

  cam_img = cv.line(cam_img, (int(0.5 * w), int(0.3*h)), (int(0.5 * w), int(0.4*h)), (255, 0, 0), 2)
  cam_img = cv.line(cam_img, (int(0.5 * w), int(0.6*h)), (int(0.5 * w), int(0.7*h)), (255, 0, 0), 2)
  cam_img = cv.line(cam_img, (int(0.5 * w), int(0.3*h)), (int(0.5 * w), int(0.4*h)), (255, 255, 0), 1)
  cam_img = cv.line(cam_img, (int(0.5 * w), int(0.6*h)), (int(0.5 * w), int(0.7*h)), (255, 255, 0), 1)

  cam_img = cv.line(cam_img, (int(0.47 * w), int(0.3*h)), (int(0.47 * w), int(0.7*h)), (255, 0, 0), 2)
  cam_img = cv.line(cam_img, (int(0.53 * h), int(0.3*h)), (int(0.53 * w), int(0.7*h)), (255, 0, 0), 2)
  cam_img = cv.line(cam_img, (int(0.47 * w), int(0.3*h)), (int(0.47 * w), int(0.7*h)), (255, 255, 0), 1)
  cam_img = cv.line(cam_img, (int(0.53 * h), int(0.3*h)), (int(0.53 * w), int(0.7*h)), (255, 255, 0), 1)

  #return cv.remap(cam_img, flex_x, flex_y, cv.INTER_LANCZOS4) 
  return cv.remap(cam_img, flex_x, flex_y, cv.INTER_LINEAR) 
  #return cv.cuda.remap(cam_img, flex_x, flex_y, cv.INTER_LINEAR)
#------------------------------------------------------
# OCR
#------------------------------------------------------

#def draw_boxes(image, bounds, color='blue', width=1):
#    draw = ImageDraw.Draw(Image.fromarray(image))
#      for bound in bounds:
#        p0, p1, p2, p3 = bound[0]
#        print(p0, p1 ,p2 ,p3)
#        p21, p22 = np.array(p2)
#        p31, p32 = np.array(p3)
#        cv.line(image,(int(p21),int(p22)),(int(p31),int(p32)),(255,0,0),2)
#  return image 

# *************888change with queues***********
#def ocr_img(cam_img):
#  bounds = reader.readtext(cam_img)
  #q.put('Process Done')
#  return bounds
  
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
  cam_img = apply_brightness_contrast(cam_img,10,32,gamma_table)
  ret,v = cv.threshold(cam_img,155,255,cv.THRESH_BINARY)
  return cv.cvtColor(v, cv.COLOR_GRAY2BGR)

#------------------------------------------------------
# Image white-black high contrast
#------------------------------------------------------

def wbhc_img(cam_img):
  cam_img = cv.cvtColor(cam_img, cv.COLOR_BGR2GRAY)
  cam_img = apply_brightness_contrast(cam_img,10,32,gamma_table)
  ret,v = cv.threshold(cam_img,155,255,cv.THRESH_BINARY)
  v = cv.cvtColor(v, cv.COLOR_GRAY2BGR)
  return 255 - v  

#------------------------------------------------------
# Image overlaying
#------------------------------------------------------

#<--------------------------------------------------------------------cuda added
def apply_img_overlay(cam_img, cam_img_src, scale_factor, pos):
  #Function to overlay windowed mini-image on input image

  w ,h = cam_img.shape[:2]
  w_src, h_src = cam_img_src.shape[:2]
  
  overlay_img = cv.resize(cam_img_src,(int(w_src//scale_factor) , int(h_src//scale_factor)),interpolation = cv.INTER_LINEAR )
  #overlay_img = cv.cuda.resize(cam_img_src,(int(w_src//scale_factor) , int(h_src//scale_factor)),interpolation = cv.INTER_LINEAR )
  
  overlay_img = img_border(overlay_img)
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

#**********************************************************************************CHECK
def apply_zoom_overlay(cam_img, overly_zoom_factor):
  #h = cam_img.shape[0]
  w = cam_img.shape[0]
  #w = cam_img.shape[1]
  h = cam_img.shape[1]
  #overlay_left = cam_img[  w -int(w * 0.3)  : w , int(h * 0.3) : int(h * 0.7) ]
  overlay_left = cam_img[  0 : int(w * 0.3) , int(h * 0.4) : int(h * 0.6) ]
  overlay_left = zoom_image(overlay_left, overly_zoom_factor) 
  overlay_left = img_border(overlay_left)
  overlay_img = zoom_image(cam_img, overly_zoom_factor)
  overlay_img = overlay_img[ 0  : w , int(h * 0.4) : int(h * 0.6) ]
  overlay_img = img_border(overlay_img)

  cam_img[ 0  : w ,int(h * 0.4) : int(h * 0.6)] = overlay_img
  #cam_img[ w -int(w * 0.3)  : w ,int(h * 0.4) : int(h * 0.6)] = overlay_left
  cam_img[ 0 : int(w * 0.3) , int(h * 0.4) : int(h * 0.6)] = overlay_left
  return cam_img


#### small helper fn to draw border around image - need to modify
def img_border(cam_img):
  row, col = cam_img.shape[:2]
  cam_img = cv.rectangle(cam_img,(0,0),(col,row),(255,255,255),3) # really thick white rectangle
  
  return cam_img

#+++++++++++++++++++++++++*********************************************************************************
#**********************************************************************************************************
def process_main(cam_img,cam_img_src, mode_select,scenic_select,reading_select,setting_select,zoom_factor,overlay_scale, pos, flex_x, flex_y ):
    
  #cam_img = noise_reduction(cam_img)

  # Scenic mode
  if mode_select == 0:
    mode_name = 'Scenic'

    if scenic_select == 1:# No glare mode
      sub_mode = 'Noglare' 
      cam_img = remove_image_glare(cam_img)
    
    if scenic_select == 2:# Edge mode
      sub_mode = 'Edge'
      cam_img = edge_image(cam_img)
      
    else:
      sub_mode = 'Normal' 

    cam_img = zoom_image(cam_img,zoom_factor)
  
  # Reading mode - 1 -> Image with small window overlay
  elif mode_select == 1:
    mode_name = 'Reading1'

    if reading_select == 1:#BW mode
      sub_mode = 'BW'
      cam_img = bw_img(cam_img)

    elif reading_select == 2:#BWHC mode
      sub_mode = 'BWHC'
      cam_img = bwhc_img(cam_img)

    elif reading_select == 3:#WBHC mode
      sub_mode = 'WBHC'
      cam_img = wbhc_img(cam_img)

    else:
      sub_mode = 'Normal'

    cam_img = zoom_image(cam_img,zoom_factor)
    cam_img = apply_img_overlay(cam_img, cam_img_src , overlay_scale, pos)
  
  # Reading mode - 2 -> Image with center window overlay
  elif mode_select == 2:
    mode_name = 'Reading2'

    if reading_select == 1:#BW mode
      sub_mode = 'BW'
      cam_img = bw_img(cam_img)

    elif reading_select == 2:#BWHC mode
      sub_mode = 'BWHC'
      cam_img = bwhc_img(cam_img)

    elif reading_select == 3:#WBHC mode
      sub_mode = 'WBHC'
      cam_img = wbhc_img(cam_img)

    else:
      sub_mode = 'Normal'
    
    cam_img = apply_zoom_overlay(cam_img, overly_zoom_factor)
    
  # Reading mode - 3 -> Image with bubble magnification
  elif mode_select == 3:
    mode_name = 'Reading3'

    if reading_select == 1:#BW mode
      sub_mode = 'BW'
      cam_img = bw_img(cam_img)
    

    elif reading_select == 2:#BWHC mode
      sub_mode = 'BWHC'
      cam_img = bwhc_img(cam_img)

    elif reading_select == 3:#WBHC mode
      sub_mode = 'WBHC'
      cam_img = wbhc_img(cam_img)

    else:
      sub_mode = 'Normal' 
    
    cam_img = bubble(cam_img, flex_x, flex_y)

  # Setting mode
  elif mode_select == 4  :
    mode_name = 'Setting'

    if setting_select == 0:#Brightness mode
      sub_mode = 'Brightness'
      
    elif setting_select == 1:#Contrast mode
      sub_mode = 'Contrast'

    elif setting_select == 2: #IPD mode
      sub_mode = 'IPD'

    elif setting_select == 3: #Display mode
      sub_mode = 'Display mode'
    
    else: #Primary eye
      sub_mode = 'Primary eye'
      
  else:
    mode_select = 0
  
  return cam_img, mode_name, sub_mode

#******************************************************************************************************+***
#**********************************************************************************************************

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
  zoom_factor = round(zoom_factor,1)

  text_img = np.zeros((frame_h,frame_w),np.uint8)
  text_img = cv.cvtColor(text_img,cv.COLOR_GRAY2BGR)

  cam_img = cv.rectangle(cam_img,(int(frame_w * 0.94),int(frame_h * 0.5)),(int(frame_w * 0.98),int(frame_h * 0.65)),(0,0,0),-1)
  text_img = cv.putText(text_img,'fps:' + str(fps),(int(frame_w * 0.5),int(frame_h * 0.05)),font,1,(255,255,0),3,cv.LINE_AA)
  text_img = cv.putText(text_img,'fps:' + str(fps),(int(frame_w * 0.5),int(frame_h * 0.05)),font,1,(255,0,0),2,cv.LINE_AA)

  cam_img = cv.rectangle(cam_img,(int(frame_w * 0.94),int(frame_h * 0.68)),(int(frame_w * 0.98),int(frame_h * 0.76)),(0,0,0),-1)
  text_img = cv.putText(text_img,'X' + str(zoom_factor),(int(frame_w * 0.68),int(frame_h * 0.05)),font,1,(0,255,255),3,cv.LINE_AA)
  text_img = cv.putText(text_img,'X' + str(zoom_factor),(int(frame_w * 0.68),int(frame_h * 0.05)),font,1,(0,0,255),2,cv.LINE_AA)
  
  cam_img = cv.rectangle(cam_img,(int(frame_w * 0.94),int(frame_h * 0.78)),(int(frame_w * 0.98),int(frame_h * 0.96)),(0,0,0),-1)
  text_img = cv.putText(text_img,mode_name,(int(frame_w * 0.78),int(frame_h * 0.05)),font,1,(255,255,255),3,cv.LINE_AA)
  text_img = cv.putText(text_img,mode_name,(int(frame_w * 0.78),int(frame_h * 0.05)),font,1,(0,255,0),2,cv.LINE_AA)

  cam_img = cv.rectangle(cam_img,(int(frame_w * 0.89),int(frame_h * 0.78)),(int(frame_w * 0.93),int(frame_h * 0.96)),(0,0,0),-1)
  text_img = cv.putText(text_img,sub_mode,(int(frame_w * 0.78),int(frame_h * 0.1)),font,1,(180,255,0),3,cv.LINE_AA)
  text_img = cv.putText(text_img,sub_mode,(int(frame_w * 0.78),int(frame_h * 0.1)),font,1,(0,255,180),2,cv.LINE_AA)

  if mode_name == 'Reading1':
    text_img = draw_border(text_img, (int(frame_w * 0.5),int(frame_h * 0.1)), (int(frame_w * 0.5),int(frame_h * 0.9)), (int(frame_w * 0.9),int(frame_h * 0.5)), (int(frame_w * 0.1),int(frame_h * 0.5)), int(frame_w * 0.05), 'edge')
  
  if mode_name == 'Reading2':
    text_img = draw_border(text_img, (int(frame_w * 0.1),int(frame_h * 0.1)), (int(frame_w * 0.1),int(frame_h * 0.9)), (int(frame_w * 0.9),int(frame_h * 0.1)), (int(frame_w * 0.9),int(frame_h * 0.9)), int(frame_w * 0.05), 'corner')
  
  if mode_name == 'Reading3':
    text_img = draw_border(text_img, (int(frame_w * 0.1),int(frame_h * 0.1)), (int(frame_w * 0.1),int(frame_h * 0.9)), (int(frame_w * 0.9),int(frame_h * 0.1)), (int(frame_w * 0.9),int(frame_h * 0.9)), int(frame_w * 0.05), 'corner')
  
  if mode_name == 'Setting':
    if sub_mode == 'Brightness':
      cam_img = cv.rectangle(cam_img,(int(frame_w * 0.51),int(frame_h * 0.40)),(int(frame_w * 0.55),int(frame_h * 0.66)),(0,0,0),-1)
      text_img = cv.putText(text_img,'Brightness:' + str(brightness),(int(frame_w * 0.4),int(frame_h * 0.48)),font,1,(0,255,255),3,cv.LINE_AA)      
      text_img = cv.putText(text_img,'Brightness:' + str(brightness),(int(frame_w * 0.4),int(frame_h * 0.48)),font,1,(255,0,0),2,cv.LINE_AA)      
    if sub_mode == 'Contrast':
      cam_img = cv.rectangle(cam_img,(int(frame_w * 0.51),int(frame_h * 0.40)),(int(frame_w * 0.55),int(frame_h * 0.62)),(0,0,0),-1)
      text_img = cv.putText(text_img,'Contrast:' + str(contrast),(int(frame_w * 0.4),int(frame_h * 0.48)),font,1,(255,255,0),3,cv.LINE_AA)
      text_img = cv.putText(text_img,'Contrast:' + str(contrast),(int(frame_w * 0.4),int(frame_h * 0.48)),font,1,(0,0,255),2,cv.LINE_AA)
    if sub_mode == 'IPD':
      cam_img = cv.rectangle(cam_img,(int(frame_w * 0.51),int(frame_h * 0.48)),(int(frame_w * 0.55),int(frame_h * 0.59)),(0,0,0),-1)
      text_img = cv.putText(text_img,'IPD:' + str(ipd),(int(frame_w * 0.48),int(frame_h * 0.48)),font,1,(255,255,0),3,cv.LINE_AA)
      text_img = cv.putText(text_img,'IPD:' + str(ipd),(int(frame_w * 0.48),int(frame_h * 0.48)),font,1,(0,0,255),2,cv.LINE_AA)
    if sub_mode == 'Display mode':
      cam_img = cv.rectangle(cam_img,(int(frame_w * 0.51),int(frame_h * 0.40)),(int(frame_w * 0.55),int(frame_h * 0.82)),(0,0,0),-1)
      text_img = cv.putText(text_img,'Display mode: ' + display_mode_name,(int(frame_w * 0.40),int(frame_h * 0.48)),font,1,(255,255,0),3,cv.LINE_AA)
      text_img = cv.putText(text_img,'Display mode: ' + display_mode_name,(int(frame_w * 0.40),int(frame_h * 0.48)),font,1,(0,0,255),2,cv.LINE_AA)
    if sub_mode == 'Primary eye':
      cam_img = cv.rectangle(cam_img,(int(frame_w * 0.51),int(frame_h * 0.40)),(int(frame_w * 0.55),int(frame_h * 0.70)),(0,0,0),-1)
      text_img = cv.putText(text_img,'Primary eye: ' + primary_eye,(int(frame_w * 0.40),int(frame_h * 0.48)),font,1,(255,255,0),3,cv.LINE_AA)
      text_img = cv.putText(text_img,'Primary eye: ' + primary_eye,(int(frame_w * 0.40),int(frame_h * 0.48)),font,1,(0,0,255),2,cv.LINE_AA)
  text_img = cv.rotate(text_img, cv.ROTATE_90_CLOCKWISE)
  cam_img = cv.addWeighted(cam_img, 1, text_img, 3 ,0)
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
contrast = 20
brightness = 50
ipd = 5
zoom_factor = 1.0
overly_zoom_factor = 1.0
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
pos = 3

# bubble mapping parameters
flex_x = []
flex_y = []

# gamma adjustment parameters
gamma = [0.8, 0.9, 1.0, 1.1, 1.2]
ganna_table = []

eyebox_w = 900 
eyebox_h = 900 

display_mode = 0
primary_eye = 'right'

first_run = True

if __name__ == '__main__':

  video_stream_widget = VideoStreamWidget(camSet)

  #mp.set_start_method('spawn')
  #pool = mp.Pool(2)
  
  button = Button(but_pin_19,but_pin_15,but_pin_13,but_pin_21,but_pin_23)
  #button_plus = Button(but_pin_13)
  #button_minus = Button(but_pin_21) 
  #button_mode = Button(but_pin_19) 
  #button_submode = Button(but_pin_15) 
  #button_shutdown = Button(but_pin_23) 
  gamma_table = gamma_table(gamma)

  while True:

    start = time.time()
    vid_frame = video_stream_widget.frame
    frame_h, frame_w = vid_frame.shape[:2]

    #-----------------------------------------Pre-Processing and Processing------------------------------------
    split_len = frame_h//2
    #extract src image for overlay
    # calculate remap values for bubble distortion 
    
    #*****++**********+++++++++++++++************change first********************************
    
    #*********************+++++++++++++++++**+++++change firt+++++++++++++*+++++++++++++++++++++
    
    if display_mode  == 1:
      display_mode_name = 'binocular'    
  
      # right image
      camera_A = vid_frame[ : split_len , : ]#right image
      camera_A = cv.flip(camera_A, -1)# flips image about both vertical and horizontal axis
      camera_A_h, camera_A_w = camera_A.shape[:2]
      # left image
      camera_B = vid_frame[ split_len : , : ]#left image
      camera_B = cv.flip(camera_B, -1)# flips image about both vertical and horizontal axis
      camera_B_h, camera_B_w = camera_B.shape[:2]
      
      # call remap function for bubble
      if first_run:
        flex_x , flex_y = bubble_map(camera_A)
        first_run = False

      # adjust gamma
      #camera_A = adjust_gamma(camera_A)
      #camera_B = adjust_gamma(camera_B)

      # Brightness and contrast functions
      camera_A = apply_brightness_contrast(camera_A,brightness,contrast, gamma_table)
      camera_B = apply_brightness_contrast(camera_B,brightness,contrast, gamma_table)

      camera_A_src = camera_A
      camera_B_src = camera_B

      camera_A, mode_name, sub_mode = process_main(camera_A,camera_A_src,mode_select,scenic_select,reading_select,setting_select,zoom_factor,overlay_scale, pos, flex_x , flex_y)
      camera_B, mode_name, sub_mode = process_main(camera_B,camera_B_src,mode_select,scenic_select,reading_select,setting_select,zoom_factor,overlay_scale, pos, flex_x , flex_y)

    else:
      display_mode_name = 'biocular' 
      if primary_eye == 'right':
        # right image
        camera_A = vid_frame[ : split_len , : ]#right image
        camera_A = cv.flip(camera_A, -1)# flips image about both vertical and horizontal axis
        camera_A_h, camera_A_w = camera_A.shape[:2]
        
        # call remap function for bubble
        if first_run:
          flex_x , flex_y = bubble_map(camera_A)
          first_run = False
        
        # noise reduction, gamma correction, brightness and contrast adjustment
        #camera_A = noise_reduction(camera_A)
        #camera_A = adjust_gamma(camera_A)
        camera_A = apply_brightness_contrast(camera_A,brightness,contrast, gamma_table)

        camera_A_src = camera_A
        camera_A, mode_name, sub_mode = process_main(camera_A,camera_A_src,mode_select,scenic_select,reading_select,setting_select,zoom_factor,overlay_scale, pos, flex_x , flex_y)
        
        # left image
        camera_B = camera_A
        camera_B_src = camera_B
        camera_B_h, camera_B_w = camera_B.shape[:2]

      else:
        # left image
        camera_B = vid_frame[ split_len : , : ]#left image
        camera_B = cv.flip(camera_B, -1)# flips image about both vertical and horizontal axis
        camera_B_h, camera_B_w = camera_B.shape[:2]

        # call remap function for bubble
        if first_run:
          flex_x , flex_y = bubble_map(camera_B)
          first_run = False
        
        #camera_B = adjust_gamma(camera_B)
        camera_B = apply_brightness_contrast(camera_B,brightness,contrast, gamma_table)
        
        camera_B_src = camera_B
        camera_B, mode_name, sub_mode = process_main(camera_B,camera_B_src,mode_select,scenic_select,reading_select,setting_select,zoom_factor,overlay_scale, pos, flex_x , flex_y)
        
        # right image
        camera_A = camera_B
        camera_A_src = camera_A
        camera_A_h, camera_A_w = camera_A.shape[:2]

    # introduce ipd to right image
    if ( eyebox_h//2 + ipd ) <= camera_A_h//2 :
      midw_A = (int(camera_A_w//2), int(camera_A_h//2 + ipd ))
    else:
      midw_A = (int(camera_A_w//2), int(camera_A_h//2 + ( camera_A_w//2 - eyebox_w//2 )))
    camera_A = camera_A[ midw_A[1] - int(eyebox_h//2) : midw_A[1] + int(eyebox_h//2) , midw_A[0] - int(eyebox_w//2) : midw_A[0] + int(eyebox_w//2)]
    
    
    # introduce ipd to left image
    if ( eyebox_h//2 + ipd ) <= camera_B_w//2 :
      midw_B = (int(camera_B_w//2), int(camera_B_h//2 - ipd ))
    else:
      midw_B = (int(camera_B_w//2), int(camera_B_h//2 - ( camera_B_w//2 - eyebox_w//2 )))
    camera_B = camera_B[ midw_B[1] - int(eyebox_h//2) : midw_B[1] + int(eyebox_h//2) , midw_B[0] - int(eyebox_w//2) : midw_B[0] + int(eyebox_w//2)]
    

    # remove glare from image
    #camera_A = remove_image_glare(camera_A)
    #camera_B = remove_image_glare(camera_B)

#-----------------------------------------Modes------------------------------------------------
    #data_pair = [(camera_A,camera_A_src,mode_select,scenic_select,reading_select,setting_select,zoom_factor,overlay_scale, pos), (camera_B,camera_B_src,mode_select,scenic_select,reading_select,setting_select,zoom_factor,overlay_scale, pos)]
    #process_op = pool.starmap(process_main, data_pair)
    #process_op_A = process_op[0] # process result for camera_A
    #process_op_B = process_op[1] # process result for camera_B
    
    #camera_A = process_op_A[0] 
    #camera_B = process_op_B[0]
    #mode_name = process_op_A[1]
    #sub_mode = process_op_A[2]
    
#-----------------------------------------Post-Processing--------------------------------------    
        
    # fps calculation
    end = time.time()
    time_elapsed = end - start
    fps = 1 // time_elapsed

    # Display status texts    
    camera_A = img_border(camera_A)
    camera_B = img_border(camera_B)
    frame_A_h, frame_A_w = camera_A.shape[:2]
    frame_B_h, frame_B_w = camera_B.shape[:2]
    camera_A = puttext_img(camera_A, frame_A_h, frame_A_w, mode_name, sub_mode, fps, zoom_factor, brightness, contrast, ipd)
    camera_B = puttext_img(camera_B, frame_B_h, frame_B_w, mode_name, sub_mode, fps, zoom_factor, brightness, contrast, ipd)
    
    # Apply radial distortion on image
    camera_A = apply_radial_distortion(camera_A, mtx, dist)
    camera_B = apply_radial_distortion(camera_B, mtx, dist)
      
    #fill image background
    fill_A = np.zeros([camera_A_h, camera_A_w, 3], dtype = np.uint8)
    fill_A  [ int(camera_A_h - eyebox_h - 5) : int(camera_A_h - 5) , int(camera_A_w//2 -int(eyebox_w//2)) : int(camera_A_w//2 + int(eyebox_w//2)) ] = camera_A
    fill_B = np.zeros([camera_B_h, camera_B_w, 3], dtype = np.uint8)
    fill_B  [ 5 : int(eyebox_h + 5) , int(camera_B_w//2 - int(eyebox_w//2)) : int(camera_B_w//2 + int(eyebox_w//2)) ] = camera_B
   
    new_frame = np.concatenate((fill_A,fill_B),axis=0)
    #new_frame = np.concatenate((camera_A,camera_B),axis=0)

    #-------------------------------------------User Input-----------------------------------------    

    #winname = "digieye"
    #cv.namedWindow(winname)        # Create a named window
    #cv.moveWindow(winname, 0,0)    # Move it to (0,0)
    #cv.imshow(winname, new_frame)
    
    key = cv.waitKey(1)
    time.sleep(0.05)
    
    if key == ord('q'):
      pool.close()
      pool.join()
    video_stream_widget.show_frame(new_frame,key)

    if button.pressed_shutdown:        
      pool.close()
      pool.join()
      key = 1
    video_stream_widget.show_frame(new_frame,key)
    
    if button.pressed_mode:#mode change:--> Scenic, Reading1, Reading2, Reading3, Setting
      if mode_select == 4:
        mode_select = 0
      else:  
        mode_select +=1 

    if button.pressed_submode:
      if mode_select == 0:#scenic mode change:--> Normal, Noglare, Edge
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
        if reading_select == 4: 
          reading_select = 0
        else:  
          reading_select +=1 
      if mode_select == 3:#reading mode 3 change:--> Normal, BW, BWHC, WBHC
        if reading_select == 4: 
          reading_select = 0
        else:  
          reading_select +=1 
      if mode_select == 4:#setting mode change:--> Brightness, Contrast, ipd, display mode, primary eye
        reading_select = 0
        if setting_select == 5: 
          setting_select = 0
        else:  
          setting_select +=1 
    
    if button.pressed_plus:
      if mode_select == 4:
        if setting_select == 0:
          if brightness < 100: 
            brightness+=1
          else:
            brightness = 100
        if setting_select == 1:
          if contrast < 100:
            contrast+=1
          else:
            contrast = 100
        if setting_select == 2:
          ipd+=1
        if setting_select == 3:
          if display_mode == 1:
            display_mode = 0
          else:
            display_mode = 1
        if setting_select == 4:
          if primary_eye == 'right':
            primary_eye = 'left'
          else:
            primary_eye = 'right'
      elif mode_select == 2:
        if overly_zoom_factor >= 6.0:
          overly_zoom_factor = 6.0
        else:  
          overly_zoom_factor = overly_zoom_factor + 0.2
      else:#zoom in:
        if zoom_factor >= 8.0:
          zoom_factor = 8.0
        else:  
          zoom_factor = zoom_factor + 0.2

    if button.pressed_minus:
      if mode_select == 4:
        if setting_select == 0:
          if brightness > 0: 
            brightness-=1
          else:
            brightness = 0
        if setting_select == 1:
          if contrast > 0: 
            contrast-=1
          else:
            contrast = 0
        if setting_select == 2:
          ipd-=1
        if setting_select == 3:
          if display_mode == 1:
            display_mode = 0
          else:
            display_mode = 1
        if setting_select == 4:
          if primary_eye == 'right':
            primary_eye = 'left'
          else:
            primary_eye = 'right'
      elif mode_select == 2:
        if overly_zoom_factor <= 1.0:
          overly_zoom_factor = 1.0
        else:  
          overly_zoom_factor = overly_zoom_factor - 0.2
      else:#zoom out
        if zoom_factor <= 1.0:
          zoom_factor = 1.0
        else:  
          zoom_factor = zoom_factor - 0.2
