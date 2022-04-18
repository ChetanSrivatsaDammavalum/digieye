import cv2 as cv
import numpy as np
import utils
from config import parameters
# brightness, contrast, ipd, overlay(h_pos, w_pos, h_size, w_size)

class settings:

#---------

# set IPD
#TBD#

#---------
#---------

# set Brightness
img = brightness_img(img, brightness)

#---------
#---------

# set Contarst 
img = contrast_img(img, contrast)   

#---------



class preprocessing(settings):

#---------

# image size orientation and pos manipulation  
imga_a, img_b = split_img(img)
#---------
#---------

# Image glare removal
def glar_img(img):
  img = bgr_hsv(img)
  h, s, v = split_channel_img(img)
  ret,v = threshold_img(v)
  img = merge_channel_img((h,s,v))
  return hsv_bgr(img)

#---------
#---------

# Extract resized overlay image
img_ovrly = resize_img(img, h, w)

#---------
#---------

# zoom in/out
img = crop_img(img, h_crop, w_crop)
img = resize_img(img, h, w)

#---------

class reading:
#---------

# overlay img
img = overly(img,img_ovrly, h_pos, w_pos)

#---------

#---------

# convert img to BW

#---------

#---------

# convert img to BW-HC

#---------

class scenic:
#---------

# overlay edges

#---------

#---------

# apply bubble distortion

#---------


class processing(scenic,reading):

#---------

# set mode

#---------
#---------

# set submode

#---------
#---------

# zoom in/out

#---------
#---------

# text overlay

#---------







