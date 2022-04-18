import cv2 as cv
import numpy as np

#------------------------------------------------------------------
# Transformation functions
#------------------------------------------------------------------
# splits image into two along the longer dimension
def split_img(img):
  h, w = cam_img.shape[:2]
  if w > h:               #horizontal
    split_len = h//2
    img_a = img[ : , split_len : ]
    img_b = img[ : , : split_len ]
  else:                   #vertical
    split_len = h//2
    img_a = img[ split_len : , : ]
    img_b = img[ : split_len , : ]
  
  return img_a, img_b
 
# Flip image to desired orientation: ax = 0(flip vertically), 1(flip horizontally), -1(flip 
# vertically and horizontally)
def flip_img(img, ax):
  return cv.flip(img, ax)

# Concatenate two imges: ax = 0(horizontal) ,1(vertical)
def concat_img(img_a, img_b, ax):
  return np.concatenate((img_a, img_b),axis=ax)

# Crop image to the required size (h_size, w_size)
def crop_img(img, h_size, w_size):
    h, w = img.shape[:2]    
    img = img[int(w//2 - w_size//2):int(w//2 + w_size//2), int(h//2 - h_size//2):int(h//2 + h_size//2)

# Resize the image to the desired size (h_size, w_size)
def resize_img(img, h_size, w_size):
  return cv.resize(img,(w_size, h_size),interpolation = cv.INTER_LANCZOS4)

# Overly an image over another image at pixel start position (h_pos, w_pos)
def overly(img, img_ovrly, h_pos, w_pos)
  h_ovrly, w_ovrly = .shape[:2]
  img[ h_pos : h_ovrly , w_pos  : w_ovrly ]= img_ovrly
  return img

# Radially distort image to correct for VR lens distortion (Barrel distortion)***Look for simpler method like remap function******************************************************************
def barrel_img(img, c_mtx, d_mtx):
  h, w = img.shape[:2]
  c_mtx[0,2] = w // 2 # shift camera pixel center to image center
  c_mtx[1,2] = h // 2 # shift camera pixel center to image center
  newc_mtx, roi=cv.getOptimalNewCameraMatrix(c_mtx,d_mtx,(w,h),1,(w,h))
  return cv.undistort(img, c_mtx, d_mtx, None, newc_mtx)
  
# Distort image with bubble/explode distortion for localized magnification*****Figure out code*****************************************************Shift function code snippet to post processing module*******************************************
def barrel_img(img, c_mtx, d_mtx):
  h, w = img.shape[:2]
  c_mtx[0,2] = w // 2 # shift camera pixel center to image center
  c_mtx[1,2] = h // 2 # shift camera pixel center to image center
  newc_mtx, roi=cv.getOptimalNewCameraMatrix(c_mtx,d_mtx,(w,h),1,(w,h))
  return cv.undistort(img, c_mtx, d_mtx, None, newc_mtx)

#------------------------------------------------------------------
# Manipulation functions
#------------------------------------------------------------------

# Change the brightness of input image****Need to set limits and increamentsteps
def brightness_img(img, brightness):     
  if brightness != 0:
    if brightness > 0:
        shadow = brightness
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + brightness
    alpha_b = (highlight - shadow)/255
    gamma_b = shadow
    return cv.addWeighted(img, alpha_b, cam_img, 0, gamma_b)
  return img

# Change the contrast of input image****Need to set limits and increament steps    
def contrast_img(img, contrast):     
  if contrast != 0:
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    return cv.addWeighted(img, alpha_c, img, 0, gamma_c)
  return img

# Change image from color(bgr) to gray
def bgr_gray(img):
  return cv.cvtcolor(img, cv.COLOR_BGR2GRAY)

# Change image from gray to color(bgr) 
def gray_bgr(img):
  return cv.cvtcolor(img, cv.COLOR_GRAY2BGR)

# Change image channels from b,g,r to h,s,v 
def bgr_hsv(img):
  return cv.cvtcolor(img, cv.COLOR_BGR2HSV)

# Change image channels from h,s,v to b,g,r 
def hsv_bgr(img):
  return cv.cvtcolor(img, cv.COLOR_HSV2BGR)

# split image to individual channels: returns n separate arrays for n channels in image
def split_channel_img(img):
  return cv.split(img)

# merge individual channels to image : returns image and takes in n channels of same shape 
def merge_channel_img(ch1,ch2,ch3):
  return cv.merge((ch1,ch2,ch3))
  
# Threshold pixel value in image: returns output flag, thresholded image array and takes in single channel image array
def threshold_img(chanl):
  return cv.threshold(chanl,180,255,cv.THRESH_TRUNC)


# Blurr image using gaussian blurr function with filter size (3,3)
def blurr_img(img):  
  return cv.GaussianBlur(img,(3,3),0) 

# Extract edges in image using canny edge method
def edge_img(img):
  return cv.Canny(img,80,200)
  
# merge two images
def merge_img(img_a,img_b):
  return cv.addWeighted(img_a, 1, img_b, 0.4, 1.2 )


