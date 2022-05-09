import cv2 as cv
import numpy as np
from threading import Thread
from multiprocessing import Process, Queue
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
      cv.destroyAllWindows()
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

# def ocr_img(q, cam_img):
def ocr_img(cam_img):
  print(' ocr process called')
  bounds = reader.readtext(cam_img)
  if bounds:
    # q.put(bounds)
    return draw_boxes(cam_img, bounds)
    # return bounds
  else:
  #   q.put(0)
    # return 0
    return cam_img
#------------------------------------------------------
# Parameters
#------------------------------------------------------
# ocr multiprocessing
flag_A = 1
flag_B = 1
dummy_bounds_A = []
dummy_bounds_A = []

# other
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
count = 0
if __name__ == '__main__':
  video_stream_widget = VideoStreamWidget()
  
  #q_A = Queue()
  #q_B = Queue()
  
  while True:
      print('loop count:',count)
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

      #-----------------------------------------Modes------------------------------------------------q
      mode_name = 'ocr'
      # start pocess
      #
      # camera_A = ocr_img(camera_A)
      # camera_B = ocr_img(camera_B)
      # img_process_A = Process(target=ocr_img, args=[ q_A, camera_A])
      # img_process_B = Process(target=ocr_img, args=[ q_B, camera_B])
      
      img_process_A = Process(target=ocr_img, args=[camera_A])
      img_process_B = Process(target=ocr_img, args=[camera_B])
      
      img_process_A.start()
      img_process_B.start()
      
      img_process_A.join()
      img_process_B.join()

      # if flag_A == 1 and flag_B == 1:
      #   img_process_A = Process(target=ocr_img, args=[ q_A, camera_A])
      #   img_process_A.start()
      #   img_process_B = Process(target=ocr_img, args=[ q_B, camera_B])
      #   img_process_B.start()
      #   print('started process A and B, processed bounds for A and B will be put in respective queues queue_A and queue_B')
      #   flag_A = 0
      #   flag_B = 0
        
      # bounds_A = q_A.get()
      # bounds_B = q_B.get()
      # print(bounds_A)
      # print(bounds_B)

      # if bounds_A.empty():
      #   print('no element in queue_A, passing dummy bounds')
      #   camera_A = draw_boxes(camera_B, dummy_bounds_A) 
      # else:  
      #   print('Process A done, element found in queue_A, passing bounds_A')
      #   camera_A = draw_boxes(camera_B, bounds_A) 
      #   dummy_bounds_A = bounds_A 
      #   img_process_A.join()
      #   flag_A = 1
           
      # if bounds_B.empty():
      #   print('no element in queue_B, passing dummy bounds')
      #   camera_B = draw_boxes(camera_B, dummy_bounds_B) 
      # else:  
      #   print('Process A done, element found in queue_B, passing bounds_B')
      #   camera_B = draw_boxes(camera_B, bounds_B) 
      #   dummy_bounds_B = bounds_B 
      #   img_process_B.join()
      #   flag_B = 1

      #-----------------------------------------Post-Processing--------------------------------------
      count+=1
      # fps calculation
      end = time.time()
      time_elapsed = end - start
      fps = 1 // time_elapsed
      
      # Display status texts
      cv.putText(camera_A,'fps:' + str(fps),(300,30),font,1,(255,0,0),1,cv.LINE_AA)
      cv.putText(camera_B,'fps:' + str(fps),(300,30),font,1,(255,0,0),1,cv.LINE_AA)
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
      
      #-------------------------------------------User Input-----------------------------------------
      #**********need to optimize***********
      key = cv.waitKey(1)
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
      # if key==ord('m') and key==ord('m'):#zoom in:
        # print('success')
      # if key==ord('q'):
      #   break
  # cam.release()
  # cv.destroyAllWindows()
    

