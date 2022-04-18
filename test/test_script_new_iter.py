import cv2 as cv
import numpy as np
import time
from threading import Thread
import multiprocessing as mp
import easyocr
from PIL import ImageDraw, Image

# print(cv.__version__)

font = cv.FONT_HERSHEY_SIMPLEX

# for pc display
dispW=int(1920 * 0.5) # WIDTH OF OUTPUT IMAGE
dispH=int(1080 * 0.5) # HEIGHT OF ROTATED OUTPUT IMAGE
flip=2 # vertically up oriented, imgB+imgA
#flip=0 # vertically down oriented, imgB+imgA

#dispW=int(1080 * 0.5) # WIDTH OF ROTATED OUTPUT IMAGE
#dispH=int(1920 * 0.5) # HEIGHT OF ROTATED OUTPUT IMAGE
#flip=3 # horizontally left oriented, imgB+imgA
#flip=1 # horizontally right oriented, imgB+imgA
#---------------------------------------------------------------------------------------------------------------------
# for vr display
#dispW=1080 # WIDTH OF ROTATED OUTPUT IMAGE
#dispH=1920 # HEIGHT OF ROTATED OUTPUT IMAGE
#flip=3 # horizontally left oriented, imgB+imgA
#flip=1 # horizontally right oriented, imgB+imgA

# camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=1848, format=NV12, framerate=28/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! vaapipostproc ! video/x-raw, denoise = 5 ! videoconvert ! video/x-raw, format=BGR ! appsink'
# cam= cv.VideoCapture(camSet)

#-----------------------------------------------------------------
# camera thread method
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
      exit(1) # instead of exit(0)

#-----------------------------------------------------------------
# ocr function

reader = easyocr.Reader(['de','en'])

def draw_boxes(image, bounds, color='blue', width=1):
    print('box draw called')
    draw = ImageDraw.Draw(Image.fromarray(image))
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        # print(p0, p1 ,p2 ,p3)
        p21, p22 = np.array(p2)
        p31, p32 = np.array(p3)
        # print(p11,p22,p21,p22)
        # draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        # draw.line([ *p1, *p2 ], fill=color, width=width)
        cv.line(image,(int(p21),int(p22)),(int(p31),int(p32)),(255,0,0),2)
    return image 

def ocr_img(cam_img):
  print('ocr called')
  bounds = reader.readtext(cam_img)
  print('bounds calculated',bounds)
  return bounds
  # if bounds:
    # return draw_boxes(cam_img, bounds)
    # return bounds
  #else:
    # return cam_img
  #  return 0

#-----------------------------------------------------------------
# multiprocessig function

def process_images(images):
  print('image is being processed')
  return images
 


#-----------------------------------------------------------------
# main code
if __name__ == '__main__':
  video_stream_widget = VideoStreamWidget()
  
  mp.set_start_method('spawn')

  while True:
    start = time.time()

    video_frame = video_stream_widget.frame

    #---------------------------------------------------------
    # preprocessing

    frame_h = video_frame.shape[0]
    frame_w = video_frame.shape[1]
    camera_A = video_frame
    camera_B = video_frame
    camera_A_h = camera_A.shape[0]
    camera_A_w = camera_A.shape[1]
    camera_B_h = camera_B.shape[0]
    camera_B_w = camera_B.shape[1] 

    #---------------------------------------------------------
    # main process
    img_pair = [camera_A, camera_B]
    
    pool = mp.Pool(2)
    # img_p = pool.map(ocr_img, img_pair)
    line_pos = pool.map(ocr_img, img_pair)
    pool.close()
    pool.join()
    
    camera_A = draw_boxes(camera_A, line_pos[0])
    camera_B = draw_boxes(camera_B, line_pos[1])

    #cv.imshow('img_pair_1', img_p[1])
    #cv.imshow('img_pair_0', img_p[0])

    
    #camera_A = ocr_img(camera_A)
    #camera_B = ocr_img(camera_B)

    #---------------------------------------------------------
    # post processing

    end = time.time()
    time_elapsed = end - start
    fps = 1 // time_elapsed

    cv.putText(camera_A,'fps:' + str(fps),(300,30),font,1,(255,0,0),1,cv.LINE_AA)
    cv.putText(camera_B,'fps:' + str(fps),(300,30),font,1,(255,0,0),1,cv.LINE_AA)
    
    new_frame = np.concatenate((camera_A,camera_B),axis=1)
    
    key = cv.waitKey(1)
    video_stream_widget.show_frame(new_frame,key)
    



