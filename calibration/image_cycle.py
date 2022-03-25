import numpy as np
import cv2
import glob
import numpy as np


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*13,3), np.float32) #chess board has 10x14 grid with 9x13 internal corners
objp[:,:2] = np.mgrid[0:9,0:13].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

path = '/home/digieye/Desktop/digieye_main/calibration/camera_normal/'
images = glob.glob(path + '*.jpg')

dispH=1080 
dispW=1920
count=0 
for fname in images:
    count+=1
    print('filename', fname)
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray,(dispW//2,dispH),interpolation = cv2.INTER_LINEAR)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = np.concatenate((img,img),axis=0)

    cv2.imshow('img',img)

    key = cv2.waitKey(0)
    #if key==ord('m'): 
    #    print('success')
    if key==ord('q'):
        break

print('Count:', count)

cv2.destroyAllWindows()
