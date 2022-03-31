import numpy as np
#from wand.image import Image
import cv2
import glob
import time

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*13,3), np.float32) #chess board has 10x14 grid with 9x13 internal corners
objp[:,:2] = np.mgrid[0:9,0:13].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

path = '/home/digieye/Desktop/digieye_main/calibration/'
#path = '/home/digieye/Desktop/digieye_main/calibration/camera_normal/'
images = glob.glob(path + '*.jpg')

dispH=1080//2 
dispW=1920//2

#dispH=1080 
#dispW=1920

#mtx = np.array([[ 102.92304868, 0.0, 624.00010882 ],[ 0.0, 286.10318854,  623.99997651 ],[ 0.0, 0.0, 1.0 ]])
#dist = np.array([[ -1.64128705e-02, 7.21592651e-05, 1.55204297e-03, 2.67011805e-03, -8.04152141e-08 ]])

#mtx = np.array([[ 637.44018555, 0.0, 624.0 ],[ 0.0, 212.66660774, 624.0 ],[ 0.0, 0.0, 1.0 ]])
#dist = np.array([[ -1.69956852e-03, 8.48746652e-07, 2.73780637e-04, -1.35316424e-04, -1.21446721e-10]])

#mtx = np.array([[ 3.41333934e+04, 0.00000000e+00, 3.10891046e+03],[ 0.00000000e+00, 5.08659037e+03, 8.18202447e+02],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#dist = np.array([[  3e+01,  -1e+04, -9e-01,  -1e+00,    5e+05]])

#mtx = np.array([[ 1.38116563e+04, 0.00000000e+00, 5.05714420e+02 ],[ 0.00000000e+00, 2.45985175e+04, 3.57315723e+02 ],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#dist = np.array([[ 2.68477150e+02, -8.11525354e+04, -1.44799892e+00, -1.11433899e+00, -2.07015935e+02]])

#original camera matrix plus derived distortion above
#mtx = np.array([[ 1.64660910e+05, 0.00000000e+00, 6.79444298e+02 ],[ 0.00000000e+00, 8.79941816e+04, 1.44588328e+03 ],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00 ]])
#dist = np.array([[ -2.47941173e+02, 2.09957743e+05, 7.09693546e+00, 1.93719496e+00, 3.11508885e+02 ]])

#mtx = np.array([[ 7.39490934e+04, 0.00000000e+00, 6.45939205e+02 ],[ 0.00000000e+00, 1.82389546e+05, 1.00326992e+03 ],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#dist = np.array([[ 2.25579275e+03, 9.22447474e+05, 3.20905472e+01, 6.60018353e+00, 1.53352294e+02 ]])

#modelling human eye for camera mtx: 338 ppi @ nearpoint:250cm or 14ppmm, with vr lens positioned at 49mm, 
#distance of focus from eye to screen is 686pixels ie, f_x = f_y = 686
#mtx = np.array([[ 6.86000000e+05, 0.00000000e+00, 0.00000000e+00 ],[ 0.00000000e+00, 6.86000000e+05, 0.00000000e+00 ],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
mtx = np.array([[ 3.00000000e+06, 0.00000000e+00, 0.00000000e+00 ],[ 0.00000000e+00, 3.00000000e+06, 0.00000000e+00 ],[ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 1.70000000e+05 , 1.00000000e+04 , 0.00000000e+00, 0.00000000e+00, 1.00000000e+03 ]])

count=0 

for fname in images:
    count+=1
    print('filename', fname)
    img = cv2.imread(fname)
    start = time.time()
    h, w = img.shape[:2]
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    mtx[0,2] = w //2
    mtx[1,2] = h //2

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    #dst = cv2.undistort(img, mtx, dist, None, mtx)
    end = time.time()
    time_of_distortion = start - end
    #img = cv2.resize(gray,(dispW//2,dispH),interpolation = cv2.INTER_LINEAR)
    #img_dist = cv2.resize(dst,(dispW//2,dispH),interpolation = cv2.INTER_LINEAR)
    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    #Image.virtual_pixel = 'transparent'
    #Image.distort('barrel', (0.2, 0.0, 0.0, 1.0))

    img_shw = np.concatenate((img,dst),axis=1)
    img_shw = cv2.resize(img_shw,(dispW,dispH),interpolation = cv2.INTER_LINEAR)

    cv2.imshow('img',img_shw)
    print('time_of_distortion',time_of_distortion)

    key = cv2.waitKey(0)
    #if key==ord('m'): 
    #    print('success')
    if key==ord('q'):
        break

print('Count:', count)

cv2.destroyAllWindows()
