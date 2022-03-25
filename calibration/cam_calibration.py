import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*13,3), np.float32) #chess board has 10x14 grid with 9x13 internal corners
objp[:,:2] = np.mgrid[0:9,0:13].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

path = '/home/digieye/Desktop/digieye_main/calibration/'
images = glob.glob(path + '*.jpg')
mean_error = 0
tot_error = 0
count = 0

# Calibrating camera
for fname in images:
    print('filename', fname)
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (13,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        count+=1
        objpoints.append(objp)
        #print('objpoints:', objpoints)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        #print('imgpoints:', imgpoints)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (13,9), corners2,ret)
        img = cv2.resize(img,(int(w * 0.2),int(h * 0.2)),interpolation = cv2.INTER_LINEAR)
        cv2.imshow('img',img)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # Printing out camera matrix
        print('Derived camera mtx:', mtx)
        # Printing out distortion matrix 
        print('Derived distortion mtx:', dist)
        # Calculating Re-Projection error: estimate accuracy of derived parameters
     
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error
        print('total error: ', mean_error/len(objpoints)) 

        cv2.waitKey(500)

print('Count:', count)
# Camera original
# Derived camera mtx: [[  9.51493418e+02   0.00000000e+00   1.56000002e+03]
# [  0.00000000e+00   3.02679563e+02   1.56000001e+03]
# [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
# Derived distortion mtx: [[ -2.40746795e-02   1.34025503e-04  -9.59547051e-04   4.20920362e-04
#   -2.13270837e-07]]

cv2.destroyAllWindows()
