import math
import numpy as np
#import matplotlib.pyplot as plt
import cv2 as cv


#f_img = '/home/chetan/Desktop/example.jpg'
f_img = '/home/chetan/Desktop/t3qWG.png' 
im_cv = cv.imread(f_img)
# grab the dimensions of the image
(h, w, _) = im_cv.shape
print('h,w:',h,w)

center_y = h//2
center_x = w//2


radius = int(1.8*h)
#radius_x = int(0.5*w)
#radius_y = int(0.5*h)
scale_x = 2      # smaller scale number, higher magnification 0.6,0.9
scale_y = 2
amount_x = 4    # higher amount(power), higher rate of magnification 6,6
amount_y = 4
shift = 0.5

factor_x = 1.0
factor_y = 1.0


# set up the x and y maps as float32
flex_x = np.zeros((h, w), np.float32)
flex_y = np.zeros((h, w), np.float32)
# create map with the barrel pincushion distortion formula
for y in range (h):
    delta_y = scale_y * (y - center_y)
    #delta_y = (y - center_y)
    for x in range (w):
    # determine if pixel is within an ellipse
        delta_x = scale_x * (x - center_x)
        #delta_x = (x - center_x)
        distance = delta_x * delta_x + delta_y * delta_y

        if distance >= (radius * radius):
        #if (delta_x*delta_x) >= (radius_x * radius_y) and (delta_y *delta_y) >= (radius_x *radius_y) :
            flex_x[y, x] = x
            flex_y[y, x] = y
        else :
            #if distance > (0.064 * (radius * radius)):
            if distance > 0.0:
            #if (delta_x * delta_x) >= 0.0 and (delta_y * delta_y) >= 0.0 :
                factor_x = math. pow (math.sin(math.pi * math.sqrt(distance ) /radius / 2), amount_x)
                factor_y = math. pow (math.sin(math.pi * math.sqrt(distance ) /radius / 2), amount_y)
            #    factor_x = math. pow (math.sin(math.pi * 0.25 / 2), amount_x)
            #    factor_y = math. pow (math.sin(math.pi * 0.25 / 2), amount_y)
                 
                flex_x[y, x] = factor_x * delta_x / scale_x + center_x
                flex_y[y, x] = factor_y * delta_y / scale_y + center_y
            #else:
                #factor_x = math. pow (math.sin(math.pi * (math.sqrt(distance * scale_x)+ shift) /radius / 2), amount_x)
                #factor_y = math. pow (math.sin(math.pi * (math.sqrt(distance * scale_y)+ shift) /radius / 2), amount_y)
                
                #factor_x = math. pow (math.sin(math.pi * 0.025 / 2), amount_x)
                #factor_y = math. pow (math.sin(math.pi * 0.025 / 2), amount_y)

                #factor_x = math. pow (math.sin(math.pi * math.sqrt(delta_x * delta_x) /radius_x / 2 ), amount_x)
                #factor_y = math. pow (math.sin(math.pi * math.sqrt(delta_y * delta_y) /radius_y / 2 ), amount_y)
                #flex_x[y, x] = factor_x * delta_x / scale_x + center_x
                #flex_y[y, x] = factor_y * delta_y / scale_y + center_y
                flex_x[y, x] = factor_x * delta_x / scale_x + center_x
                flex_y[y, x] = factor_y * delta_y / scale_y + center_y
            

# do the remap this is where the magic happens
im_cv = cv.line(im_cv,(int(0.1*w),int(0.4*h)),(int(0.9*w),int(0.4*h)),(0,255,0),2)
im_cv = cv.line(im_cv,(int(0.1*w),int(0.6*h)),(int(0.9*w),int(0.6*h)),(0,255,0),2)
dst = cv.remap(im_cv, flex_x, flex_y, cv.INTER_LANCZOS4)
cv.imshow( 'src' , im_cv)
cv.imshow( 'dst' , dst)
cv.waitKey( 0 )
cv.destroyAllWindows()
exit()