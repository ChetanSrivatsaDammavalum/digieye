import math
from cv2 import line
import numpy as np
import matplotlib.pyplot as plt
import random

# 1d array simulating a single row sweep of an image

w = 640
weep_r = np.random.rand(w)
map_x = np.ones(w, np.float32)
dist = np.ones(w, np.float32)
delta = np.ones(w, np.float32)

factor = np.ones(w, np.float32)

radius_factor = 0.7

radius = w * radius_factor 
pow = np.arange(0,4,0.2,dtype=np.float32)
line = []
#fig, ax = plt.subplots()

for k in pow:
    for x in range(w):
        delta[x] = (x - w//2)
        dist[x] = delta[x] * delta[x]
        if dist[x] >= (radius * radius):
            map_x[x] = x
        else: 
            factor[x] = math.pow(math.sin(math.pi * math.sqrt(dist[x])/radius/2 ),k)
            #print('factor,k',factor[x],k)
            map_x[x] = factor[x] * delta[x]  + w//2
    #line[x], = ax.plot(delta, factor,label='k '+str(k))
    plt.plot(delta, factor,label='k '+str(k))
    plt.plot()
    #ax.legend()
plt.xlabel('delta')
plt.ylabel('factor')
plt.title('bubble mag function analysis')
plt.show()
    

