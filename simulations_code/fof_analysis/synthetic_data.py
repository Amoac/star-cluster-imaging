import matplotlib.pyplot as plt
import numpy as np
import fof
import pdb
from importlib import reload

#to reload imported function fof: reload(fof)

points = 1000
halos = np.array([0.,100,150.])

#generate 3 data sets
#dataset 1: id 0-->999, dataset 2: id 1000 --> 1999; dataset 3: id 2000 --> 2999
#try generating ids inside fof find code
#id = np.arange(3*points)

x = np.random.rand(3*points)
y = np.random.rand(3*points)
z = np.random.rand(3*points)

#populate 1000 points with ids 0:999 lying between 0 and 1 for x, y, z
for i in np.arange(0, points, 1):
    x[i] = x[i] + halos[0]
    y[i] = y[i] + halos[0]
    z[i] = z[i] + halos[0]

#populate 1000 points with ids 1000:1999 lying between 100 and 101 for x, y, z 
for i in np.arange(points, 2*points, 1):
    x[i] = x[i] + halos[1]
    y[i] = y[i] + halos[1]
    z[i] = z[i] + halos[1]

for i in np.arange(2*points, 3*points, 1):
    x[i] = x[i] + halos[2]
    y[i] = y[i] + halos[2]
    z[i] = z[i] + halos[2]

#now x, y, z are 3000 elements long each and ids go from 0 --> 2999

indices, xcm, ycm, zcm, mtot, grpid, r_90, r_max = fof.find(x,y,z,b=1)

#print len(indices)
#pdb.set_trace()

##plot the data to check
plt.clf()
plt.plot(x[indices[0]], y[indices[0]], 'ro')
if len(grpid) > 1:
    plt.plot(x[indices[1]], y[indices[1]], 'bo')
    if len(grpid) > 2:
        plt.plot(x[indices[2]], y[indices[2]], 'go')
plt.ylabel('y')
plt.xlabel('x')
plt.show()



