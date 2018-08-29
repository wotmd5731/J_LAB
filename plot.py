# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# Fixing random state for reproducibility
#np.random.seed(19680801)
#
#
#X = 10*np.random.rand(5, 3)
#
#fig, ax = plt.subplots()
#ax.imshow(X, interpolation='nearest')
#
#numrows, numcols = X.shape
#
#
#def format_coord(x, y):
#    col = int(x + 0.5)
#    row = int(y + 0.5)
#    if col >= 0 and col < numcols and row >= 0 and row < numrows:
#        z = X[row, col]
#        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
#    else:
#        return 'x=%1.4f, y=%1.4f' % (x, y)
#
#ax.format_coord = format_coord
#plt.show()

#------------------------------
##
#fig, axs = plt.subplots(2, 2)
#
#axs[0, 0].imshow(np.random.random((100, 100)))
#
#axs[0, 1].imshow(np.random.random((100, 100)))
#
#axs[1, 0].imshow(np.random.random((100, 100)))
#
#axs[1, 1].imshow(np.random.random((100, 100)))
#
#plt.subplot_tool()
#plt.show()

## Fixing random state for reproducibility
#np.random.seed(19680801)
#
plt.figure(figsize=(15,10))
def draw22(X):
#    
#    for xx in X:
    plt.clf()
    plt.subplot(141)
    plt.imshow(X[0], cmap=plt.cm.BuPu_r)
    plt.subplot(142)
    plt.imshow(X[1], cmap=plt.cm.BuPu_r)
    plt.subplot(143)
    plt.imshow(X[2], cmap=plt.cm.BuPu_r)
    plt.subplot(144)
    plt.imshow(X[3], cmap=plt.cm.BuPu_r)


    
    
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
#    plt.show()
    plt.pause(0.05)

#    plt.plot()

#draw([np.random.rand(10,10),np.random.rand(10,10),np.random.rand(10,10),np.random.rand(10,10)])


#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
#
#X = np.arange(-5, 5, 0.25)
#Y = np.arange(-5, 5, 0.25)
#X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
##Z = np.sin(R)
#Z = np.random.rand(40, 40)
#
#
#
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
#
#plt.show()