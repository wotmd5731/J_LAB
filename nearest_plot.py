import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,4,figsize=(10,5))

def nearest_plot(XX):
    global fig,ax
    for i,X in enumerate(XX):
        ax[i].imshow(X,interpolation='nearest')
        numrows,numcols=X.shape

        def format_coord(x,y):
            col = int(x+0.5)
            row = int(y+0.5)
            if col>=0 and col<numcols and row>=0 and row <numrows:
                z=X[row,col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x,y,z)
            else:
                return 'x=%1.4f, y=%1.4f ' % (x,y)
        ax[i].format_coord= format_coord

    plt.pause(0.001)


if __name__=='__main__':
    np.random.seed(1000)
    X=[10*np.random.rand(5,3) for i in range(4) ]
    nearest_plot(X)



