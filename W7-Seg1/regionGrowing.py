import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def idxFromPosition(pos, ImShape):
    idx = pos[0]*ImShape[1] + pos[1]
    return np.int(idx)

def posFromIdx(idx, ImShape):
    pos = np.zeros(2,np.int)   
    pos[0] = np.int(idx/ImShape[1]) 
    pos[1] = np.int(idx % ImShape[1])
    return pos

def onClick(event):
    x, y = int(event.xdata), int(event.ydata)
    outIm = regionGrowing(valChannel,np.array([y,x]),10)
    plt.figure()
    plt.imshow(outIm)    
    plt.show()

def onClose(event):
    exit(0)
    
def regionGrowing(image, seed, threshold):
    points = np.zeros(image.shape[0]*image.shape[1])
    VisitedPoints = np.zeros(image.shape[0]*image.shape[1])
    outImage = np.zeros(image.shape)
    points[idxFromPosition(seed,image.shape)] = 1
    VisitedPoints[idxFromPosition(seed,image.shape)] = 1 
    outImage[seed[0],seed[1]] = 1
    while np.size(np.where(points == 1)):
        allPts = np.array(np.where(points == 1)).T
        p = posFromIdx(allPts[0],image.shape)
        points[allPts[0]] = 0
        neighbors = np.array([[p[0]-1,p[1]] ,
                              [p[0]+1,p[1]] ,
                              [p[0], p[1]-1],
                              [p[0], p[1]+1]])
        try:
            for i in range(4):
                if not VisitedPoints[idxFromPosition(neighbors[i,:], image.shape)]:
                    if np.abs(image[neighbors[i,0],neighbors[i,1]] - image[p[0],p[1]]) <= threshold:
                        outImage[neighbors[i,0],neighbors[i,1]] = 1
                        points[idxFromPosition(neighbors[i,:], image.shape)] = 1
                        VisitedPoints[idxFromPosition(neighbors[i,:], image.shape)] = 1
        except:
            continue
                
                
    return outImage
                


if __name__ == '__main__':
    image = plt.imread('images/seg1.jpg')
    hsvImage = colors.rgb_to_hsv(image)
    global valChannel 
    valChannel = hsvImage[...,2]
    fig = plt.figure()
    plt.imshow(valChannel)
    plt.set_cmap('gray')
    fig.canvas.mpl_connect('button_press_event', onClick)
    fig.canvas.mpl_connect('close_event', onClose)
    
    