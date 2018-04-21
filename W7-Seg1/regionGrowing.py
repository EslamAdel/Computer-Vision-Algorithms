import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def onClick(event):
    x, y = int(event.xdata), int(event.ydata)
    outIm = regionGrowing(valChannel,np.array([y,x]),10)
    segmentedImage = np.copy(myImage)
    segmentedImage[...,0] *= outIm.astype('uint8')
    segmentedImage[...,1] *= outIm.astype('uint8')
    segmentedImage[...,2] *= outIm.astype('uint8')
    plt.figure()
    plt.imshow(segmentedImage)    
    plt.show()
    
def regionGrowing(image, seed, threshold):
    # Initilize points List
    points = []
    outImage = np.zeros(image.shape)
    points.append(seed)
    outImage[seed[0],seed[1]] = 1
    while len(points):
        p = points.pop()
        neighbors = np.array([[p[0]-1,p[1]] ,
                              [p[0]+1,p[1]] ,
                              [p[0], p[1]-1],
                              [p[0], p[1]+1]])
        try:
            for i in range(4):
                if not outImage[neighbors[i,0],neighbors[i,1]]:
                    if np.abs(image[neighbors[i,0],neighbors[i,1]] - image[p[0],p[1]]) <= threshold:
                        outImage[neighbors[i,0],neighbors[i,1]] = 1
                        points.append(np.array([neighbors[i,0],neighbors[i,1]]))
        except:
            continue
        
    return outImage
                


if __name__ == '__main__':
    global myImage
    myImage = plt.imread('images/seg1.jpg')
    hsvImage = colors.rgb_to_hsv(myImage)
    global valChannel 
    valChannel = hsvImage[...,2]
    fig = plt.figure()
    plt.imshow(myImage)
    plt.set_cmap('gray')
    fig.canvas.mpl_connect('button_press_event', onClick)
    plt.show()
    
    
