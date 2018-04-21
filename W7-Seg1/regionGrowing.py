import numpy as np
import matplotlib.pyplot as plt


def idxFromPosition(pos, ImShape):
    idx = pos[0]*ImShape[1] + pos[1]
    return idx

def posFromIdx(idx, ImShape):
    pos = np.zeros(2)    
    pos[0] = np.int(idx/ImShape[1]) 
    pos[1] = np.int(idx % ImShape[1])
    return pos



def regionGrowing(image, seed):
    points = np.zeros(image.shape[0]*image.shape[1])
    VisitedPoints = np.zeros(image.shape[0]*image.shape[1])
    outImage = np.zeros(image.shape)
    points[idxFromPosition(seed,image.shape)] = 1
    VisitedPoints[idxFromPosition(seed,image.shape)] = 1 
    outImage[seed[0],seed[1]] = 1
    threshold = 5
    while np.size(np.where(points == 1)):
        allPts = np.array(np.where(points == 1)).T
        p = posFromIdx(allPts[0],image.shape)
        points[allPts[0]] = 0
        try:
            neighbors = np.array([[p[0]-1,p[1]] ,
                                  [p[0]+1,p[1]] ,
                                  [p[0], p[1]-1],
                                  [p[0], p[1]+1]])
        except:
            continue
        for i in range(4):
            if not VisitedPoints[idxFromPosition(neighbors[i,:], image.shape)]:
                if np.abs(image[neighbors[i,0],neighbors[i,1]] - image[p[0],p[1]]) <= threshold:
                    outImage[neighbors[i,0],neighbors[i,1]] = 1
                    points[idxFromPosition(neighbors[i,:], image.shape)] = 1
                    VisitedPoints[idxFromPosition(neighbors[i,:], image.shape)] = 1
                
                
    return outImage
                


if __name__ == '__main__':
    image =np.zeros((100,100))
    image[25:75, 25:75] = 255
    plt.figure()
    plt.imshow(image)
    plt.set_cmap('gray')
    outIm = regionGrowing(image,np.array([50,50]))
    plt.figure()
    plt.imshow(outIm)    
    plt.show()