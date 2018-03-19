import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as color

def fastDetect(image, t):
    '''
    Check if poit is a corner using 4 points 1, 5, 9, and 13
    inputs : 
        image : 
        t : threshold 
    output : binary image with ones at corner points and zero otherwise
    '''
    cornerImage = np.zeros(image.shape)
    # Get points 1, 5, 9, 13
    for i in range(3,image.shape[0]-3):
        for j in range(3,image.shape[1]-3):
            p  = image[i,j]
            p1 = image[i-3,j] 
            p5 = image[i,j+3] 
            p9 = image[i+3,j] 
            p13 = image[i,j-3]
            count = 0
            if abs(p-p1) > t:
                count +=1
            if abs(p-p5) > t:
                count +=1 
            if abs(p-p9) > t:
                count +=1 
            if abs(p-p13) > t:
                count +=1 
            
            if count >= 2: 
                cornerImage[i,j] = 1

    return cornerImage

def nonMaxSupp(cornerImage):
    
    corners = np.array(np.nonzero(cornerImage)).T
    for p in corners:
        if not isMax(cornerImage,p):
            cornerImage[p[0], p[1]] = 0
    return cornerImage

def isMax(image,p):
    otherCorners = np.array(np.nonzero(image[p[0]-2:p[0]+5,p[1]-2:p[1]+5])).T
    wp = weighFunction(image,p)
    for c in otherCorners:
        #Calculate Weight function
        c = c + p - 1
        if c[0] != p[0] and c[1] != p[1]:
            wc = weighFunction(image,c)
            if wc >= wp :
                return 0
    return 1


def weighFunction(image,p):
    i = p[0]
    j = p[1]
    try:
        subIm = np.array([image[i+3,j-1:j+2], image[i-3,j-1:j+2], image[i-1:i+2,j+3], image[i-1:i+2,j-3], image[i+2,j+2],image[i-2,j+2],image[i-2,j-2],image[i+2,j-2]])
        weight = np.sum(np.sum(np.abs(subIm - image[i,j])))
    except:
      weight = abs(image[i,j])  
    return weight

def plot_corner_points(image,filtered_coords):
    """ Plots corners found in image. """
    plt.figure()
    plt.set_cmap('gray')
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'+',color='red')
    plt.axis('off')
    plt.show()

image = plt.imread("images/squares.jpg")
hsvImage = color.rgb_to_hsv(image)
corners = fastDetect(hsvImage[...,2],150)
corners = nonMaxSupp(corners)
cors = np.array(np.nonzero(corners)).T
plot_corner_points(image, cors)
#plt.imshow(corners)
#plt.set_cmap("gray")
#plt.show()