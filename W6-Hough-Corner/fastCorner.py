import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as color
from scipy import signal
from scipy.ndimage import filters

def fastDetect(image, t):
    '''
    Check if poit is a corner using 4 points 1, 5, 9, and 13
    inputs : 
        image, t : threshold 
    output : binary image with ones at corner points and zero otherwise
    '''
    #Initialization 
    cornerImage = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # try and except to avoid boundary condition 
            try:
                #Point to test
                p  = image[i,j]
                # Get points 1 to 16
                
                points = [image[i+3,j-1:j+2], image[i-3,j-1:j+2],
                                   image[i-1:i+2,j+3], image[i-1:i+2,j-3], 
                                   image[i+2,j+2],image[i-2,j+2],
                                   image[i-2,j-2],image[i+2,j-2]]
#                points = [image[i+3,j], image[i-3,j], image[i,j+3], image[i,j - 3]]
                #Counter holds number of pixels > threshold
                count = 0
                for a in points:
                    for point in a:
                        # Calculate absolute difference 
                        if abs(p-point) > t:
                            count +=1
#                        else:
#                            count = 0
                    # Check number of pixels that have same  diff > threshold
                    # Original Fast stated that it must be >= 3 
                    # But for count >= 3 corners with 90 degree will not be detected
                    # So I made it >= 2
                    if count >= 5: 
                        cornerImage[i,j] = 1
            except:
                pass
    
    # Return corner image
    return cornerImage



def nonMaxSupp(cornerImage):
    '''
    It is the second main step in fast algorithm
    The idea is to suppress corners that weaker than neighbor corners
    '''
    
    # Select corners only from cornenr image
    corners = np.array(np.nonzero(cornerImage)).T
    # Loop for each corner point
    for p in corners:
        #Check if it is the local maximum or not
        if not isMax(cornerImage,p):
            #if not so suppress
            cornerImage[p[0], p[1]] = 0
    #Return cornenrs after non-maxima suppression
    return cornerImage

def isMax(image,p):
    '''
    Checking if the point is local max
    for all nonzero neighbors calculate a score or weight
    true if it is max its neighbors.
    input : image , p is corner point to be tested
    '''
    #Select nonzero neighbors 
    otherCorners = np.array(np.nonzero(image[p[0]-3:p[0]+4,p[1]-3:p[1]+4])).T
    # Weight for the corner point
    wp = weighFunction(image,p)
    for c in otherCorners:
        #Calculate Weight function
        # Map values to image indices. 
        c = c + p - 2
        #Chek that it is another point
        if c[0] != p[0] and c[1] != p[1]:
            #Get weight for it 
            wc = weighFunction(image,c)
            #if not max so return false
            if wc >= wp :
                return 0
    #Return true
    return 1


def weighFunction(image,p):
    '''
    Caculate the weight or score of corner point p in the image
    imput: image, point(p)
    output : score or weight
    '''
    i = p[0]
    j = p[1]
    # try for boundary conditions
    try:
        # Select 16 surounding points on a circle of radius 3 
        subIm = np.array([image[i+3,j-1:j+2], image[i-3,j-1:j+2], image[i-1:i+2,j+3], image[i-1:i+2,j-3], image[i+2,j+2],image[i-2,j+2],image[i-2,j-2],image[i+2,j-2]])
        # the weight is sum of absolute difference
        weight = np.sum(np.sum(np.abs(subIm - image[i,j])))
    except:
        # Approximation of weight of boundary points
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


if __name__ == '__main__':
    #Load Image    
    image = plt.imread("images/BW2.jpg")
    hsvImage = color.rgb_to_hsv(image)
    im = hsvImage[...,2]
    # Working on value channel
    # Detect corners using fast
#    im = np.zeros((100,100))
#    im[25:75,25:75] = 255
#    im1 = filters.gaussian_filter(im,1)
    corners = fastDetect(im,75)
    #Apply non-max suppression
    scorners = nonMaxSupp(corners)
    #Plot corners
    coords = np.array(np.nonzero(scorners)).T
    plot_corner_points(im, coords)
  