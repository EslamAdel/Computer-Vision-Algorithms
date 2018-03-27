import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as color
from scipy import signal
from scipy.ndimage import filters

def fastDetect(image, t):
    '''
    Check if poit is a corner using 16 point on the circle. It will be a corner 
    if you have 5 or more pixels with absolute differenc from center pixel that 
    greater than threshold t. 
    
    Original algorithm is more sophisticated. Advanced machine learing is applied.
    Here you are asked to implement the basic idea with no more improvements. 
    Also we will ignore non-maxima suppression step. 
    
    inputs : 
          image : gray image
          t : threshold 
    output : Corner image
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''  
    cornerImage = image
    return cornerImage


def plotCorners(image, cornerImage):
    '''
    Plot detected corners from corner image superimposed on original image
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    #Show original image 
    # Mark up the corners on it
    pass


if __name__ == '__main__':
    #Load Image    
    image = plt.imread("images/BW.jpg")
    hsvImage = color.rgb_to_hsv(image)
    valIm = hsvImage[...,2]
    # Detect corners
    cornerImage = fastDetect(valIm,75)
    #Show Original Image
    plt.figure("Original Image")
    plt.imshow(image)
    plt.set_cmap("gray")
    #show corner image
    plt.figure("Corner Image")
    plt.imshow(cornerImage)
    #Plot corners
    plotCorners(image, cornerImage)
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    