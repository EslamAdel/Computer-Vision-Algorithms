from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors



def getHarris(image,sigma):
    '''
    Compute Harris operator for the image 
    inputs: 
        image: Original gray image
        sigma: sigma of gaussian filter
    output : Harris image
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''    
    H = image
    return H


def plotHarris(image, harrisIm, threshold):
    '''
    Plot selected harris corners on the original image. Selection of corners is 
    based on the threshold (fraction of max value in harris image). 
    inputs: 
        image: Original Image
        harrisIm: Harris image 
        threshold: The threshold 
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    ''' 
    # Find top corners according to threshold 
    # Show original image 
    # Mark up the corners on it
    pass


if __name__ == '__main__':
    #Load image
    #Detect corners also in BW2.jpg which is similar to BW.jpg but with some rotation and 
    #different illumination. What could you conclude from that?
    image = plt.imread('images/BW.jpg')   
    #Extract value channel (intensity)
    hsvImage = colors.rgb_to_hsv(image)
    valIm = hsvImage[...,2]
    # Get Harris image
    harrIm = getHarris(valIm,3)
    #Show Original Image
    plt.figure("Original Image")
    plt.imshow(image)
    plt.set_cmap("gray")
    #show harris image
    plt.figure("Harris Image")
    plt.imshow(harrIm)
    # Show final image
    plotHarris(image, harrIm, 0.4)
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    