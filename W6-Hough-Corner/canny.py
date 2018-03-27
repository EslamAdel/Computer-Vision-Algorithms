import numpy as np
from scipy import signal
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from matplotlib import colors

def myCanny(image, tl, th):
    '''Canny edge detection algorithm 
    inputs : Grayscale image , tl : low threshold, th : high threshold
    output : Edge image or Canny image
    Basic steps of canny are : 
        1. Image smoothing using gaussian kernel for denoising 
        2. Getting gradient magnitude image
        3. None maxima suppression: 
            Suppression of week edges at the same direction to have a thin edge
        4. Double thresholding : 
            Suppress globally weak edges that bellow tl, and keep that above th 
        5. Edge tracking:
            track remaining pixels with values in between tl and th. Suppress them
            if they haven't a strong edge in its neighbors.
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    canny_image = image
    return canny_image



if __name__=='__main__':
    #Load Image
    image = plt.imread("images/Lines.jpg")
    #Extract value channel (intensity)
    hsvImage = colors.rgb_to_hsv(image)
    valIm = hsvImage[...,2]
    #Apply canny on the image
    cannyIm = myCanny(valIm, 50, 100)
    #Show Original Image
    plt.figure("Original Image")
    plt.imshow(valIm)
    plt.set_cmap("gray")
    #Show Canny image
    plt.figure("Canny Image")
    plt.imshow(cannyIm)
    plt.show()
            
    
        
        
                
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
