import numpy as np
import matplotlib.pyplot as plt
from skimage import feature


def houghCircle(image):
    ''' Basic hough Circle transform that builds the accumulator array
    Input : image : edge image (canny)
    Output : accumulator : the accumulator of hough space (3D) space
    '''
    m, n = image.shape
    maxR = np.round((m**2 + n**2)**0.5)
    radInterval = np.arange(1,maxR)
    accumulator = np.zeros((m,n,len(radInterval)))
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    return accumulator

def detectCircles(image,accumulator, threshold):
    ''' Extract Circles with accumulator value > certain threshold
        Input : 
            image : Original image
            accumulator : Hough space (3D)
            threshold : fraction of max value in accumulator                
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    # Get maximum value in accumulator 
    # Now Sort accumulator to select top points
    # Initialzie lists of selected lines
    detectedCircles = [] 
    # Now plot detected Circles in image
    plotCircles(image, detectedCircles)
        
def plotCircles(image, Circles):
    ''' Plot detected lines by detecLines method superimposed on original image
        input : image : original image
                lines : list of lines(r,theta)
    '''  
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    # Show image
    # Plot Circles overloaded on the image
    # just do nothing (Delete it after adding your code)
    pass


    
if __name__ == '__main__':
    # Load the image
    image = plt.imread('images/coins.jpg')  
    # Edge detection (canny)
    edgeImage = feature.canny( image,sigma=1.4, low_threshold=40, high_threshold=150)    
    # Show original image
    plt.figure('Original Image')
    plt.imshow(image)
    plt.set_cmap('gray')
    # Show edge image
    plt.figure('Edge Image')
    plt.imshow(edgeImage)
    plt.set_cmap('gray')
    # build accumulator    
    accumulator = houghCircle(edgeImage)
    # Detect and superimpose lines on original image
    detectCircles(image, accumulator, 0.3)
    plt.show()
    
    
    
