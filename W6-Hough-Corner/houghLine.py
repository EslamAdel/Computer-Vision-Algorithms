import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import matplotlib.colors as color


def houghLine(image):
    ''' Basic hough line transform that builds the accumulator array
    Input : image : edge image (canny)
    Output : accumulator : the accumulator of hough space
             thetas : values of theta (-90 : 90)
             rs : values of radius (-max distance : max distance)
    '''
    # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90)) 
    #Get image dimensions
    # y for rows and x for columns 
    Ny = image.shape[0]
    Nx = image.shape[1] 
    #Max diatance is diagonal one 
    Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2))) 
    #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist) 
    # initialize accumulator array to zeros
    accumulator = np.zeros((2 * Maxdist, len(thetas))) 
    # Now Start Accumulation
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    return accumulator, thetas, rs


def detectLines(image,accumulator, threshold, rohs, thetas):
    ''' Extract lines with accumulator value > certain threshold
        Input : image : Original image
                accumulator: Hough space
                threshold : fraction of max value in accumulator
                rhos : radii array ( -dmax : dmax)
                thetas : theta array
                
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    # Get maximum value in accumulator 
    # Now Sort accumulator to select top pointss
    # Initialzie lists of selected lines
    detectedLines = []
    # Check current value relative to threshold value
    # Add line if value > threshold
    # Now plot detected lines in image
    plotLines(image, detectedLines)
        
def plotLines(image, lines):
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
    # Plot lines overloaded on the image
    # just do nothing (Delete it after adding your code)
    pass

    
if __name__ == '__main__':
    # Load the image
    image = plt.imread('images/Lines.jpg')
    # Get value Channel (intensity)
    hsvImage = color.rgb_to_hsv(image)
    ValImage = hsvImage[..., 2]
    # Detect edges using canny 
    edgeImage = feature.canny( ValImage,sigma=1.4, low_threshold=40, high_threshold=150)
    # Show original image
    plt.figure('Original Image')
    plt.imshow(image)
    plt.set_cmap('gray')
    # Show edge image
    plt.figure('Edge Image')
    plt.imshow(edgeImage)
    plt.set_cmap('gray')
    # build accumulator    
    accumulator, thetas, rhos = houghLine(edgeImage)
    # Visualize hough space
    plt.figure('Hough Space')
    plt.imshow(accumulator)
    plt.set_cmap('gray')
    # Detect and superimpose lines on original image
    detectLines(image, accumulator, 0.3, rhos, thetas)
    plt.show()
    
    
    
