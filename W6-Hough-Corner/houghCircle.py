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
    for y in range(Ny):
        for x in range(Nx):
            # Check if it is an edge pixel
            #  NB: y -> rows , x -> columns
             if image[y,x] > 0:
                 # Map edge pixel to hough space
                 for k in range(len(thetas)):
                     # Calculate space parameter
                     r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                     # Update the accumulator
                     # N.B: r has value -max to max
                     # map r to its idx 0 : 2*max
                     accumulator[int(r) + Maxdist,k] += 1
    
    return accumulator, thetas, rs


def detectLines(image,accumulator, threshold, rohs, thetas):
    ''' Extract lines with accumulator value > certain threshold
        Input : image : Original image
                accumulator array
                threshold : fraction of max value in accumulator
                rhos : radii array ( -dmax : dmax)
                thetas : theta array
                
    '''
    
    # Get maximum value in accumulator 
    maxVal = np.max(accumulator)
    
    # Now Sort ( in ascending order) accumulator to select top points
    # axis=None sort all matrix to 1D vector
    sortedAcc = np.argsort(accumulator, axis=None)
    
    # Initialzie lists of selected lines
    detectedLines = []
    
    for i in reversed(sortedAcc):
        # Get 2D idxs from 1D idx
        idr = int(i/accumulator.shape[1])
        idt = int(i%accumulator.shape[1])
        
        # Check current value relative to threshold value
        if accumulator[idr, idt] >= threshold * maxVal: 
            # Add line if value > threshold
            r = rhos[idr] 
            theta = thetas[idt]      
            #Update our list
            detectedLines.append((r,theta))     
        else:
            # No more points 
            break
    
    # Now plot detected lines in image
    plotLines(image, detectedLines)
        
def plotLines(image, lines):
    ''' Plot detected lines by detecLines method superimposed on original image
        input : image : original image
                lines : list of lines(r,theta)
    '''
    # initialize x ( width of the image)
    x = range(image.shape[1])
    
    # Figure
    fig, ax = plt.subplots()
    
    # Set figure limits in x and y 
    plt.xlim(0,image.shape[1])
    plt.ylim(image.shape[0],0)
    
    # Show image
    ax.imshow(image)
    
    # Plot lines overloaded on the image
    for line in lines:
        
        rho = line[0]
        theta = np.round(np.rad2deg(line[1]))
        if theta != 0:
            # Reverse map from hough space to image space
            y = np.round(-1 * np.cos(np.deg2rad(theta)) * x / np.sin(np.deg2rad(theta)) + rho / np.sin(np.deg2rad(theta)))            
            # Plot Line
            # TODO detect initial and final points of the line
            ax.plot(x, y, '-', linewidth=2, color='red')
        else:
            plt.axvline(rho,linewidth=2, color='red')

    
if __name__ == '__main__':
    # Load the image
    image = plt.imread('images/Lines.jpg')
    hsvImage = color.rgb_to_hsv(image)
    ValImage = hsvImage[..., 2]
    #image = misc.imresize(image,(300,400))
    # Edge detection (canny)
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
    detectLines(image, accumulator, 0.27, rhos, thetas)
    plt.show()
    
    
    
