from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy import signal



def harrisCorner(image):
    '''
    Compute Harris corenr using hessian matrix of the image 
    input : image 
    output : Harris operator.
    
    '''
    
    # Sobel operator is an approximation of first order derivative
    # x derivative
    sobelx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    
    #y derivative
    sobely = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])
                     
    # Get Ixx 
    # To get second derivative differentiate twice.
    Ixx = signal.convolve2d(signal.convolve2d(image, sobelx),sobelx)
    # Iyy  
    Iyy = signal.convolve2d(signal.convolve2d(image, sobely),sobely)
    # Ixy Image 
    Ixy = signal.convolve2d(signal.convolve2d(image, sobelx),sobely)
    
    #Hessian Matrix is [Ixx Ixy
    #                    Ixy Iyy]
    
    # Lets show them 
    plt.figure("Ixx")
    plt.imshow(Ixx)
    plt.set_cmap("gray")
    plt.figure("Iyy")
    plt.imshow(Iyy)
    plt.figure("Ixy")
    plt.imshow(Ixy)
    
    # Get Determinnate and trace 
    det = Ixx*Iyy - Ixy**2
    trace = Ixx + Iyy
    
    # Harris is det(H) - a * trace(H) let a = 0.2 
    H = det - 0.2 * trace
    
    # lets show Harris matrix
    plt.figure("Harris Operator")
    plt.imshow(np.abs(H))
    plt.show()
    
    #Return harris matrix
    return H
    
    
def compute_harris_response(im,sigma=3):
    '''
    Corner Detector by Harris and Stephens, instead of getting second order 
    derivative. Blure the image using gaussian and then get the first order 
    derivative 
    '''
    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    return Wdet / Wtr
    

def getHarrisPoints(harrisImage,threshold=0.1):
    ''' Extract corners > threshold where threshold is a fraction of maximum 
    corner value.
    inputs: harrisImage , threshold 
    Output : selected poits coordinates 
    '''
    #Find corners >  threshold
    selectedCorners = (harrisImage > threshold * harrisImage.max()) * 1
    
    # get coordinates of selected corners
    coords = np.array(selectedCorners.nonzero()).T
    
    return coords



def plotHarrisPoints(image,filtered_coords):
    """ Plots corners found in image. """
    plt.figure()
    plt.set_cmap('gray')
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'+', color='red')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    image = plt.imread('images/squares.jpg')
    hsv_image = col.rgb_to_hsv(image)
    vIm = hsv_image[...,2]
    H = harrisCorner(vIm)
    filtered_coords = getHarrisPoints(H, 0.4)
    plotHarrisPoints(image, filtered_coords)