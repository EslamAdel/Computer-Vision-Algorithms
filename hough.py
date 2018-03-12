from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
 
def houghLine(image):
    # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90))
    
    #Get image dimensions
    width = image.shape[0]
    hight = image.shape[1]
    
    #Max diatance is diagonal one 
    Maxdist = np.round(np.sqrt(width**2 + hight ** 2))
    
    #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    accumulator = np.zeros((2 * Maxdist, len(thetas)))
    
    
    for i in range(width):
        for j in range(hight):
             if image[i,j] > 0:
                 for k in range(len(thetas)):
                     r = i*np.cos(thetas[k]) + j * np.sin(thetas[k])
                     accumulator[r + Maxdist,k] += 1
    return accumulator, thetas, rs
    
    
    
if __name__ == '__main__':
    image = np.zeros((50,50))
    image[25, 25] = 1
    #image[10, 10] = 1
   #image[10:30, 10:30] = np.eye(20)
    
    accumulator, thetas, rhos = houghLine(image)
    plt.figure('Original Image')
    plt.imshow(image)
    plt.set_cmap('gray')
    plt.figure('Hough Space')
    plt.imshow(accumulator)
    plt.set_cmap('gray')
    plt.show()
    idx = np.argmax(accumulator)
    rho = rhos[int(idx / accumulator.shape[1])]
    theta = thetas[int(idx % accumulator.shape[1])]
    print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))
    
    
    