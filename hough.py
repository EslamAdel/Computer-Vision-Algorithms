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
    Maxdist = int(np.round(np.sqrt(width**2 + hight ** 2)))
    
    #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    accumulator = np.zeros((2 * Maxdist, len(thetas)))
    
    
    for i in range(width):
        for j in range(hight):
             if image[i,j] > 0:
                 for k in range(len(thetas)):
                     r = i*np.cos(thetas[k]) + j * np.sin(thetas[k])
                     accumulator[int(r) + Maxdist,k] += 1
    return accumulator, thetas, rs


def detectLines(image,accumulator, threshold, rohs, thetas):
    maxVal = np.max(accumulator)
    sortedAcc = np.argsort(accumulator, axis=None)
    lineIdxs = []
    for i in reversed(sortedAcc):
        if accumulator[int(i/accumulator.shape[1]),int(i%accumulator.shape[1])] >= threshold*maxVal: 
            lineIdxs.append(i)
        else:
            break
    selectedRos = []
    selectedThetas = []
    for idx in lineIdxs:
        rho = rhos[int(idx / accumulator.shape[1])]
        selectedRos.append(rho)
        theta = thetas[int(idx % accumulator.shape[1])]
        selectedThetas.append(theta)
        plotLine(image, rho, theta)
        
def plotLine(image, rho, theta):
    Nx = image.shape[0]
    x = range(Nx)
    y = -1*np.cos(theta) * x / np.sin(theta) + rho / np.sin(theta)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.plot(x, y, '-', linewidth=5, color='green')
    
    
if __name__ == '__main__':
    image = np.zeros((50,50))
    #image[25, 25] = 1
    #image[10, 10] = 1
    image[10:30, 10:30] = np.eye(20)
    
    accumulator, thetas, rhos = houghLine(image)
    plt.figure('Original Image')
    plt.imshow(image)
    plt.set_cmap('gray')
    plt.figure('Hough Space')
    plt.imshow(accumulator)
    plt.set_cmap('gray')
    plt.show()
    detectLines(image, accumulator, 1, rhos, thetas)
#    idx = np.argmax(accumulator)
#    rho = rhos[int(idx / accumulator.shape[1])]
#    theta = thetas[int(idx % accumulator.shape[1])]
#    plotLine(image, rho, theta)
#    print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))
    
    
    