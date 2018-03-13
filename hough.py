from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
 
def houghLine(image):
    # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90))
    
    #Get image dimensions
    Ny = image.shape[0]
    Nx = image.shape[1]
    
    #Max diatance is diagonal one 
    Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
    
    #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    accumulator = np.zeros((2 * Maxdist, len(thetas)))
    
    
    for y in range(Ny):
        for x in range(Nx):
             if image[y,x] > 0:
                 for k in range(len(thetas)):
                     r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                     accumulator[int(r) + Maxdist,k] += 1
    return accumulator, thetas, rs


def detectLines(image,accumulator, threshold, rohs, thetas):
    Nx = image.shape[1]
    Ny = image.shape[0]
    maxVal = np.max(accumulator)
    sortedAcc = np.argsort(accumulator, axis=None)
    lineIdxs = []
    #Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
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
        #plotLine(image, rho, theta)
    plotLine(image,selectedRos,selectedThetas)
        
def plotLine(image, rhos, thetas):
    Nx = image.shape[1]
    Ny = image.shape[0]
    x = range(Nx)
    fig, ax = plt.subplots()
    plt.xlim(0,Nx)
    plt.ylim(Ny,0)
    ax.imshow(image)
    for i in range(len(rhos)):
        rho = rhos[i]
        theta = thetas[i]
        y = -1*np.cos(theta) * x / np.sin(theta) + rho / np.sin(theta)
        ax.plot(x, y, '-', linewidth=1, color='green')
    plt.show()
    
    
if __name__ == '__main__':
    image = plt.imread('images/Regular-Shapes.jpg')
    channel = image[...,2]
    #    
    edgeImage = feature.canny(channel)

    #image = np.zeros((50,50))
    #image[25, 25] = 1
    #image[10, 10] = 1
    #image[10:30, 10:30] = np.eye(20)
    
    accumulator, thetas, rhos = houghLine(edgeImage)
    plt.figure('Original Image')
    plt.imshow(edgeImage)
    plt.set_cmap('gray')
    plt.figure('Hough Space')
    plt.imshow(accumulator)
    plt.set_cmap('gray')
    plt.show()
    detectLines(channel, accumulator, 0.8, rhos, thetas)
#    idx = np.argmax(accumulator)
#    rho = rhos[int(idx / accumulator.shape[1])]
#    theta = thetas[int(idx % accumulator.shape[1])]
#    plotLine(image, rho, theta)
#    print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))
    
    
    