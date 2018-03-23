import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt


def myHarris(image):
    
    sobelx = np.array([[-1, 0, 1],
                      [-2 , 0 , 2],
                      [-1,  0 ,1]])
    sobely = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

    Ixx = signal.convolve2d(signal.convolve2d(image, sobelx, "same"),sobelx,"same")
    
    Iyy = signal.convolve2d(signal.convolve2d(image, sobely, "same"),sobely,"same")
    
    Ixy = signal.convolve2d(signal.convolve2d(image, sobelx, "same"),sobely,"same")
    
    plt.figure("Original Image")
    plt.set_cmap("gray")
    plt.imshow(image)
    
    plt.figure("Ixx")
    plt.imshow(Ixx)
    
    plt.figure("Iyy")
    plt.imshow(Iyy)
    
    plt.figure("Ixy")
    plt.imshow(Ixy)
    
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    
    H = det - 0.2 * trace
    
    plt.figure("Harris")
    plt.imshow(np.abs(H))
    plt.show()


    
    
image = np.zeros((200,200))
image[50:150,50:150] = 255

myHarris(image)
    

