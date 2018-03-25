import numpy as np
from scipy import signal
from scipy.ndimage import filters
import matplotlib.pyplot as plt

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
    #1. Image smoothing 
    sigma = 1.4
    im1 = filters.gaussian_filter(image,(sigma, sigma))
    
    # 2. Getting gradient magnitude image 
    sobelx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobely = sobelx.T
    
    Gx = signal.convolve2d(im1, sobelx, "same")
    Gy = signal.convolve2d(im1, sobely, "same")
    
    G = np.sqrt(Gx**2 + Gy**2)
    
    #3. None maxima suppression 
    # Getting gradient direction at first
    theta = np.arctan(Gy/Gy)
    # Digitalize value to be 0, 45, 90, 135
    idxs = [i for i in np.array(np.argsort(theta)).T if theta[i] < 0]
    theta[idxs] += np.pi
    bins = np.array([0, np.pi/8, 3*np.pi/8 , 5*np.pi/8, 7*np.pi/8, pi])
    dirs = np.digitize(theta, bins)%4
    # Apply none-max suppression
    edgeCoords = np.array(G.nonzero()).T
    for c in edgeCoords:
        gradDir = dirs[c]
        if gradDir == 0:
            idx = [[c[0]+1, c[1]],
                   [c[0]-1,  c[1]],
                   [c[0]+2, c[1]],
                   [c[0]-2, c[1]]]
        elif gradDir == 1:
            idx = [[c[0]+1, c[1]+1],
                   [c[0]-1,  c[1]-1],
                   [c[0]+2, c[1]+2],
                   [c[0]-2, c[1]]-2]
        elif gradDir == 2:
            idx = [[c[0], c[1]+1],
                   [c[0],  c[1]-1],
                   [c[0], c[1]+2],
                   [c[0], c[1]-2]]
        elif gradDir == 3 :
            idx = [[c[0]+1, c[1]-1],
                   [c[0]-1,  c[1]+1],
                   [c[0]+2, c[1]-2],
                   [c[0]-2, c[1]+2]]
        for i in idx:
            if G[i] > G[c]:
                G[c] = 0
                
    #4. Double Thresholding 
    remainingEdges = np.array(G.nonzero()).T
    for e in remainingEdges:
        if G[e] < tl:
            G[e] = 0
        elif G[e] > th:
            G[e] = 255
    
    #5. Edge tracking by hestrisis
    remEdges = np.array(G.nonzero()).T
    for re in remEdges:
        if re != 255:
            neighbors = remEdges[re[0]-1:re[0]+2, re[1]-1:re[2]+2].flatten()
            if np.max(neighbors) == 255:
                G[re] = 255
    
    return G



if __name__=='__main__':
    image = np.zeros((200,200))
    image[25:75,25:75] = 255
            
    
        
        
                
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
