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
    theta = np.arctan2(Gy,Gx)
    theta = 180 + (180/np.pi)*theta
    x0,y0 = np.where(((theta<22.5)+(theta>157.5)*(theta<202.5)
                       +(theta>337.5)) == True)
    x45,y45 = np.where( ((theta>22.5)*(theta<67.5)
                      +(theta>202.5)*(theta<247.5)) == True)
    x90,y90 = np.where( ((theta>67.5)*(theta<112.5)
                      +(theta>247.5)*(theta<292.5)) == True)
    x135,y135 = np.where( ((theta>112.5)*(theta<157.5)
                        +(theta>292.5)*(theta<337.5)) == True)
    # Digitalize value to be 0, 45, 90, 135
#    idxs = np.array(theta.nonzero()).T
#    for idx in idxs: 
#        if theta[idx[0], idx[1]] < 0:
#            theta[idx[0], idx[1]] += np.pi 
#    bins = np.array([0, np.pi/8, 3*np.pi/8 , 5*np.pi/8, 7*np.pi/8, np.pi])
#    dirs = np.digitize(theta, bins)%4
    # Apply none-max suppression
    theta[x0,y0] = 0
    theta[x45,y45] = 1
    theta[x90,y90] = 2
    theta[x135,y135] = 3
    dirs = theta
    edgeCoords = np.array(G.nonzero()).T
    for c in edgeCoords:
        gradDir = dirs[c[0], c[1]]
        try:
            if gradDir == 0:
                idx = [[c[0], c[1]+1],
                       [c[0],  c[1]-1],
                       [c[0], c[1]+2],
                       [c[0], c[1]-2]]
            elif gradDir == 1:
                idx = [[c[0]+1, c[1]+1],
                       [c[0]-1,  c[1]-1],
                       [c[0]+2, c[1]+2],
                       [c[0]-2, c[1]-2]]
            elif gradDir == 2:
                idx = [[c[0]+1, c[1]],
                       [c[0]-1,  c[1]],
                       [c[0]+2, c[1]],
                       [c[0]-2, c[1]]]
            elif gradDir == 3 :
                idx = [[c[0]+1, c[1]-1],
                       [c[0]-1,  c[1]+1],
                       [c[0]+2, c[1]-2],
                       [c[0]-2, c[1]+2]]
            for i in idx:
                if G[i[0],i[1]] > G[c[0],c[1]]:
                    G[c[0],c[1]] = 0
        except:
            pass
                
    #4. Double Thresholding 
    remainingEdges = np.array(G.nonzero()).T
    for e in remainingEdges:
        if G[e[0], e[1]] < tl:
            G[e[0], e[1]] = 0
        elif G[e[0], e[1]] > th:
            G[e[0], e[1]] = 255
    
    #5. Edge tracking by hestrisis
    remEdges = np.array(G.nonzero()).T
    for re in remEdges:
        if G[re[0],re[1]] != 255:
            try:
                neighbors = G[re[0]-1:re[0]+2, re[1]-1:re[1]+2].flatten()
                if np.max(neighbors) == 255:
                   G[re[0],re[1]] = 255
                else:
                   G[re[0],re[1]] = 0 
            except:
                G[re[0],re[1]] = 0
                continue    
    return G



if __name__=='__main__':
#    image = np.zeros((200,200))
#    image[25:175,25:175] = 255
    image = plt.imread("images/Lines.jpg")
    hsvImage = colors.rgb_to_hsv(image)
    valIm = hsvImage[...,2]
    cannyIm = myCanny(valIm, 75, 150)
    plt.figure()
    plt.imshow(cannyIm)
    plt.set_cmap("gray")
    plt.show()
            
    
        
        
                
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
