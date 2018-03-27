import numpy as np
import matplotlib.pyplot as plt



def otsuThreshold(image):
    rows, cols =  image.shape
    plt.figure()
    H, binEdges, patches = plt.hist(image.ravel(),256)
    pdf = H /(rows*cols)
    cdf = np.cumsum(pdf)
    othresh = 1
    for t in range(1,256):
        bg = np.arange(0,t)
        obj = np.arange(t, 256)
        mBg    = sum(bg*pdf[0:t])/cdf[t]
        mObj   = sum(obj*pdf[t:256])/(1-cdf[t])
        varBg  = sum((bg**2)*pdf[0:t])/cdf[t] - mBg**2
        varObj = sum((obj**2)*pdf[t:256])/(1-cdf[t]) - mObj**2        
        varW = cdf[t] * varBg + (1-cdf[t]) * varObj
        if t == 1:
            minVarW = varW
        if varW < minVarW:
            minVarW = varW
            othresh = t

    return othresh
    
def binarize( gray_image , threshold ):
    return 1 * ( gray_image > threshold )

image = plt.imread('images/Pyramids1.jpg')
myIm = image[...,2]
plt.figure()
plt.imshow(myIm)
plt.set_cmap("gray")
oT = otsuThreshold(myIm)
#Binarize 
myIm = binarize(myIm, oT)
plt.figure()
plt.imshow(myIm)
plt.show()
    
    