import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def otsuThresholdB(image):
    """Otsu Binarization using between classes variance 
    the optimal threshold is the maximum between classes variance     

    Args:
        image (np.array): gray image to be binarized

    Returns:
        int : the otsu threshold using between class variance method
    """
    rows, cols = image.shape
    plt.figure()
    H = plt.hist(image.ravel(), 256)[0]
    pdf = H / (rows*cols)
    cdf = np.cumsum(pdf)
    o_thresh = 1
    maxVarB = 0
    for t in range(1, 255):
        bg = np.arange(0, t)
        obj = np.arange(t, 256)
        mBg = sum(bg*pdf[0:t])/cdf[t]
        mObj = sum(obj*pdf[t:256])/(1-cdf[t])

        varB = cdf[t] * (1-cdf[t]) * (mObj - mBg)**2
        if varB > maxVarB:
            maxVarB = varB
            o_thresh = t

    return o_thresh


def otsuThresholdW(image):
    """Otsu Binarization using within class variance 
    the optimal threshold is the minimum within class variance     

    Args:
        image (np.array): gray image to be binarized

    Returns:
        int : the otsu threshold using within class variance method
    """
    rows, cols = image.shape
    plt.figure()
    H = plt.hist(image.ravel(), 256)[0]
    pdf = H / (rows*cols)
    cdf = np.cumsum(pdf)
    o_thresh = 1
    for t in range(1, 255):
        bg = np.arange(0, t)
        obj = np.arange(t, 256)
        mBg = sum(bg*pdf[0:t])/cdf[t]
        mObj = sum(obj*pdf[t:256])/(1-cdf[t])
        varBg = sum((bg**2)*pdf[0:t])/cdf[t] - mBg**2
        varObj = sum((obj**2)*pdf[t:256])/(1-cdf[t]) - mObj**2
        varW = cdf[t] * varBg + (1-cdf[t]) * varObj
        if t == 1:
            minVarW = varW
        if varW < minVarW:
            minVarW = varW
            o_thresh = t

    return o_thresh


def binarize(gray_image, threshold):
    """Binarize the image 0 or 1

    Args:
        gray_image (np.array): the gray image
        threshold (int): the threshold of binarization

    Returns:
        np.array: Binary image
    """
    return 1 * (gray_image > threshold)


if __name__ == '__main__':
    image = plt.imread('images/MRIbrain3.jpg')
    hsvImage = colors.rgb_to_hsv(image)
    myIm = hsvImage[..., 2]
    plt.figure()
    plt.imshow(myIm)
    plt.set_cmap("gray")
    oTw = otsuThresholdW(myIm)
    oTb = otsuThresholdB(myIm)
    print("Within class threshold : ", oTw)
    print("Between classes threshold : ", oTb)
    # Binarize
    myIm = binarize(myIm, oTw)
    plt.figure()
    plt.imshow(myIm)
    plt.show()
