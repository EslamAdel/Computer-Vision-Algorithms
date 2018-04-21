import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def onClick(event):
    # Get position of selected point
    x, y = int(event.xdata), int(event.ydata)
    # Start growing from this seed with specified threshold
    outRegion = regionGrowing(valChannel,np.array([y,x]),threshold)
    # Show segmente image
    showSegmentedImage(outRegion)
    
def showSegmentedImage(region):
    # Show only segmented regions 
    segmentedImage = np.copy(myImage)
    segmentedImage[...,0] *= region.astype('uint8')
    segmentedImage[...,1] *= region.astype('uint8')
    segmentedImage[...,2] *= region.astype('uint8')
    plt.figure()
    plt.imshow(segmentedImage)    
    plt.show()

       
def regionGrowing(image, seed, threshold):
    # Initilize points stack
    points = []
    # Initialize output image
    outImage = np.zeros(image.shape)
    # Add seed point to the list 
    points.append(seed)
    # Add seed point to segmented region in output image
    outImage[seed[0],seed[1]] = 1
    # Loop till no point in the stack
    while len(points):
        # Pop a point
        p = points.pop()
        # Get its four neighbors
        neighbors = np.array([[p[0]-1,p[1]] ,
                              [p[0]+1,p[1]] ,
                              [p[0], p[1]-1],
                              [p[0], p[1]+1]])
        # Try for boundary condition
        try:
            # For all neighbors
            for i in range(4):
                # Check that if its already added to region or not 
                if not outImage[neighbors[i,0],neighbors[i,1]]:
                    # Checking similarity based on intensity value
                    if np.abs(image[neighbors[i,0],neighbors[i,1]] - 
                              image[p[0],p[1]]) <= threshold:
                        # Add point to region
                        outImage[neighbors[i,0],neighbors[i,1]] = 1
                        # Push it to the stack
                        points.append(np.array([neighbors[i,0],
                                                neighbors[i,1]]))
        except:
            continue
        
    return outImage
                


if __name__ == '__main__':
    # Read the image
    myImage = plt.imread('images/MRIbrain3.jpg')
    hsvImage = colors.rgb_to_hsv(myImage)
    # Get Intensity channel
    valChannel = hsvImage[...,2]
    # Set similarity threshold 
    threshold = 5
    # Show the original Image
    fig = plt.figure()
    plt.imshow(myImage)
    plt.set_cmap('gray')
    # Enable mouse event
    fig.canvas.mpl_connect('button_press_event', onClick)
    plt.show()
    
    
