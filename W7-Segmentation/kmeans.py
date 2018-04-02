import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import  misc

def kmeans(image, k, num_iterations=10):
    #1. Construnct feature space
    m, n = image.shape[0:2]
    num_points = m*n
    hsv_image = colors.rgb_to_hsv(image)
    #We will select H and S channels (color information)
    # We have 2D feature space
    feature_space = np.reshape(hsv_image[...,0:2],(num_points, 2)).T
    # Lets visualize that space 
    plt.figure('feature space')
    plt.scatter(feature_space[0], feature_space[1])
    # 2. Getting Initial centers 
    idxs = np.round(num_points * np.random.rand(k))
    #Boundary condition
    idxs[np.where(idxs >= m*n)] -= 1 
    initial_centers = np.zeros((2,k))
    for i in range(k):
        initial_centers[:,i] = feature_space[:,int(idxs[i])]
    clusters_centers = initial_centers
    # Initialize distance vector 
    distance = np.zeros((k,1))
    #cluster points determines cluster of each point in space
    cluster_points = np.zeros((num_points, 1))
    #3 - start clustering for number of iterations
    for j in range(num_iterations):
        #Cluster all points according min distance
        for l in range(num_points):
            #Get distance to all centers 
            for h in range(k):
                distance[h] = np.sqrt(np.sum((feature_space[:,l]-clusters_centers[:,h])**2))
            #Select minimum one
            cluster_points[l] = np.argmin(distance)
        # Update centers of clusters according new points 
        for c in range(k):
            # Get points associated with that cluster
            idxs = np.where(cluster_points == c)
            points = feature_space[:,idxs[0]]
            # Get its new center
            clusters_centers[:,c] = np.mean(points, 1)
            if np.isnan(clusters_centers[1,c]) or  np.isnan(clusters_centers[1,c]):
                idx =  np.round(num_points * np.random.rand())
                clusters_centers[:,c] = feature_space[:,int(idx)]
    
    # Now assign color to pixels according to its cluster
    for c in range(k):
        idxs = np.where(cluster_points == c)
        for ii in idxs[0]:
            feature_space[:,ii] = clusters_centers[:,c]
    # Return to image space 
    hs_space = np.reshape(feature_space.T, (m, n,2))
    hsv_image[...,0:2] = hs_space
    hsv_image[...,2] /= np.max(hsv_image[...,2])
    segmented_image = colors.hsv_to_rgb(hsv_image)
    return segmented_image
        


if __name__=='__main__':
    image = plt.imread('images/seg2.jpg')
    image = misc.imresize(image, (150,150))
    plt.figure('Original Image')
    plt.imshow(image)
    segmented_image = kmeans(image, 6,10)
    plt.figure('segmented image')
    plt.imshow(segmented_image)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


