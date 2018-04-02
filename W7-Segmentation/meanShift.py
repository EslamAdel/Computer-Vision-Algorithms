import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def getFeatureVector(image):
    '''
    Extract feature vector which is the color space of the image. In RGB color 
    space we have three channels that holds color information, feature space will 
    be 3D space. For hsv color space only two channels holds color informations. 
    Now we can segment based on colors using only 2D feature vector. 
    '''
    #Get image dimentions
    m, n = image.shape[0:2]
    #Move to hsv space
    hsv_image = colors.rgb_to_hsv(image)
    #Extract color channels only 
    color_space = hsv_image[...,0:2]
    #Reshape it in a feature vector with size 2 coordinate with n*m points (pixels)
    #TOD Scatter 3D feature vector 
    feature_vector = np.reshape(color_space,(n*m, 2)).T
    #Let's see the feature space
    plt.figure('Feature Space')
    plt.scatter(feature_vector[0], feature_vector[1])
    #Return that space
    return feature_vector

def getInitialMean(feature_vector, not_visited_idxs):
    '''
    Get a random point point from feature space as a starting mean
    inputs: 
    feature vector: feature vector of image color space
    not visited idx : indices of points that not clustered yet
    output: 
    a random mean.
    '''
    # Get a random index
    idx = int(np.round(len(not_visited_idxs) * np.random.rand()))
    #Check boundary condition
    if idx >= len(not_visited_idxs):
        idx -= 1
    return feature_vector[:,int(not_visited_idxs[idx])]
    

def clusterImage(image, out_vector, clusters):
    '''
    Assign color to each image pixel according to its cluster center
    inputs : 
    image -> original image
    out_vector -> a vector containing the cluster of each point in color space
    cluster -> containing the color of each cluster
    '''
    #Get image dimensions
    m, n = image.shape[0:2]
    # Move to hsv color space
    hsv_image = colors.rgb_to_hsv(image)
    #Initialize feature vector
    feature_vector = np.zeros((2, m*n))
    #Iterator variable
    i = 0
    for c in clusters:
        #Extract the pixels belongs to that cluster
        s = np.where(out_vector == i)
        #Assign the color of this cluster to all pixels
        for ii in s[0]:
            feature_vector[:,ii] = c
        #Next cluster
        i += 1
    #Now lets return back to image space from feature space
    hsv_image[...,0:2] = np.reshape(feature_vector.T,(m,n,2))
    #Normalize intensity channel (It is noticed that it must be in range 0 to 1)
    hsv_image[..., 2] /= np.max(hsv_image[...,2])
    #Return to RGB color space
    segmented_image = colors.hsv_to_rgb(hsv_image)
    #Retrurn segmented image
    return segmented_image
        
    

def meanShift(image, bandwidth):
    '''
    The mean shift algorithm for uniform kernel
    Basic algorithm steps are : 
    1. Start with random point in feature space
    2. according to specific bandwidth get in range points 
    3. Mark that points as visited points and assign them to your cluster
    4. Get the new mean from your new points and check difference between it and old one
    5. if distance between old and new mean < specific threshold you must check 
       merge condition with other means.
    6. Merge if distance between this cluster mean and other < 0.5 bandwidth and 
       The new mean of both cluters will be at half distance from both cluster means
    7. Repeat untill no more unvisited points
    
    inputs : 
    image -> to be segmented or clustered
    bandwidth -> window radius of in range points
    output : segmented image and number of clusters
    '''
    #Get the feature vector from the image
    feature_vector = getFeatureVector(image)
    #Get number of points in feature space
    num_points = feature_vector.shape[1]
    # A binary vector contains zero for unvisited point and one for visited 
    # Initially all points are not visited yet
    visited_points = np.zeros(num_points)
    # Threshold of convergence it is a ratio of specified bandwidth
    threshold = 0.05*bandwidth
    # Initialize an empty list of clusters
    clusters = []
    # It holds index of current cluster and number of clusters - 1
    num_clusters = -1
    # Number of unvisited points initially all points not visited yet.
    not_visited = num_points
    #Idices of unvisited points (Initially all points noy visited yet)
    not_visited_Idxs = np.arange(num_points)
    # Cluster number of each data point (Initially no clusters so all = -1)
    out_vector = -1*np.ones(num_points)
    #Start Clustering
    while not_visited:
        # Getting a random mean 
        new_mean = getInitialMean(feature_vector, not_visited_Idxs)
        # Assign 1 for point belongs to that clusters and 0 for others
        # Initially no points belongs to that cluster
        this_cluster_points = np.zeros(num_points)
        while True:
            # Get distance between all points and that mean
            dist_to_all = np.sqrt(np.sum((feature_vector.T-new_mean)**2,1)).T
            #Select points within the bandwidth
            in_range_points_idxs = np.where(dist_to_all < bandwidth)
            # Mark that points as visited points 
            visited_points[in_range_points_idxs[0]] = 1
            # Mark them as belongs to that cluster
            this_cluster_points[in_range_points_idxs[0]] = 1
            #Store the old mean
            old_mean = new_mean
            # Get the new mean of in range points 
            new_mean = np.sum(feature_vector[:,in_range_points_idxs[0]],
                              1)/in_range_points_idxs[0].shape[0]
            #Checking if no points so mean will be nan (not a number) and break 
            if np.isnan(new_mean[0]) or np.isnan(new_mean[1]):
                break
            # Checking covergence
            if np.sqrt(np.sum((new_mean - old_mean)**2)) < threshold:
                #Merge checking with other clusters 
                merge_with = -1
                for i in range(num_clusters+1):
                    # Get distance between clusters
                    dist = np.sqrt(np.sum((new_mean- clusters[i])**2))
                    # Merge condition
                    if dist < 0.5 * bandwidth:
                        # Id of cluster that we merge with
                        merge_with = i
                        break
                if merge_with >= 0:
                    # In case of merge
                    #Get in between mean and update it to old cluster
                    clusters[merge_with] = 0.5*(new_mean + clusters[merge_with])
                    #Mark this cluster point as belongs to cluster we merge with
                    out_vector[np.where(this_cluster_points == 1)] = merge_with
                else:
                    #No merging 
                    #Make a new cluster
                    num_clusters += 1
                    # Add it to our list
                    clusters.append(new_mean)
                    #Mark points of that cluster to its id
                    out_vector[np.where(this_cluster_points == 1)] = num_clusters                    
                break
        #Get remaining points indices
        not_visited_Idxs = np.array(np.where(visited_points == 0)).T
        #Number of remaining points
        not_visited = not_visited_Idxs.shape[0]
    #Now cluster the image 
    segmented_image = clusterImage(image, out_vector, clusters)
    # Return segmented image and number of clusters
    return segmented_image, num_clusters+1
    
    
if __name__ == '__main__':
    #Loading image
    image = plt.imread('images/seg3.png')
    #Show Original Image 
    plt.figure('Original Image')
    plt.imshow(image)
    #Apply mean shift segmentation
    segmented_image, num_clusters = meanShift(image, 0.1)
    #Show segmented image
    plt.figure("Segmented Image")
    plt.imshow(segmented_image)
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


