import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def getFeatureVector(image, d):
    '''
    Extract feature space according to type of feature 
    inputs:
        image : the image itself
        feature : intensity(1D), color(HS) (2D) or color(RGB)(3D)
    outputs:
        feature vector.
    '''
    m, n = image.shape[0:2]
    hsv_image = colors.rgb_to_hsv(image)
    num_points = m*n
    if d == 1:
        im_space = hsv_image[...,2]
    elif d == 2:
        im_space = hsv_image[...,0:2]
    elif d == 3:
        im_space = image
    else: 
        exit('Not supported feature')
    feature_vector = np.reshape(im_space, (num_points,d)).T
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
    

def clusterImage(image, clustering_out, clusters):
    '''
    Extract results of clustering by assigning the cluster center to all its 
    points and returning back to image space
    inputs:
        clustering_out: a 1D lookup table for each pixel cluster pair (1xnum_points)
        clusters: a lookup table for cluster feature pair (kxd) where 
        k is number of clusters and d is feature dimension 
    output: 
        segmented Image (in image domain)
    '''
    m, n = image.shape[0:2]
    clusters = np.asarray(clusters).T
    d, k = clusters.shape[0:2]
    clusterd_feature_space = np.zeros((len(clustering_out),clusters.shape[0])).T
     # Now assign values to pixels according to its cluster
    for c in range(k):
        idxs = np.where(clustering_out == c)
        for j in idxs[0]:
            clusterd_feature_space[:,j] = clusters[:,c]
    # Return to image space     
    im_space  = np.reshape(clusterd_feature_space.T, (m, n,d))
    if d == 1:
        im_space = im_space[...,0]
        segmented_image = im_space
    elif d == 2:
         hsv_image = colors.rgb_to_hsv(image)
         hsv_image[...,0:2] = im_space
         hsv_image[..., 2] /= np.max(hsv_image[...,2])
         segmented_image = colors.hsv_to_rgb(hsv_image)
    else:
        segmented_image = im_space
    return segmented_image
        
    

def meanShift(image, bandwidth, d):
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
    feature_vector = getFeatureVector(image, d)
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
            if np.isnan(new_mean[0]):
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
    bw = 0.1*np.max(image)
    segmented_image, num_clusters = meanShift(image, bw , 3)
    #Show segmented image
    plt.figure("Segmented Image")
    plt.imshow(segmented_image)
    plt.set_cmap('gray')
    plt.show()
