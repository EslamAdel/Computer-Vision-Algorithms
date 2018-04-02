import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy.matlib

def getFeatureVector(image):
    '''
    Extract feature vector which is the color space of the image. In RGB color 
    space we have three channels that holds color information, feature space will 
    be 3D space. For hsv color space only two channels holds color informations. 
    Now we can segment based on colors using only 2D feature vector not. 
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
    plt.scatter(feature_vector[0], feature_vector[1])
    return feature_vector

def getInitialMean(feature_vector, not_visited_idxs):
    idx = int(np.round(len(not_visited_idxs) * np.random.rand()))
    if idx >= len(not_visited_idxs):
        idx -= 1
    return feature_vector[:,int(not_visited_idxs[idx])]
    

def clusterImage(image, out_vector, clusters):
    m, n = image.shape[0:2]
    hsv_image = colors.rgb_to_hsv(image)
    feature_vector = np.zeros((2, m*n))
    i = 0
    for c in clusters:
        s = np.where(out_vector == i)
#        col = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
        
        for ii in s[0]:
            feature_vector[:,ii] = c
        i += 1
    hsv_image[...,0:2] = np.reshape(feature_vector.T,(m,n,2)) 
    segmented_image = colors.hsv_to_rgb(hsv_image)
    return segmented_image
        
    

def meanShift(image, bandwidth):
    feature_vector = getFeatureVector(image)
    num_points = feature_vector.shape[1]
    visited_points = np.zeros(num_points)
    threshold = 0.05*bandwidth
    clusters = []
    num_clusters = -1
    not_visited = num_points
    not_visited_Idxs = np.arange(num_points)
    out_vector = -1*np.ones(num_points)
    while not_visited:
        new_mean = getInitialMean(feature_vector, not_visited_Idxs)
        this_cluster_points = np.zeros(num_points)
        while True:
            dist_to_all = np.sqrt(np.sum((feature_vector.T-new_mean)**2,1)).T
            in_range_points_idxs = np.where(dist_to_all < bandwidth)
            visited_points[in_range_points_idxs[0]] = 1
            this_cluster_points[in_range_points_idxs[0]] = 1
            old_mean = new_mean
            new_mean = np.sum(feature_vector[:,in_range_points_idxs[0]],1)/in_range_points_idxs[0].shape[0]
            if np.isnan(new_mean[0]) or np.isnan(new_mean[1]):
                continue
            if np.sqrt(np.sum((new_mean - old_mean)**2)) < threshold:
                merge_with = -1
                for i in range(num_clusters+1):
                    dist = np.sqrt(np.sum((new_mean- clusters[i])**2))
                    if dist < 0.5 * bandwidth:
                        merge_with = i
                        break
                if merge_with >= 0:
                    clusters[merge_with] = 0.5*(new_mean + clusters[merge_with])
                    out_vector[np.where(this_cluster_points == 1)] = merge_with
                else:
                    num_clusters += 1
                    clusters.append(new_mean)
                    out_vector[np.where(this_cluster_points == 1)] = num_clusters                    
                break
        not_visited_Idxs = np.array(np.where(visited_points == 0)).T
        not_visited = not_visited_Idxs.shape[0]
    segmented_image = clusterImage(image, out_vector, clusters)
    plt.figure("Segmented Image")
    plt.imshow(segmented_image)
    plt.show()
    return num_clusters
    
    
if __name__ == '__main__':
    image = plt.imread('images/seg3.png')
    num_clusters = meanShift(image, 0.2) + 1

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


