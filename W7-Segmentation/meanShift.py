import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def getFeatureVector(image):
    '''
    Extract feature vector which is the color space of the image. In RGB color 
    space we have three channels that holds color information, feature space will 
    be 3D space. For hsv color space only two channels holds color informations. 
    Now we can segment based on colors using only 2D feature vector not. 
    '''
    #Get image dimentions
    m, n = image.shape
    #Move to hsv space
    hsv_image = colors.rgb_to_hsv(image)
    #Extract color channels only 
    color_space = hsv_image[...,0:2]
    #Reshape it in a feature vector with size 2 coordinate with n*m points (pixels)
    feature_vector = np.reshape(color_space,(n*m, 2)).T
    return feature_vector

def getInitialMean(feature_vector, not_visited_idxs):
    idx = np.round(len(not_visited_idxs) * np.random.rand())
    return feature_vector[:,not_visited_idxs[idx]]
    

def clusterImage(image, out_vector, clusters):
    m, n = image.shape
    hsv_image = colors.rgb_to_hsv(image)
    feature_vector = np.zeros(2, m*n)
    for c in range(clusters.shape[1]):
        s = np.where(out_vector == c)
        feature_vector[:,s] = clusters[:,c]
    hsv_image[:,0:2] = np.reshape(out_vector.T,(m,n,2))
    segmente_image = colors.hsv_to_rgb(hsv_image)
    return segmente_image
        
    

def meanShift(image, bandwidth):
    feature_vector = getFeatureVector(image)
    num_points = feature_vector.shape[1]
    visited_points = np.zeros(num_points)
    threshold = 0.001*bandwidth
    clusters = []
    num_clusters = 0
    not_visited = num_points
    not_visited_Idxs = np.arange(num_points)
    out_vector = np.zeros(num_points)
    while not_visited:
        new_mean = getInitialMean(feature_vector, not_visited_Idxs)
        this_cluster_points = np.zeros(num_points)
        while True:
            dist_to_all = np.sqrt(np.sum((new_mean - feature_vector)**2))
            in_range_points_idxs = np.where(dist_to_all < bandwidth)
            visited_points[in_range_points_idxs] = 1
            this_cluster_points[in_range_points_idxs] = 1
            old_mean = new_mean
            new_mean = np.mean(feature_vector[:,in_range_points_idxs],2)
            
            if np.sqrt(np.sum((new_mean - old_mean)**2)) < threshold:
                merge_with = 0
                for i in range(num_clusters):
                    dist = np.sqrt(np.sum((new_mean- clusters[:,i])**2))
                    if dist < 0.5 * bandwidth:
                        merge_with = 1
                        break
                if merge_with != 0:
                    clusters[:,merge_with] = 0.5*(new_mean + clusters[:,merge_with])
                    out_vector[this_cluster_points == 1] = merge_with
                else:
                    num_clusters += 1
                    clusters[:,num_clusters] = new_mean
                    out_vector[this_cluster_points == 1] = num_clusters
                break
        not_visited_Idxs = np.where(visited_points == 0)
        not_visited = len(not_visited_Idxs)
    segmented_image = clusterImage(image, out_vector, clusters)
    plt.figure("Segmented Image")
    plt.imshow(segmented_image)
    plt.show()
    
    
if __name__ == '__main__':
    image = plt.imread('images/seg3.png')
    meanShift(image, 0.2)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


