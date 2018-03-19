from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy import signal



def hessianMatrix(image):
    '''
    Compute Hessina matrix of the image and visualize it 
    
    '''
    sobelx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobely = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])
                     
    # Get Ixx image
    Ixx = signal.convolve2d(signal.convolve2d(image, sobelx),sobelx)
    # Iyy Image 
    Iyy = signal.convolve2d(signal.convolve2d(image, sobely),sobely)
    # Ixy Image 
    Ixy = signal.convolve2d(signal.convolve2d(image, sobelx),sobely)
    
#    plt.figure("Ixx")
#    plt.imshow(Ixx)
#    plt.set_cmap("gray")
#    plt.figure("Iyy")
#    plt.imshow(Iyy)
#    plt.figure("Ixy")
#    plt.imshow(Ixy)
#    
#    # Get Determinnate
    det = Ixx*Iyy - Ixy**2
    trace = Ixx + Iyy
    H = det - 0.2 * trace
#    plt.figure("Harris Operator")
#    plt.imshow(H)
#    plt.show()
    return H
    
    
""" Compute the Harris corner detector response function
for each pixel in a graylevel image. """
def compute_harris_response(im,sigma=3):
    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    return Wdet / Wtr
    

def get_harris_points(harrisim,threshold=0.1):
    """ Return corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary. """
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T
#    # ...and their values
#    candidate_values = [harrisim[c[0],c[1]] for c in coords]
#    # sort candidates
#    index = np.argsort(candidate_values)
#    # store allowed point locations in array
#    allowed_locations = np.zeros(harrisim.shape)
#    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
#    # select the best points taking min_distance into account
#    filtered_coords = []
#    for i in index:
#        if allowed_locations[coords[i,0],coords[i,1]] == 1:
#            filtered_coords.append(coords[i])
#            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
#                              (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    return coords



def plot_harris_points(image,filtered_coords):
    """ Plots corners found in image. """
    plt.figure()
    plt.set_cmap('gray')
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'+', color='red')
    plt.axis('off')
    plt.show()

im = plt.imread('images/squares.jpg')
hsv_image = col.rgb_to_hsv(im)

H = hessianMatrix(hsv_image[...,2])
#harrisim = compute_harris_response(im)
filtered_coords = get_harris_points(H,0.4)
plot_harris_points(im, filtered_coords)