# Computer Vision Algorithms

Implementation of basic computer vision algorithms. Implemented as a part of [SBE404 Computer vision class.](https://sbme-tutorials.github.io/2018/cv/cv.html) 

## Canny Edge Detection 

Canny multi-stage edge detector operator 

|   |   |
|---|---|
|![](images/Lines.jpg)|![](images/results/Figure_1.png)|


## Line Detection 
Detection of lines in images using hough space 

| Image | Hough Space |
|----| ----|
|![](images/results/Original_Image.png) |![](images/results/Hough_Space.png)  |

Results on real Image

|  |  |
|----| ----|
|![](images/Lines.jpg) |![](images/results/imagehoughfpng.png)  |

## Corner Detection 

### Harris Corner Detector
|  |  |
|----| ----|
|![](images/squares.jpg) |![](images/results/Figure_3.png)  |

### Fast Corner Detector
|  |  |
|----| ----|
|![](images/Lines.jpg) |![](images/results/Figure_2.png)  |

## Image Segmentation 

### Threshod based Segmentation

Otsu thresholding using both within and between class variances methods

|  |  |
|----| ----|
|![](images/MRIbrain3.jpg) |![](images/results/Figure_4.png)  |

### Region Based Segmentation

Region growing segmentation algorithm 

|  |  |
|----| ----|
|![](images/MRIbrain3.jpg) |![](images/results/Figure_5.png)  |

### Color Based Segmentation

#### Kmeans Segmentation (Clustering)

|  |  |
|----| ----|
|![](images/seg3.png) |![](images/results/segmented_image.png)  |

#### Mean Shift Segmentation (Clustering)

|  |  |
|----| ----|
|![](images/seg3.png) |![](images/results/MS_Segmented_Image.png)  |