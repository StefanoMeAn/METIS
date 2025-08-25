import numpy as np
import pandas as pd


def crop_psf(image, size = 25):
    """
    Convert image into small size.

    Args:
        image(array2d): image of possible object.
        size (int): size reduction factor.
    
    """
    return (image[size:-size, size:-size])

def normalize_psf(image):
    """Normalize image to 0-1 range"""
    return (image - np.min(image))/(np.max(image) - np.min(image))

def find_patches(img):
    """
    Find "holes" in a given image.

    Parameters:
        img (array2d): image of PSF.

    Output:
        [X,Y] (list): x, y position of found holes.
    """

    # Define threshold.
    thrs = np.median(img)
    X = []
    Y = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] < thrs:
                X.append(i)
                Y.append(j)
    return [X,Y]


def patch_pixels_flexible(img, X, Y):
    """
    Given an x, y position of a pixel in an image, apply smoothing.

    Parameters:
        img (array2d): image of possible PSF.
        X (list): x position of pixel.
        Y (list): y position of pixel.
    
    Output:
        img (array2d): patched image.
    
    """

    h, w = img.shape

    for i, j in zip(X, Y):
        # Define patch bounds, clipping to valid image range
        i_min = max(i - 1, 0)
        i_max = min(i + 2, h)
        j_min = max(j - 1, 0)
        j_max = min(j + 2, w)

        patch = img[i_min:i_max, j_min:j_max]
        
        # Flatten and remove the center if it's in bounds
        flat = patch.flatten()

        center_i = i - i_min
        center_j = j - j_min
        center_idx = center_i * patch.shape[1] + center_j

        if 0 <= center_idx < flat.size:
            neighbors = np.delete(flat, center_idx)
        else:
            neighbors = flat

        img[i, j] = np.mean(neighbors)

    return img

def removing_patches(img):
    """
    Compute points with value less than the median and apply smoothing.
    
    Parameters: 
        img (array2d): image of possible PSF.
    """
    x, y = find_patches(img)

    return patch_pixels_flexible(img, x, y)

def detect_neighbors(img, threshold = 0.5, center = [5,5]):
    """
    Detect neighbors around main peak. 

    Parameters:
        img (array2d): image of PSF.
        threshold (int): minimum intensity to consider a neighbor.
        center (list): main peak position. 
    
    Output:
        number of detected neighbors.

    """

    x, y = center
    peak = img[x,y]*threshold

    img = img[4:-4, 4:-4]
    min_values = img[img > peak]
    return len(min_values) -1

def check_center(img, center = [5,5]):
    """
    Check if the center pixel is the corresponding peak.

    Parameters:
        img(array2d): image of PSF.
        center(list): x, y coordinate of central pixel.

    Output:
        Bool value.
    """
    x, y = center
    if img[x,y] == 1.0:
        return True
    
    else:
        return False
    

def detect_broken_image(img):
    """
    Detect if a full region belongs to a broken sample. 

    Parameters:
        img(array2d): image of PSF.
    """

    img_left = img[:30,:]
    img_right = img[30:, :]
    mean_l = np.mean(img_left)
    mean_r = np.mean(img_right)

    if mean_l>mean_r:
        ratio_1 = mean_r/mean_l
    else:
        ratio_1 = mean_l/mean_r
    
    img_up = img[:,:30]
    img_down = img[:, 30:]
    mean_u = np.mean(img_up)
    mean_d = np.mean(img_down)

    if mean_u>mean_d:
        ratio_2 = mean_d/mean_u
    else:
        ratio_2 = mean_u/mean_d
    
    return np.min([ratio_1, ratio_2])


def obtain_interesting_objects(dataframe, croping_size = 25, center_pos = [5,5],
                               thrs_neig = 0.8, BROKEN_RATION = 0.6, STD_VALUE = 2000):
    
    #### NORMALIZATION AND CLEANING ####
    # Apply cropping. 
    dataframe["img_norm"] = dataframe["REGION"].apply(lambda x: crop_psf(x, size = croping_size))
    # Apply normalization.
    dataframe["img_norm"] = dataframe["img_norm"].apply(lambda x: normalize_psf(x))
    # Apply patching.
    dataframe["img_norm"] = dataframe["img_norm"].apply(lambda x: removing_patches(x))
    # Apply normalization.
    dataframe["img_norm"] = dataframe["img_norm"].apply(lambda x: normalize_psf(x))

    #### FEATURES FOR CLASSIFICATION ####
    # Detect number of neighbors.
    dataframe["n_neighbors"] = dataframe["img_norm"].apply(lambda x: detect_neighbors(x, threshold=thrs_neig, center = center_pos))
    # Detect if main pixel is True.
    dataframe["main_pixel"] = dataframe["img_norm"].apply(lambda x: check_center(x, center=center_pos))
    # Detect if image is broken.
    dataframe["broken_image_ratio"] = dataframe["REGION"].apply(lambda x: detect_broken_image(x))
    # Compute std of full image.
    dataframe["std_r"] = dataframe["REGION"].apply(lambda x: np.std(x))

    #### CLASSIFICATION ###

    # Take objects with at least one neighbor.
    dataframe = dataframe[dataframe["n_neighbors"]!=0]
    # Take objects whose main peak coincides with the detected one.
    dataframe = dataframe[dataframe["main_pixel"] == True]
    # Discard broken images.
    dataframe = dataframe[dataframe["broken_image_ratio"] > BROKEN_RATION]
    # Discard difussive images.
    dataframe = dataframe[dataframe["std_r"] < STD_VALUE]

    return dataframe