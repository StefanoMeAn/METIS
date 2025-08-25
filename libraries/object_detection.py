

import pandas as pd
import numpy as np
import math
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
from libraries.utilities import get_iterable
import itertools
import libraries.starfunctions as sf


def image_slicer(image, size, overlapping_size = 3):
    """
    Slice a 2D-numpy array into several symmetrical regions that may share borders
    with its neighbors.


    Args:
        image (array): .fits image.
        size (int): size of the proposal region.
        overlapping (int): indicate length of share region (overlapping).

    Return:
        proposal_regions (list): list with proposed regions.
        proposal_coordinates (list): list with the coordinates from the original image.

    """

    # Extract size of image.
    size_x, size_y = image.shape
    # Create grid for storing positions

   
    image_coordinates = np.zeros((size_x,size_y), dtype=object)
    size_x_vals = np.arange(size_x)
    size_y_vals = np.arange(size_y)

    for idx, valx in enumerate(size_x_vals):
        for idy, valy in enumerate(size_y_vals):
            image_coordinates[idx, idy] = [valx, valy]

    # Add padding for obtaining uniform regions.
    n_regionsx = math.ceil(size_x/size)
    n_regionsy = math.ceil(size_y/size)

    padding_x = (n_regionsx*size - size_x)/2
    padding_y = (n_regionsy*size - size_y)/2

    # Add "False" padding to the borders of the image. If padding is even, then it is symmetrical.
    image = np.pad(image, [(math.ceil(padding_x), math.floor(padding_x) + overlapping_size),
                           (math.ceil(padding_y), math.floor(padding_y) + overlapping_size)],
                           constant_values=0)
    image_coordinates = np.pad(image_coordinates, [(math.ceil(padding_x), math.floor(padding_x)+ overlapping_size),
                           (math.ceil(padding_y), math.floor(padding_y) + overlapping_size)],
                           constant_values=0)
    
    size_x, size_y = image.shape

    # Create list for storing proposals and its corresponding coordinates
    proposal_regions = []
    proposal_coordinates = []

    # Crop symmetrical regions of size sizeXsize and store them into a list.
    for i in range(int(size_x/size)):
        for j in range(int(size_y/size)):
            proposal_regions.append(image[i*size:(i+1)*size + overlapping_size, j*size:(j+1)*size+overlapping_size])
            proposal_coordinates.append(image_coordinates[i*size:(i+1)*size + overlapping_size, j*size:(j+1)*size+overlapping_size])


    return proposal_regions, proposal_coordinates

def create_coordinates(size_x, size_y):
    """
    Create coordinate matrix made of pairs that represents the position of the value in the grid.

    Parameters:
        size_x (int): size in x dimension.
        size_y (int): size in y dimension.
    
    Returns:
        grid (array): 2D grid with ij coordinates as elements.

    """

    # Create array with indexes of a given image.
    X = np.arange(size_x)
    Y = np.arange(size_y)

    # Create meshgrid
    Xmsh, Ymsh = np.meshgrid(X, Y)
    
    # Merge values
    pairs = np.stack((Xmsh, Ymsh), axis=-1)
    return pairs


def statistics_region(region, sigma = 3.0):
    """
    Create a mask for 0 values in a region and compute statistics.

    Parameters:
    
        region (2d array): region to be masked and analyzed.
        sigma (float): sigma value for statistics.
    
    Return:
        mask (2d array): mask for peak detection.
        std (float): standard deviation of region.
        median (float): median of the data.
    """
    
    # Create mask for 0-valued items in the region.
    mask = region == 0
    # Compute statistics.
    mean, median, std = sigma_clipped_stats(region[~mask], sigma=3.0)

    return mask, std, median


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def peak_detector_main(image, coordinates, coeff, full_image, lim, filename):
    """
    Detect points-of-interest in a 2d-array.

    Parameters:
        image (2d-array): image where points-of-interest may be found.
        coordinates (2d-array): real coordinates of the proposal region.
        coeff (int): coefficient for threshold in find_peaks.
    
    Return:
        points_of_interest (dataframe): dataframe with the detected points.
    """
    # Generate dataframe
    # Extract size
    pd_df = pd.DataFrame(columns = ["PEAK_VAL", "X_COORD", "Y_COORD", "PRE_LABEL", "INFO", "REGION", "FILENAME"])
    size= image[0].shape[0]

    # Open all proposed regions and read one by one.
    for n_reg, reg in enumerate(image):
    
        # Create mask and compute statistics
        masked, std, median = statistics_region(reg)
        # Start peak detection.
        points = find_peaks(reg-median, threshold=coeff*std, box_size=size/2, 
                           npeaks=1, mask=masked)
    
        # Check for obtained values
        if points:
            # Iterate all over the rows in the detected points dataframe                
                real_pos = coordinates[n_reg][int(points["y_peak"]), int(points["x_peak"])]

                if len(get_iterable(real_pos)) == 2:
                    pd_df.loc[len(pd_df)] = [points["peak_value"][0]+median, real_pos[1],
                                              real_pos[0], "object", "info", cropped_region(full_image, real_pos[1], real_pos[0], lim),
                                              filename]
                
    return pd_df    

def cropped_region(image, x_pos, y_pos, lim):
    """
    Crop a small region from a given image. If coordinates are near boundaries, generate zero padding.

    Parameters:
        image (2d-array): .fits image with data.
        x_pos (int): x-coordinate with an identified object.
        y_pos (int): y-coordinate with an identified object.
        lim (int): radius of cropped region.

    Output:
        cropped_image (2d-array): cropped region from .fits image.
    """

    # Extract size of the image.
    H, W = image.shape
    # Calculate final size of the cropped region.
    crop_size = 2 * lim + 1

    # Generate cropping boundaries.
    # Add +1 factor in the cropped region so xy coordinates are in the center of the image.
    y_min = max(y_pos - lim, 0)
    y_max = min(y_pos + lim + 1, H)
    x_min = max(x_pos - lim, 0)
    x_max = min(x_pos + lim + 1, W)

    # Generate cropped image.
    # As coordinates are in x-y cartesian plane, for converting them into indexes, flip them.
    cropped = image[y_min:y_max, x_min:x_max]

    # Check if image is in-boundaries.
    if cropped.shape[0] == crop_size and cropped.shape[1] == crop_size:
        return cropped
    
    # If not, apply zero padding by generating a new image.
    padded_crop = np.zeros((crop_size, crop_size), dtype=image.dtype)

    # Determine placement in the zero-padded array
    pad_y_start = max(lim - y_pos, 0)
    pad_y_end = pad_y_start + (y_max - y_min)
    pad_x_start = max(lim - x_pos, 0)
    pad_x_end = pad_x_start + (x_max - x_min)

    # Sum up both regions.
    padded_crop[pad_y_start:pad_y_end, pad_x_start:pad_x_end] = cropped

    return padded_crop


def crop_star_position(ltp, stp, id, df, star_df, image, lim):
    """
    Crop an area where a star is supposed to be located.

    Parameters:
        df (pandas): pandas dataframe with detected objects.
        star_df (pandas): pandas dataframe with star position.
        image (2d-array): .fits image.
        lim (int): size of the cropped image.
    
    Output:
        updated_df (pandas): updated dataframe with added star images.
    """
    # Iterate over stars.
    for ids in range(len(star_df)):
        # Create region with star.
        x = int(star_df["xsensor"].iloc[ids])
        y = int(star_df["ysensor"].iloc[ids])
        cropped = cropped_region(image, x , y, lim )
        df.loc[len(df)] = [ltp, stp, id, image[y,x], x, y, "star", "non detected", cropped ]
    
    return df


### OBJECT PRE-CLASSIFICATION ###

def star_comparison(dataframe, star_catalogue):
    """
    Check if a found object is a previously detected star by computing the euclidean distance between
    both points.
    
    Parameters:
        dataframe (table): astropy table with detected peaks.
        star_catalogue (table): astropy table with detected stars.
    
    """
    # Extract coordinates from the detected object.
    values = len(dataframe)
    for idx in range(values):

        x1 = dataframe["X_COORD"].iloc[idx]
        y1 = dataframe["Y_COORD"].iloc[idx]

        # Run for all detected stars in star_catalogue.

        for ids in range(len(star_catalogue)):
            # Extract star position.
            x2, y2 = star_catalogue["xsensor"].iloc[ids], star_catalogue["ysensor"].iloc[ids]
            # Compute euclidean distance.
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist < 10:
                dataframe["PRE_LABEL"].iloc[idx] = "star"
                dataframe["INFO"].iloc[idx] = str(star_catalogue["MAIN_ID"].iloc[ids])
        

    return dataframe, star_catalogue


def remove_similar_objects(pandas_df, threshold = 5):
    """
    Given a pandas dataframe with detected objects, remove the duplicated ones.

    Parameters:
        pandas_df (dataframe): dataframe with object position and peak value.
        threshold (int): pixel distance between two objects for being considered same object.
    Return:
        cleaned_df (dataframe): cleaned dataframe with no duplicates.
    """

    # Create an array from 0 to size of dataframe.
    elements = np.arange(len(pandas_df))
    pairs = []

    # Create non repetitive pairs of indexes.
    for x, y in itertools.combinations(elements, 2):
        pairs.append([x,y])

    # Create list for duplicated elements
    remove_indexes = []

    # Iterate all over the pairs generated
    for pair in pairs:
        px, py = pair[0], pair[1]
        # Compute euclidean distance of detected objects
        y = (pandas_df["Y_COORD"].iloc[px] - pandas_df["Y_COORD"].iloc[py])**2
        x = (pandas_df["X_COORD"].iloc[px] - pandas_df["X_COORD"].iloc[py])**2
        dist = np.sqrt(y+x)
        # Store the index object whose peak was lower.
        if dist < threshold:
            index = [px if pandas_df["PEAK_VAL"].iloc[px] < pandas_df["PEAK_VAL"].iloc[py] else py]
            # Store index.
            remove_indexes.append(pandas_df.iloc[index[0]].name)

    # Remove repeated objec with lowest peak value.
    pandas_df = pandas_df.drop(remove_indexes)

    return pandas_df    

def points_inside_fov(x, y, center, scale):

    # Metis FOV.
    radius1 = sf.METIS_fov_min/scale
    radius2 = sf.METIS_fov/scale

    outside_first = (x - center[0])**2 + (y - center[1])**2 >= radius1**2
    inside_second = (x - center[0])**2 + (y - center[1])**2 <= radius2**2

    return outside_first and inside_second

