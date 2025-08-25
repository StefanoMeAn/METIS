import os
import numpy as np
import astropy.wcs
import math
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
import urllib.request
from astropy.io import fits
import pandas as pd




    ### FITS RELATED LIBRARIES ###

def extract_timestamp(fits_data):
    """
    Extract timestamp from a .fits object.

    Parameters:
        fits_data (astropy object): fits loaded.
    Output:
        timestamp (float): time where image was taken in UTC.
    """
    # Extrat time.
    obt_beg = fits_data[0].header['OBT_BEG'] 
    obt_end = fits_data[0].header['OBT_END']
    obt_avg = (obt_beg + obt_end) / 2
    frac, whole = math.modf(obt_avg)
    frac *= 65536.
    return str(int(whole))+':'+str(int(frac))

def fits_loader(path):

    # Load .fits file
    fits_file = fits.open(path)
    # Extract timestamp 
    timestamp = extract_timestamp(fits_file)
    # Extract required headers for storing in .csv
    fits_header = fits_file[0].header
    # Extract image.
    image = fits_file[0].data
    # Close file
    fits_file.close()

    return timestamp, fits_header, image

def plot_fits(fits, waveband = None):
    """Plot a .fits file with its corresponding tags"""
    fig ,axis = plt.subplots(1, 1, figsize= (5,5))
    low, high = scoreatpercentile(fits[0].data, per=(10, 99), limit=(-np.inf ,np.inf))
    axis.imshow(fits[0].data, cmap = "gray", vmin = low, vmax = high, origin = "lower")
    axis.set_xlabel("x detector"), axis.set_ylabel("y detector")
    if fits[0].header["LEVEL"] == "L2":
        axis.set_title(f'{fits[0].header["LEVEL"]} {fits[0].header["WAVEBAND"]} \n {fits[0].header["DATE-OBS"]}')

        return fig, axis
    else:
        axis.set_title(f'{fits[0].header["LEVEL"]} {waveband} \n {fits[0].header["FILE_RAW"][:20]}')
        return fig, axis
    
def download_fits(url, filename, folder, timeout = 5):
    """Download a fits file and save it in a specific path.

    Parameters:
        url (str): url to retrieve file.
        filename (str): name of the file to be saved.
        folder (str): directory where file will be stored.    
    """

    with urllib.request.urlopen(url, timeout=timeout) as response, open(os.path.join(folder, filename), 'wb') as out_file:
            out_file.write(response.read())

def extract_headers(df, headers_new):

    """
    Save headers in each fits into a dataframe and later save it into pkl.

    Parameters: 
        df (str): name of the original pandas df.
        headers_new (astropy headers): headers from fits file.
    """

    # Load pkl file.
    dataframe = pd.read_pickle(df)
    # Concat dfs.
    dataframe = pd.concat([dataframe, pd.DataFrame([headers_new])], ignore_index= True)
    # Save dataframes.
    dataframe.to_pickle(df)


### EXTRAS ###
def sorter(list):
    """
    Sort a list with elements of the type ABCNN, where A, B, C are letters and N is a number.

    Args:
        list (list): List with LTPs or STPs.
    Returns:
        Sorted list.
    """
    return sorted(list, key=lambda x: int(x[3:]))

def get_iterable(x):
    """
    Check if a variable is a list. If it is not, covert it into a list.
    """
    if isinstance(x, list):
        return x
    else:
        return (x,)

def normalize_array(croped_img):
    """
    Normalize array given max and min value of the .fits
    """
    max_val = np.max(croped_img.flatten())
    min_val = np.min(croped_img.flatten())
    return (croped_img - min_val) / (max_val - min_val)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

def objects_to_pdfs(images, n_samples, name_file = "objects.pdf", metadata = None):
    """
    Convert images into a pdf.
    Parameters:
        images (list): images of PSFs.
        name_file: name for the new generated file.
    
    """

    # Define pdf structure.
    n_pages = math.ceil(n_samples/50)
    n_col = 5
    n_row = 10
    lim = 10

    # Set index page.
    j = 0

    # Start pdf. document.
    with PdfPages(name_file) as pdf:
        # Iterate in pages.
        for page in range(n_pages):
            # Create plot in pages.
            fig, axis = plt.subplots(n_row, n_col, figsize = (10, 20))
            plt.subplots_adjust(wspace=0.3, hspace=0.6)
            axis = axis.flatten()
            
            # Create subplots and display desired data
            for i in range(n_row*n_col):
                if (i+j)< n_samples:
                    axis[i].set_title("Sample 1", fontsize = 8)
                    axis[i].imshow(images[i])
                    axis[i].set_title(str(i))
                    axis[i].set_xticklabels([])
                    axis[i].set_yticklabels([])

            # Delete axes if no more samples were given.
                else: 
                    fig.delaxes(axis[i])
            
            j = j + n_col*n_row
            
            # Save the current figure to the PDF
            fig.suptitle("Samples of Point spread functions (PSFs)")
            pdf.savefig(fig)
            plt.close(fig)  # Close the figure to free memory