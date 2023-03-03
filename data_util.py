import numpy as np
import matplotlib.pyplot as plt


def load_bin_mapping(target_snr_per_bin=15, plot=False, url=None):
    """
    Return vornoi bin mapping. -1 is masked pixel. Unmasked pixel start
    counting from 0
    :param target_snr_per_bin: SNR per bin
    :param plot: Plot the bin mapping
    :param url: URL to the bin mapping, if None, use the default one
    :return: 2D array of bin mapping
    """
    if url is None:
        url = "./data_products/voronoi_2d_binning_KCWI_RXJ1131_icubes_mosaic_0" \
              ".1457_targetSN_{}_output.txt".format(target_snr_per_bin)
        
    bins = np.loadtxt(url)
    # bins -= 1 # unbinned pixels set to -1

    bin_mapping = np.zeros((43, 43))

    for a in bins:
        bin_mapping[int(a[1])][int(a[0])] = int(a[2]) + 1

    bin_mapping -= 1

    #voronoi_bin_mapping[voronoi_bin_mapping < 0] = np.nan

    if plot:
        cbar = plt.matshow(bin_mapping, cmap='turbo', origin='lower')
        plt.colorbar(cbar)
        plt.title('Bin mapping')
        plt.show()

    return bin_mapping


def get_kinematics_maps(array_1d, bin_mapping):
    """
    Remap the binned 1D kinematic values into 2D array using the Voronoi bin
    mapping. -1 is masked pixel
    :param array_1d: 1D array of kinematics measurements
    :param bin_mapping: 2D array of bin mapping
    :return: 2D map
    """
    # output = np.zeros((43, 43))
    output = np.zeros_like(bin_mapping)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if bin_mapping[i, j] == -1:
                continue
            
            output[i, j] = array_1d[int(bin_mapping[i, j])]

    return output
