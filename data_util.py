import numpy as np
import matplotlib.pyplot as plt


def load_bin_mapping(target_snr_per_bin=15, plot=False):
    """
    Return vornoi bin mapping. -1 is masked pixel. Unmasked pixel start counting from 0.
    """
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


def get_kinematics_maps(VD_array, bin_mapping):
    """
    Remap the kinematics measurements above into 2D array. -1 is masked pixel.
    :return: 2D velocity dispersion, uncertainty of the velocity
    dispersion, velocity, and the uncertainty of the velocity.
    """
    output = np.zeros((43, 43))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if bin_mapping[i, j] == -1:
                continue
            
            output[i, j] = VD_array[int(bin_mapping[i, j])]

    return output