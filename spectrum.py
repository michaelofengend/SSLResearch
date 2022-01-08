import sys, os
import hyperspy.api as hs
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tif
import skimage
from skimage.transform import resize
from skimage.io import imsave as imsave_tif
from numba import jit
import fileinput
import argparse
import matplotlib.pyplot as plt

def makeSpectrum(bcf, mask):

    print(f'Stack dimensions are: {EDS.data.shape}.')
    mask = skimage.io.imread(mask)
    mask = (mask / np.max(mask))
    EDS_copy = EDS.data.copy()
    
    @jit
    def domult(EDS_copy, mask):
        for i in range(len(EDS_copy)):
            for j in range(len(EDS_copy[0])):
                EDS_copy[i][j] = EDS_copy[i][j] * mask[i][j]
        return EDS_copy
    if EDS_copy.shape == mask.shape:
        EDS_copyout = domult(EDS_copy, mask)
    elif mask.shape[0] % EDS_copy.shape[0] == 0:
        BinningFactor = EDS_copy.shape[0] / mask.shape[0]
        if EDS_copy.shape[1] == mask.shape[1] * BinningFactor:
            print('Resizing Mask to match stack.')
            resized_mask = resize(mask, (EDS_copy.shape[0], EDS_copy.shape[1]))
            EDS_copyout = domult(EDS_copy, resized_mask)
    else:
        print("Mask and BCF file are not compatible")
        return 
    EDS_copyout.shape[0] = np.log(EDS_copyout.shape[0])
    #spectrum of original and masked one
    s_original = np.sum(np.sum(EDS.data, axis=0), axis=0)
    s_masked = np.sum(np.sum(EDS_copyout, axis=0), axis=0)
    #marked graph
    mask_graph = np.sum(EDS_copyout, axis=2)
    #export the spectrum numpy to csv
    E = EDS.axes_manager['Energy'].axis
    np.savetxt("spectrum_masked.csv", np.stack((E, s_masked), axis=1), delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='makeSpectrum from a mask and bcf file.')
    parser.add_argument('--exportHAADF', '-e', action='store_true', help = 'Export the HAADF file to a tif.')
    parser.add_argument('stack', metavar='stack.bcf', type=str, help = 'Stack as a bcf file.')
    parser.add_argument('mask', nargs='?', metavar='mask.tif', type=str, help = 'Mask as a tif file')
    parser.add_argument('--output', '-o', action='store', type=str, default='noname', help='Ouput file name for CSV file, the default is built from the bcf and tif file names')
    # Add optional argument -o for output file
    args = parser.parse_args()
    if args.output == 'noname':
        args.output = f'{os.path.splitext(args.stack)}_{os.path.splitext(args.mask)}.csv'
    
    HAADF, EDS = hs.load(args.stack)
    if args.exportHAADF == True:
        HAADF.save('HAADF.tif')
    
    if args.mask is not None:
        makeSpectrum(EDS, args.mask)
