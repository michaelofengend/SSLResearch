# Created 2015, Zack Gainsforth
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from skimage.transform import resize
import os, sys
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile as tif
    # Yeah, that's stupid huh?  tifffile always returns warnings so ignore them.
import argparse

# This function combines the RGB and the luminance images to make the LRGB.
def CombineImages(lum, rgb, smooth=0, decontrasting_ratio=2):
    # Make an output image.
    q = np.copy(rgb)

    # Decontrast the RGB
    # Note the current RGB range.
    qmin = np.min(q); qmax = np.max(q); qrange=qmax-qmin; qmid=qrange/2+qmin
    # A decontrast ratio of 3 will squish the range to 1/3.
    q /= decontrasting_ratio 
    # Recenter the values so they are near the midrange.
    q += qmid - qmid/decontrasting_ratio
    
    # If there is smoothing, then apply it.
    if smooth > 0:
        q[:,:,0] = filters.gaussian(q[:,:,0], sigma=smooth)
        q[:,:,1] = filters.gaussian(q[:,:,1], sigma=smooth)
        q[:,:,2] = filters.gaussian(q[:,:,2], sigma=smooth)
        
    # Combine with the luminance image.
    q[:,:,0] *= lum
    q[:,:,1] *= lum
    q[:,:,2] *= lum
    
    #Normalize for 8-bit RGB (currently we are using floats).
    q[:,:,0] -= np.min(np.min(q[:,:,0]))
    q[:,:,1] -= np.min(np.min(q[:,:,1]))
    q[:,:,2] -= np.min(np.min(q[:,:,2]))
    q[:,:,0] /= np.max(np.max(q[:,:,0]))/255
    q[:,:,1] /= np.max(np.max(q[:,:,1]))/255
    q[:,:,2] /= np.max(np.max(q[:,:,2]))/255
    
    # Convert to 8-bit RGB and return both the raw float data and the 8-bit data.
    return q, q.astype('uint8')
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MakeLRGB produces an LRGB from a luminance and RGB image.')
    parser.add_argument('--smooth', '-s', type=float, default=1.0, help='A number >= 0.0 and gives the 1-sigma length in pixels for a gaussian smoothing function applied to the RGB image before combining it with the luminance image.  0 means no smoothing.')
    parser.add_argument('--decontrast', '-d', type=float, default=1.0, help='A number > 1.0 compresses the range of the RGB image around the center value in order to reduce it\'s effect on the final image brightness.  Values between 2 and 4 usually produce good results.')
    parser.add_argument('--output', '-o', type=str, default='noname', help='Output file name for LRGB image.  The default is built from the luminance and RGB file names.')
    parser.add_argument('Luminance', metavar='Luminance.tif', type=str, help='Luminance image in tif format')
    parser.add_argument('RGB', metavar='RGB.tif', type=str, help='RGB image in tif format')
    args = parser.parse_args()
    if args.output == 'noname':
        args.output = f'{os.path.splitext(args.Luminance)[0]}_{os.path.splitext(args.RGB)[0]}.tif'

    # # Read in the input images.
    lum = tif.imread(args.Luminance).astype('float')
    rgb = tif.imread(args.RGB).astype('float')
    
    # Often the RGB is binned relative to the luminance image.  If it is a simple integer binning then just scale it up. 
    if (lum.shape[0] != rgb.shape[0]) or (lum.shape[1] != rgb.shape[1]):
        print('Resizing RGB image to match luminance image.')
        rgb = resize(rgb, (lum.shape[0], lum.shape[1], rgb.shape[2]))

    # Make the LRGB.
    qf, qi = CombineImages(lum, rgb, smooth=args.smooth, decontrasting_ratio=args.decontrast)
    
    # Save the LRGB to disk.
    tif.imsave(args.output, qi)
    
    # Show it to the user.
    plt.imshow(qi)
    plt.show()
    
