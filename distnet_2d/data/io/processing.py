from scipy.ndimage import find_objects, binary_closing, generate_binary_structure, binary_fill_holes
import numpy as np

def _close_and_fill(labelImage, iterations = 1):
    ker = generate_binary_structure(labelImage.ndim, 2)
    tempIm = np.zeros(labelImage.shape, dtype=bool)
    tempIm2 = np.zeros(labelImage.shape, dtype=bool)
    objects = find_objects(labelImage)
    if isinstance(objects, tuple):
        objects=[objects]
    for idx, o in enumerate(objects):
        l = idx+1
        tempIm[o] = labelImage[o]==l
        binary_closing(tempIm, ker, output=tempIm2, iterations=iterations)
        tempIm.fill(False)
        binary_fill_holes(tempIm2, output=tempIm)
        tempIm2.fill(False)
        otherLabels = (labelImage>0) & (labelImage!=l)
        tempIm[otherLabels] = False
        labelImage[tempIm] = l
        tempIm.fill(False)
