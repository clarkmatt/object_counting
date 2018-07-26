import sys
import os
import unittest

# Import from parent directory
parent_directory_path = os.path.realpath(os.path.join(__file__, '../../'))
sys.path.append(parent_directory_path)
from utils import build_density_map

class TestUtils(unittest.TestCase):

    def test_build_density_map(self):
        # Get test image
        import skimage.io
        img = skimage.io.imread("test/test_image.png")

        # Get people locations
        import numpy as np
        locations = np.load("test/test_locations.npy")
        
        density_map = build_density_map(img, locations)

        # View density map and original image
        #import matplotlib.pyplot as plt
        #plt.figure(1)
        #density_plot = plt.imshow(density_map)
        #plt.figure(2)
        #original = plt.imshow(img)
        #plt.show()

if __name__ == '__main__':
    unittest.main()
