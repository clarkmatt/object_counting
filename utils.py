import numpy as np
from scipy.stats import norm

def build_density_map(image, locations, sigma=3, scale=1.0):
    """
    Builds a density map given an image and the (x,y) locations of objects of interest.
    Input:
        image: the numpy array image for which we are building a density map
        locations: a list of (x,y) locations of objects in the image
        sigma: the standard deviation of the normal distribution around each object
        scale: a number between 0 and 1 which reduces the resolution of the output density map
    """

    #The density map resolution can be reduced but not increased
    assert(0 < scale <= 1)

    h, w = image.shape

    # Initialize x and y indices of each image location separately
    x_idxs = np.arange(w)
    #x_idxs = np.repeat([x_idxs], h, axis=0)
    y_idxs = np.arange(h)
    #y_idxs = np.transpose(np.repeat([y_idxs], w, axis=0))

    # Get x and y distances from each location
    x_density = []
    y_density = []
    for loc in locations:
        # Get pdf value for x & y components of each pixel
        x_comp = norm.pdf(x_idxs, int(round(loc[0])), sigma)
        y_comp = norm.pdf(y_idxs, int(round(loc[1])), sigma)

        # Repeat so dimensions are the same as original image
        x_comp = np.repeat([x_comp], h, axis=0)
        y_comp = np.transpose(np.repeat([y_comp], w, axis=0))

        # Store in list of density functions for each object
        x_density.append(x_comp)
        y_density.append(y_comp)

    # Convert from list to numpy array
    x_density = np.stack(x_density, axis=2)
    y_density = np.stack(y_density, axis=2)

    # Combine x & y components into multivariate gaussian
    density = np.sum(x_density*y_density, axis=2)
    #density = density/np.max(density)
    
    return density

def build_density_map_euclidean(image, locations):
    """
    Builds a density map using euclidean distance given an image and the (x,y)
    locations of objects of interest
    """
    h, w = img.shape

    # Initialize x and y indices of each image location separately
    x_idxs = np.arange(w)
    x_idxs = np.repeat([x_idxs], h, axis=0)
    y_idxs = np.arange(h)
    y_idxs = np.transpose(np.repeat([y_idxs], w, axis=0))

    # Get x and y distances from each location
    x_dist = []
    y_dist = []
    for loc in locations:
        x_dist.append(np.abs(int(round(loc[0])) - x_idxs))
        y_dist.append(np.abs(int(round(loc[1])) - y_idxs))
    x_dist = np.stack(x_dist, axis=2)
    y_dist = np.stack(y_dist, axis=2)

    # Calculate euclidean distances, sum them together, and normalize to get density map
    dist = np.sqrt(np.power(x_dist, 2) + np.power(y_dist, 2))
    density = np.sum(dist, axis=2)
    density = density/np.max(density)
    
    return density

if __name__ == "__main__":
    # Get test image
    import skimage.io
    img = skimage.io.imread("test/test_image.png")

    # Get people locations
    import scipy.io
    label_path = '/Users/matt/Projects/tellus_robotics/datasets/ucsdpeds/vidf-cvpr/vidf1_33_000_frame_full.mat'
    mat = scipy.io.loadmat(label_path)
    locations = mat['frame'][0,0]['loc'][0,0][:,:-1]

    density_map = build_density_map(img, locations)

    import matplotlib.pyplot as plt
    plt.figure(1)
    density_plot = plt.imshow(density_map)
    plt.figure(2)
    original = plt.imshow(img)
    plt.show()

