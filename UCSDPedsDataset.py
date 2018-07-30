import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
from skimage import io
import os
import numpy as np
from utils import build_density_map

class UCSDPedsDataset(Dataset):
    def __init__(self, set_type, label_path='vidf-cvpr/', image_path='ucscpeds/vidf/'):
        assert(set_type in ['training', 'valdation', 'testing'])

        # Get people locations for each labeled video frame
        label_files = [ filename for filename in os.listdir(label_path) if '_frame_full.mat' in filename ]
        label_files = np.array(sorted(label_files))
        label_data = np.hstack([ scipy.io.loadmat(os.path.join(label_path, filename))['frame']
                        for filename in label_files])
        self.label_data = np.squeeze(label_data)

        # Load labeled video frames
        image_dirs = [ label_file.replace('_frame_full.mat', '.y') for label_file in label_files ]
        image_files = np.array([ os.path.join(image_dir, image_file) for image_dir in image_dirs
                                for image_file in sorted(os.listdir(os.path.join(image_path, image_dir)))
                                if image_file[-4:]=='.png' ])
        self.image_data = np.array([ io.imread(os.path.join(image_path, image_file)) for image_file in image_files ])

        # generate ground truth density maps

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        data = self.image_data[idx]
        ped_locations = self.label_data[idx][0,0]['loc'][:,:-1]

        # build ground truth density map for given image
        ground_truth = build_density_map(data, ped_locations)

        return data, ground_truth 

if __name__=="__main__":

    label_path = '/Users/matt/Projects/tellus_robotics/datasets/ucsdpeds/vidf-cvpr/'
    image_path = '/Users/matt/Projects/tellus_robotics/datasets/ucsdpeds/ucsdpeds/vidf/'

    dataset = UCSDPedsDataset('training', label_path=label_path, image_path=image_path)
    data, ground_truth = dataset[0]

    # Plot results to verify data and ground_truth density map match
    import matplotlib.pyplot as plt
    plt.figure(1)
    density_plot = plt.imshow(ground_truth)
    plt.figure(2)
    original = plt.imshow(data)
    plt.show()

    

