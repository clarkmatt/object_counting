import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from UCSDPedsDataset import UCSDPedsDataset
from models import MCNN

if __name__ == "__main__":

    # Load dataset
    label_path = '/Users/matt/Projects/tellus_robotics/datasets/ucsdpeds/vidf-cvpr/'
    image_path = '/Users/matt/Projects/tellus_robotics/datasets/ucsdpeds/ucsdpeds/vidf/'
    train_set = UCSDPedsDataset('training', label_path=label_path, image_path=image_path)

    # Initial hyperparameters
    batch_size = 32
    learning_rate = 0.0001

    # Use DataLoader to sample data/label pairs at random
    if torch.cuda.is_available():
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
   
    # Initialize model, loss function, and optimizer
    model = MCNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Iterate through training set and return input image and ground truth density map
    for batch_idx, (img, gt_dmap) in enumerate(train_loader):

        # Make sure we are in training mode
        model.train()

        # Torch accumulates gradients so we must zero these out for each batch
        optimizer.zero_grad()

        # Place data and labels in variable to track gradients and place on the GPU if available
        img = Variable(img.float())
        gt_dmap = Variable(gt_dmap.float())
        if torch.cuda.is_available():
            img.cuda()
            pred_dmap.cuda()

        pred_map = model(img)
        import pdb
        pdb.set_trace()






