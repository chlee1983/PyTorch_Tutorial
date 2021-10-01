from torchvision import datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import SGD, Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# importing FMNIST dataset
data_folder = '~/data/FMNIST'
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

# importing validation dataset
val_fmnist = datasets.FashionMNIST(data_folder, download=True, train=False)
val_images = val_fmnist.data
val_targets = val_fmnist.targets


# building a class that fetches the dataset, input image is divided by 255 (maximum intensity / value of a pixel)
class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float() / (255*10000)
        ''' view function is meant to reshape the tensor. We want to be agnostic about the size of a given dimension
        , use -1 notation. In below example means data will be of size (batch_size, 784)'''
        x = x.view(-1, 28 * 28)
        self.x, self.y = x, y

    """contains logic for what should be returned when ask for the ix-th data points 
       (ix will be an integer between 0 and __len__)"""

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)

    # specify the number of data points in the __len__ method (length of x)
    def __len__(self):
        return len(self.x)


def get_model():
    class neuralnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_to_hidden_layer = nn.Linear(784, 1000)
            self.hidden_layer_activation = nn.ReLU()
            self.hidden_to_output_layer = nn.Linear(1000, 10)
        def forward(self,x):
            x = self.input_to_hidden_layer(x)
            x1 = self.hidden_layer_activation(x)
            x2 = self.hidden_to_output_layer(x1)
            return x2, x1
    model = neuralnet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer