import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Defining plotting settings

plt.rcParams['figure.figsize'] = 14,6

# Initilize normalization for the dataset

normalize_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean =(0.5,0.5,0.5),
                                     std = (0.5,0.5,0.5))])

# Downlowading the CIFAR10 dataset intro train and test sets

train_dataset = torchvision.datasets.CIFAR10(
    root ="./CIFAR10/train", train =True,
    transform =normalize_transform ,
    download = True)

test_dataset = torchvision.datasets.CIFAR10(
    root ="./CIFAR10/test",train = False,
    transform = normalize_transform,download = True
)

# Generating data loaders from the corresponding datasets

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

#Plotting 25 images from the 1st batch
dataiter = iter(train_loader)
images, labels = next(dataiter)
plt.imshow(np.transpose(torchvision.utils.make_grid(
  images[:25], normalize=True, padding=1, nrow=5).numpy(), (1, 2, 0)))
plt.axis('off')

classer = []

for batch_size, data in enumerate(train_loader,0):
  x ,y = data
  classer.extend(y.tolist())


# calculation the unique classes and the respective counts and plotting them

unique ,counts = np.unique(classer,return_counts = True)
names = list(test_dataset.class_to_idx.keys())
plt.bar(names,counts)
plt.xlabel('target classes')
plt.ylabel('Number of training instance')

class CNN(torch.nn.Module):
  def _init_(self):
    super()._init_()
    self.model = torch.nn.Sequential(
        # Input = 3 * 32 * 32 , Output = 32 * 32 *32
        torch.nn.Conv2d(in_channels = 3 , out_channels = 32 , kernel_size = 3, padding = 1 ),
        torch.nn.ReLU(),
        # Input = 32 x 32 x 32, Output = 32 x 16 x 16

        torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
        torch.nn.ReLU(),

        # Input = 32 * 16 * 16 ,Output = 64 * 8 * 8

        torch.nn.MaxPool2d (kernel_size = 2),

        # Input = 32 * 16 * 16 ,Output = 64 * 8 * 8

        torch.nn.Conv2d(in_channels = 64 , out_channels= 64 ,kernel_size= 3 ,padding = 1),
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(kernel_size = 2),

        torch.nn.Flatten(),
        torch.nn.Linear(64*4*4,512),
        torch.nn.Relu(),
        torch.nn.Linear(512, 10)

    )

def forward(self, x):
  return self.model(x)