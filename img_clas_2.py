import torch 
import torchvision 
import matplotlib.pyplot as plt 
import numpy as np 

# the below two lines are optinal and are just there tp avoid any ssl
# related errors while dowloading the CIFAR - 10 dataset

import ssl 

ssl._create_default_https_context = ssl._create_unverified_context

# Defining plotting setting 

plt.rcParams['figure.figsize'] = 14,6 

# Initializing normalizing transform for the dataset 

normilize_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean = (0.5,0.5,0.5),
                                                                                      std = (0.5,0.5,0.5))])

# Dowloading the CIFAR10 dataset intro train and test sets 
train_dataset = torchvision.datasets.CIFAR10(
    root = './CIFAR10/train',train = True,
    transform = normilize_transform,
    download = True
)

test_dataset = torchvision.datasets.CIFAR10(
    root = './COFAR10/test', train = False,
    transform = normilize_transform,
    download = True
)

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)

#Plotting 25 images from the 1st batch  

dataiter = iter(train_loader) 
images, labels = dataiter.next() 
plt.imshow(np.transpose(torchvision.utils.make_grid( 
  images[:25], normalize=True, padding=1, nrow=5).numpy(), (1, 2, 0))) 
plt.axis('off')