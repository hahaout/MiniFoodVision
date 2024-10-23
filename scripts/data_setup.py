# create dataloader for training and testing loop
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision
import os

def create_dataloaders(train_dir:str,
               test_dir:str,
               train_transforms: torchvision.transforms.Compose,
               test_transforms: torchvision.transforms.Compose,
               batch_size: int,
               num_worker: int =os.cpu_count()):
    
    # convert image from path into datsets
    train_datasets = datasets.ImageFolder(root=train_dir,
                                          transform=train_transforms,
                                          )
    test_datasets = datasets.ImageFolder(root=test_dir,
                                         transform= test_transforms)
    
    # Get classes
    classes = train_datasets.classes
    
    # convert datsets into dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_worker,
                                 pin_memory=True)
    
    return train_dataloader, test_dataloader, classes