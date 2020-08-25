#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from collections import OrderedDict
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import cv2
import PIL
import json
import argparse
import functionsRef
from workspace_utils import active_session

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#parsing arguemnts
parser = argparse.ArgumentParser(description='Train model')

parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'your data directory')
parser.add_argument('--arch', type = str, default = 'vgg16', help = 'The architecture for training transfer')
parser.add_argument('--learning_rate', type = int, default = 0.001, help = 'The model learning rate')
parser.add_argument('--hidden_units', type = list, default = [1024,512,256], help = 'An intger array for the hidden units')
parser.add_argument('--epochs', type = int, default = 16, help = 'Number of training epochs')
parser.add_argument('--save_dir', type = str, default = '', help = 'directory to save files')
parser.add_argument('--save_name', type = str, default = 'save_checkpoint.pth', help = 'the save name')
parser.add_argument('--device', type = str, default = Device, help = 'the device for training gup->cuda and cpu->cpu')
arguments = parser.parse_args()



arch = arguments.arch
if arch == "vgg":
    arch = "vgg16"

#Loading data
train_dir = arguments.data_dir + '/train'
valid_dir = arguments.data_dir + '/valid'
test_dir = arguments.data_dir + '/test'
##transforms
train_transform =  transforms.Compose([transforms.RandomRotation(15),
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
test_transform =  transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
valid_transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
##datasets
train_datasets = datasets.ImageFolder(train_dir, transform = train_transform)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transform)
valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transform)
##dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32)
valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)


#Bulding the model
model, criterion, optimizer = functionsRef.make_model(arguments.device,arch,arguments.hidden_units,102,0.5,arguments.learning_rate)

#traning the model
functionsRef.train_network( model,optimizer,criterion,train_dataloaders,valid_dataloaders,arguments.device,arguments.epochs,40) 

#testing the model
functionsRef.test_network(model,test_dataloaders,arguments.device)

#saving the model
functionsRef.save_checkpoint(model,train_datasets,arch,arguments.save_name,arguments.save_dir)
