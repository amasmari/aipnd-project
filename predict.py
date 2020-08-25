import functionsRef
import numpy as np
import PIL
import json
import torch
import torchvision
from torchvision import datasets, transforms, models

import argparse

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#parsing arguemnts
parser = argparse.ArgumentParser(description='Predct Images based on model')

input
parser.add_argument('--input_path', type = str, default = 'flowers/test/23/image_03382.jpg', help = 'your predction data directory')
parser.add_argument('--checkpoint', type = str, default = 'save_checkpoint.pth', help = 'the model checkpoint path')
parser.add_argument('--topk', type = int, default = 5, help = 'How many top reulst to represent?')
parser.add_argument('--category_names', type = str, default = "cat_to_name.json", help = 'The maping from categories to names')
parser.add_argument('--device', type = str, default = Device, help = 'GPU (cuda) vs CPU (cpu)')
arguments = parser.parse_args()

#making image tranform

transform_img =  transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

#make labels
cat_name = functionsRef.cat_to_name(arguments.category_names)

#load model
model = functionsRef.load_checkpoint(arguments.checkpoint)


#predict
Predictions = functionsRef.predict(arguments.input_path, model, arguments.device,cat_name,transform_img, arguments.topk)

#show predctions:
names = Predictions[0]
probabilities = Predictions[1]
print("classfication    :    predictions probabilities")
for i in range(arguments.topk):
    name = str(names[i])
    probability = str(probabilities[i])
    print("{}    :    {}".format(name,probability))