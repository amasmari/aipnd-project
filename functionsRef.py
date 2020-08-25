#imports (mostly as notebook)
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
import json
import PIL
from workspace_utils import active_session

"""
functions list:
-cat_to_name #
--make_classifier#
-make_model #
-train_network #
-test_network #
-save_checkpoint #
-load_classifier #
-load_checkpoint #
-process_image #
-predict #
Many defults here are similar to the argparse ones at train.py and predict.py; I think the functions need a defult value nontheless
"""
#the device defult for all funtions
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def cat_to_name(file='cat_to_name.json'):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name



"""
def make_classifier(n_input=25088,h_nodes=[1024,512,256],n_output=102,dropout=0.5):
-n_input: how many inputs to the classifier are giver defult=25088 ;since defult arch is vgg16
-nodes: a list of hidden nodes count defult=[5120,512]; since this is what I did in my model, but expcet the user to enter different values.
-n_output: the number of labesl expected at the end. defult = 102; labels of flowers

Returns: a classfier that has to be assigned to a neme in the code.
"""
def make_classifier(n_input=25088,h_nodes=[1024,512,256],n_output=102,dropout=0.5):
    try:
        Lst_classifier = ["nn.Sequential(OrderedDict([('A',nn.Linear(",str(n_input)+","]
        for i in range(len(h_nodes)):
            h = h_nodes[i]
            if type(h) != int: #to make sure it is a number
                print("the hidden layer must be filled with intger number for each layer")
                return None
            IR, ID, IL = str(i)+'R',str(i)+'D',str(i)+'R'
            Str_hidden = "{})),('{}', nn.ReLU()),('{}',nn.Dropout(p={})),('{}',nn.Linear({},".format(h,IR,ID,dropout,IL,h)
            Lst_classifier.append(Str_hidden)
        Lst_classifier.append("{})),('Logmax',nn.LogSoftmax(dim=1))]))".format(n_output))
        Str_classifier =''.join(Lst_classifier)
    except Exception as e:
        print(e)
        return None
    classifier = eval(Str_classifier)
    return classifier



"""
make_model(device,arch="vgg16",n_input=25088,h_nodes=[5120],n_output=102,dropout=0.5,learning_rate=0.001)
    -device: "cuda" or "cpu" defult = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    -arch: the pretrained model to do trasnferlearning defult = "vgg16"
    -n_input: how many inputs to the classifier are giver defult=25088 ;since defult arch is vgg16
    -nodes: a list of hidden nodes count defult=5120; since this is what I did in my model, but expcet the user to enter different values.
    -n_output: the number of labesl expected at the end. defult = 102; labels of flowers
    -dropout: the dropour precentage in float format defult = 0.5 as 50%
    -learning_rate: the learning rate for backpropogation. defult=0.001
Returns the model, critertion, and optimizer; these have to be assigned to variable names since they are made in-function
"""
def make_model(device=Device,arch="vgg16",h_nodes=[5120,1024,256],n_output=102,dropout=0.5,learning_rate=0.001):

    if True:
        declaring = "models."+arch.strip()+"(pretrained=True)"
        model = eval(declaring)
        for m in model.parameters():
            m.requires_grad = False
        n_input = model.fc.in_features
        classifier = make_classifier(n_input,h_nodes,n_output,dropout)
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        model.to(device)
    return model, criterion, optimizer #(would add if needed)
"""
train_network(model=model,optimizer=optimizer,criterion=criterion,device=Device,epochs=12,print_every=40,train_dataloaders=train_dataloaders,valid_dataloaders=valid_dataloaders)
-model: to model to be trained
-optimizer: the model parameters optimizer
-criterion: the loss criterion for the model
-device: "cuda" or "cpu" defult = torch.device("cuda" if torch.cuda.is_available() else "cpu")
-epochs the number of training rounds (epochs); defult = 12 becaue this was the one in the notebook
-print_every: number of propogations between each printing count
-train_dataloaders: the loader to the traning dataset
-valid_dataloaders: the loader to the validation dataset

Effect:
-no returns, just train the model in place using the training dataset while also showing valdiating results every specified number of steps

Note:the same funtions as the notebook, but defined the variavbes called within that were defined before).
"""
def train_network( model,optimizer,criterion,train_dataloaders,valid_dataloaders,device=Device,epochs=12,print_every=40):

    with active_session(): #all with active sessions replaced with if True 
        print("Start Training")
        model.train()
        steps = 0
        for e in range(epochs):
            running_loss = 0  
            for inputs, labels in train_dataloaders:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                Forward_pass = model.forward(inputs)
                loss = criterion(Forward_pass, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                #Validation info about the pass every "print_every" value
                if steps % print_every == 0:
                    model.eval()
                    accuracy = 0
                    with torch.no_grad():
                        test_loss = 0
                        for inputs, labels in valid_dataloaders:
                            inputs, labels = inputs.to(device), labels.to(device)
                            Forward_pass = model.forward(inputs)
                            test_loss += criterion(Forward_pass, labels).item()
                            ps = torch.exp(Forward_pass)
                            top_class = ps.topk(1, dim=1)[1]
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print(f"Epoch {e+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {test_loss/len(valid_dataloaders):.3f}.. "
                          f"Validation accuracy: {accuracy/len(valid_dataloaders):.3f}")
                    running_loss = 0
                    model.train()




"""
def test_network(model, test_dataloaders,device=Device):
-model: to model to be trained
-device: "cuda" or "cpu" defult = torch.device("cuda" if torch.cuda.is_available() else "cpu")
-test_dataloaders: the loader to the testing dataset
Note:the same funtions as th notebook, but defined the variavbes called within that were defined before).
Effect:
-no returns, just test the model on the test dataset using the test_dataloder and print the results while executing.
"""                       
def test_network(model, test_dataloaders,device=Device):

    model.eval()
    model.to(device)
    with torch.no_grad():
        accuracy = 0
        for images, labels in iter(test_dataloaders):
            try:
                images, labels = images.to(device), labels.to(device)
                Forward_pass  = model.forward(images)
                ps = torch.exp(Forward_pass)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            except Exception as e:
                print(e)
        print("Accuracy: {}".format(accuracy/len(test_dataloaders)))
  
    


"""
def save_checkpoint(model,train_datasets,arch="vgg16",save_name='save_checkpoint.pth',save_dir=''):
-model: the model to be stored
-arch: archtecture of the mode. defult = "vgg16" similar to the other defults
; I could not do this without the arch explecitly stated in way that is not all sting editing (which I feel could be improved to a better alterntive)
a function that store all the necesary information to rebuild the model as is
Returns:
- no returns, but it generate a save file in the directory
"""
def save_checkpoint(model,train_datasets,arch="vgg16",save_name='save_checkpoint.pth',save_dir=''):

    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'arch': arch,
                  'classifier': str(model.classifier),
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,}
    save_place = save_dir + save_name
    torch.save(checkpoint, save_place)


    
"""
def load_classifier(classifier_str):

-classifier_str: the string version of model.classifier bu str() tp be fed to the function that execute an execution ready version.
Returns:
-it return a classifier object in way similar to make_classifier function
"""
def load_classifier(classifier_str):

    if True:
        if "(((" in classifier_str:
            A = classifier_str.replace("(((","(").replace(")))",")")
        else:
            A = classifier_str
        B = A.replace("Sequential(\n","nn.Sequential(OrderedDict([(").replace("\n","),(").replace(":", "',nn.").replace("  ","'").replace(")),()","))]))")
        B = B.replace("'(","'").replace(")'","'")
        
        B = str(B)
    classifier = eval(B)
       
    return classifier 


"""
def load_checkpoint(checkpoint_path="save_checkpoint.pth"):

-checkpoint_path: the path to the checkpoint we made wehn we saved the model
Returns: a model from loading the whole model stats 
"""
def load_checkpoint(checkpoint_path="save_checkpoint.pth"):
    #after intially having issues loading the files onc CPU after GPU training, Udacity reviwer refered to this stackoverflow post for solutioins
    #https://stackoverflow.com/questions/55759311/runtimeerror-cuda-runtime-error-35-cuda-driver-version-is-insufficient-for
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    model= eval("models."+checkpoint["arch"]+"(pretrained=True)")      
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = load_classifier(checkpoint['classifier'])
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(model)
    return model


"""
def process_image(image):
    -image: the image full directory to the .../image.jpg 
    same as my notebook version
    -transform: a transform that do all the resizing, croping, and normalization
    
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
Returns:
    -the image in a numpy array format
"""

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image_pil = PIL.Image.open(image)
    #image_np= np.array(test_transform(image_pil))
    
    if image_pil.size[0] > image_pil.size[1]:
        image_pil.thumbnail((5000,256))
    else:
        image_pil.thumbnail((256,5000))
    #(0,0) at the top left
    left = (image_pil.width-224)/2
    right = 244 + left
    top =  (image_pil.height-224)/2
    bottom = 244 + top
    image_pil = image_pil.crop((left, top, right, bottom)) 
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = np.array(image_pil)/255
    image_np = (image_np - mean) / std
    image_np = image_np.transpose((2, 0, 1))
    return image_np

    '''
def predict(image_path, model, device, cat_to_name, transform,topk=5):
    Predict the class (or classes) of an image using a trained deep learning model.
    image_path: the path to the image intended to be predicted
    model: the model to be used
    topk: the number of presented top predection; defult = 5
    Note:same function as my notebook
    ''' 

def predict(image_path, model, device, cat_to_name, transform,topk=5):   
    image_np = process_image(image_path)
    inputs = torch.from_numpy(image_np)
    inputs=inputs.float()
    inputs = inputs.to(device)
    inputs = inputs.unsqueeze(0)
    model.to(device)
    Forward_pass = model.forward(inputs)
    Ps = torch.exp(Forward_pass)
    Pk,idxs = Ps.topk(topk)
    Pk = Pk.tolist()[0]
    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    names = list()
    #print(Pk)
    for idx in idxs.tolist()[0]:
        Class = idx_to_class[idx]
        name = cat_to_name[Class]
        names.append(name)
    return names, Pk
