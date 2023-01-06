#!/usr/bin/python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

parser = argparse.ArgumentParser(description='Image Classifier: Predicting')

parser.add_argument('image_path', 
                    type=str,
                    help='Enter path of the image')
parser.add_argument('--checkpoint',
                    type=str,
                    default='checkpoint',
                    help='Enter checkpoint of the model')
parser.add_argument('--top_k',
                    type=int,
                    default=3,
                    help='Enter the number of top_k')
parser.add_argument('--category_names',
                    default='cat_to_name.json',
                    type=str,
                    help='Enter file of category name')
parser.add_argument('--gpu',
                    action='store_true',
                    help='Enter if GPU is to be used')

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    for param in model.parameters(): param.requires_grad = False
    model.classifier = checkpoint['classifier']        
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    width, height = image.size
    image_open = PIL.Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_transform = transform(image_open)
    return img_transform
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file    
    model.to(device)
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()

    model.eval()
    output = model.forward(img)
        
    ps = torch.exp(output)
    top_ps, top_labels = ps.topk(topk)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return ps.topk(topk), top_labels, top_flowers


if __name__ == '__main__':
    args = parser.parse_args()

    image_path = args.image_path
    checkpoint_path = 'checkpoint/' + args.checkpoint
    top_k = args.top_k
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(checkpoint_path + '.pth')
    print('Model is loaded')
    img = process_image(image_path)
    print(f' {image_path}')
    ps = predict(image_path,model)
    top_ps, labels, top_flowers = predict(image_path, model) 
    print('Results')
    for i, j in enumerate(zip(top_flowers, top_ps)):
            print ("Rank {}:".format(i+1),"Flower: {}, likelihood: {}%".format(j[1], ceil(j[0]*100)))