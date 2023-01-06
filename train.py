#!/usr/bin/python3
import argparse
import json
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='Image Classifier: Training')

parser.add_argument('data_dir',
                    type=str,
                    help='Enter training data path')
parser.add_argument('--save_dir',
                    type=str,
                    default='checkpoint',
                    help='Enter path to store checkpoint (.pth) file')
parser.add_argument('--learning_rate','-lr', 
                    type=float,
                    default=0.01,
                    help='Enter learning rate (int), default value: 0.01')
parser.add_argument('--hidden_units', 
                    type=int,
                    default=512,
                    help='Enter number of hidden layer neurons (int), default value: 512')
parser.add_argument('--arch',
                    type=str,
                    default='vgg16',
                    help='Choose pretrained architecture,default value: vgg16')
parser.add_argument('--epochs',
                    default=20,
                    type=int,
                   help='Enter the number of epochs')
parser.add_argument('--gpu',
                    action='store_true',
                    help='Enter if GPU is to be used')

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir

learning_rate = args.learning_rate
epochs = args.epochs
hidden_units = args.hidden_units
arch = args.arch


if args.gpu == True:
    device = 'cuda'
else:
    device = 'cpu'

    
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

train_data =  datasets.ImageFolder(train_dir,transform = train_transforms)
valid_data =  datasets.ImageFolder(valid_dir,transform = valid_transforms)
test_data =  datasets.ImageFolder(test_dir,transform = test_transforms)

trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data,batch_size=64)
testloader = torch.utils.data.DataLoader(test_data,batch_size=64)

print('Data is transformed')

if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    
for param in model.parameters():
    param.requires_grad = False

    
print('Model is pretrained')
    
neurons = 25088

model.classifier = nn.Sequential(nn.Linear(neurons,512,bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units,102,bias=True),
                                 nn.LogSoftmax(dim=1))

model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
iteration = 10
step = 0

print('Model is training')

# Data Training
for epoch in range(epochs):
    running_train_loss = 0
    for inputs, labels in trainloader:
        step+=1
        inputs, labels = inputs.to(device),labels.to(device)
    
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)        
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        
#         checking training loss for an epoch
        if step % iteration == 0:
            valid_loss,valid_accuracy= 0,0
            model.eval()
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    valid_logps = model(images)
                    valid_loss = criterion(valid_logps,labels)
                    
                    ps = torch.exp(valid_logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    compare = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(compare.type(torch.FloatTensor)).item()
                else:
                    print(f'Epoch: {epoch+1}/{epochs},' 
                          f'Train loss: {running_train_loss/iteration:.3f},' 
                          f'Valid loss: {valid_loss/len(validloader):.3f},'
                          f'Valid accuracy: {valid_accuracy*100/len(validloader):.3f}%')      
            running_train_loss = 0
            model.train()


# Data Validation,testing
print('Model is evaluating')
model.eval()
running_test_loss, running_test_accuracy = 0,0

with torch.no_grad():
    model.eval()    
    for images,labels in testloader:
        images, labels = images.to(device), labels.to(device)

        logps = model(images)
        loss = criterion(logps, labels)
        running_test_loss += loss

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        compare = top_class == labels.view(*top_class.shape)
        running_test_accuracy += torch.mean(compare.type(torch.FloatTensor)).item()
    else:
        test_accuracy = running_test_accuracy/len(testloader)
        print('Test Accuracy: {}%'.format(test_accuracy))

print('Saving model...')
# Save model
model.class_to_idx = train_data.class_to_idx
torch.save({
            'class_to_idx': model.class_to_idx,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'classifier': model.classifier
            'hidden_units': args.hidden_units,
            'optim_state': optimizer.state_dict()
        }, 'checkpoint/' + args.save_dir + '.pth')
print('Done.')