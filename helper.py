import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
from torch import optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def train_data_loader(path, mean, std, batch_size):
    """
    Returns a DataLoader object that has been applied a transform 
    that rotates the image, crops it, flips it, and nomalizes it
    
    Input:
        path: path to the image set
        mean: list with normalization means
        std: list with normalization standard deviations    
        batch_size: size of the batch
    """

    transform = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    data = datasets.ImageFolder(path, transform=transform)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True), data


def test_data_loader(path, mean, std, batch_size):
    """
    Returns a DataLoader object that has been applied a transform 
    that crops the image and nomalizes it
    
    Input:
        path: path to the image set
        mean: list with normalization means
        std: list with normalization standard deviations    
        batch_size: size of the batch
    """
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    data = datasets.ImageFolder(path, transform=transform)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


def build_network(model_type, input_class=256, output_class=102):
    """
    Build and train a network
    Inputs:
        model_type: vgg16, densenet121, alexnet
        input_class: input size of the classifier
        output_class: output size of the classifier
    Returns:
        model, criterion, optimizer
    """
    model_options = {"vgg16":25088,
                     "densenet121":1024,
                     "alexnet":9216}

    if model_type == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_type == "densenet121":        
        model = models.densenet121(pretrained=True)
    elif model_type == "alexnet":
        model = models.alexnet(pretrained=True)
    else:
        print("No valid model found, please use 'vgg16', 'densenet121' or 'alexnet'")
	return -1
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False    
    # Classifier
    model.classifier = nn.Sequential(nn.Linear(model_options[model_type], input_class),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(input_class, output_class),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    # Train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    return model, criterion, optimizer


def eval_model(model, device, loader):
    """
    This simple function evaluates a model
    
    Inputs:
        model: model to be evaluated
        device: cpu or cuda
        loader: dataloader to use for evaluating the model
    Returns:
        loss: model loss
        accuracy: model accuracy
    """   
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                    
            loss += batch_loss.item()
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    return loss/len(loader), accuracy/len(loader)


def test_model_accuracy(model, device, testloader):
    """
    This simple function gets the model accuracy
    Inputs:
        model: model to be evaluated
        device: cpu or cuda 
        testloader: testloader to use for evaluating the model
    Returns:
        accuracy: model accuracy
    """
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)            
            outputs = model(images)                        
            # Calculate accuracy
            top_p, top_class = outputs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            correct += torch.mean(equals.type(torch.FloatTensor)).item()
            
    return correct / len(testloader)


def train_model(model, criterion, optimizer, number_epochs=10, print_every=10):
    """
    Train the model
    Inputs:
        model, criterion, optimizer: outputs from network built
        number_epochs: number of epochs
        print_every: number of epochs before printing accuracy and loss
    Return:
        trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    steps = 0
    running_loss = 0

    for epoch in range(number_epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)       
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                test_loss, accuracy = eval_model(model, device, validloader)                                
                print(f"Epoch {epoch+1}/{number_epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss:.3f}.. "
                      f"Test accuracy: {accuracy:.3f}")
                running_loss = 0
                model.train()
        return model
    

def save_model(filename, model, train_data, model_type, input_class, output_class):
    """
    Saves the model
    """
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'structure': model,
                'input_size_class': input_class,
                'output_size_class': output_class,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                filename)


def load_model(path):
    """
    Loads the model
    """
    checkpoint = torch.load(path)
    input_class = checkpoint['input_size_class']
    output_class = checkpoint['output_size_class']
    model,_,_ = build_network(input_class, output_class)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
