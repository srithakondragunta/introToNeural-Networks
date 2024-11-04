import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.
    
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """  
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    custom_transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if (training):
        train_set=datasets.FashionMNIST('./data',train=True,download=True,transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
        return loader
    else:
        test_set=datasets.FashionMNIST('./data', train=False,transform=custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
        return loader



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model




def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    #first for-loop
    for epoch in range(T):
        accuracy = 0
        totaLoss = 0.0
        #second for-loop
        for i, data in enumerate(train_loader,0):
            images,labels = data
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            totaLoss += loss.item() * len(images)
            _, predicted = torch.max(outputs.data, 1)   
            accuracy += (predicted==labels).sum().item() 
        accuracyPerc = 100. * accuracy / len(train_loader.dataset)
        avg_loss = totaLoss/ len(train_loader.dataset)
            
        print(f"Train Epoch: {epoch} Accuracy: {accuracy}/{len(train_loader.dataset)}({accuracyPerc:.2f}%) Loss: {avg_loss:.3f}")
       



    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    totaLoss = 0
    accuracy = 0
    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data)
            loss = criterion(output,labels)
            totaLoss += loss.item() * len(data)
            _, predicted = torch.max(output.data, 1)   
            accuracy += (predicted==labels).sum().item()
    accuracyPerc = 100. * accuracy / len(test_loader.dataset)
    avg_loss = totaLoss/ len(test_loader.dataset)  
    if(show_loss):
        print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracyPerc:.2f}%")     
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    image = test_images[index]
    logits = model(image)
    prob = F.softmax(logits, dim=1)
    top3P,top3I = prob.topk(3,dim=1)
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    for i in range(3):
        print(f"{class_names[top3I[0][i]]}: {top3P[0][i].item() * 100:.2f}%")

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    criterion = nn.CrossEntropLoss()
    '''
    train_loader = get_data_loader()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader,criterion,5)
    evaluate_model(model, train_loader, criterion, show_loss = True)
    test_images = next(iter(train_loader))[0]
    predict_label(model,test_images,1)
   