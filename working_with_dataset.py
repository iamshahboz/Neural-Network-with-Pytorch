import torch
import torchvision 
from torchvision import transforms, datasets 
import   matplotlib.pyplot as plt

'''
When it comes to machine learning in general, you will gonna have two 
different things

training dataset - data with which you trained the model 

testing dataset - it will contain the data you machine never seen before
'''

# MNIST dataset is a hand drawn digits dataset (0-9) 28x28 image

train = datasets.MNIST(
                       "", train=True, download=True, 
                       transform=transforms.Compose([transforms.ToTensor()])
                       )

test = datasets.MNIST(
                       "", train=False, download=True, 
                       transform=transforms.Compose([transforms.ToTensor()])
                       )

# this is training dataset
# batch_size - means we are fitting 10 items at a time through our model
# 
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

for data in trainset:
    # print(data)
    # break 

    # x, y = data[0][0], data[1][0]
    # print(y)
    # break

    # print(data[0][0].shape)
    # break 

    #showing image
    plt.imshow(data[0][0].view(28,28))
    print(plt.show())
    break 


