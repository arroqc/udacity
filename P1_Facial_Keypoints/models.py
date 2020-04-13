## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Conv2dRelUBN(nn.Module):
    
    def __init__(self, c_in, c_out, k=5, stride=1, padding=0):
        super(Conv2dRelUBN, self).__init__()
        
        self.conv = nn.Conv2d(c_in, c_out, k, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(c_out)
        
    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = Conv2dRelUBN(1, 32, 3)
        self.conv2 = Conv2dRelUBN(32, 32, 3, stride=2) #32 220 220
        self.conv3 = Conv2dRelUBN(32, 64, 3)   
        self.conv4 = Conv2dRelUBN(64, 64, 3, stride=2) #64 106 106
        self.conv5 = Conv2dRelUBN(64, 128, 3) 
        self.conv6 = Conv2dRelUBN(128, 128, 3, stride=2) #128 49 49
        self.conv7 = Conv2dRelUBN(128, 256, 3) #256 23 23
        
        self.lin1 = nn.Linear(256 * 23 * 23, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 68 * 2)
     
        self.dropout = nn.Dropout(0.3)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
