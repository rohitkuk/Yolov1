import torch.nn as nn
import torch


#  First aim is to make a simple all in one network and then Optimize it.
# Aim is to close it by today 19 Aug 2021 


class YoloV1(nn.Module):
    def __init__(self, in_channels):
        super(YoloV1, self).__init__()
        # Input shape 448 * 448 * 3

        # First Layer Grid-> 7, out_channel->64, stride=2, padding = ? maxpool by 2
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,out_channels=64,kernel_size=7,stride=2, padding=3), # To Maintain the shape
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # layer 2
        self.l2 =  nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels=192,kernel_size=3,stride=1, padding=1), # To Maintain the shape
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # layer 3
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels = 192,out_channels=128,kernel_size=1,stride=1, padding=1), #Check for padding
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 128,out_channels=256,kernel_size=3,stride=1, padding=0), #Check for padding
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256,out_channels=256,kernel_size=1,stride=1, padding=1), #Check for padding
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256,out_channels=512,kernel_size=3,stride=1, padding=0), #Check for padding
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # layer 4
        self.l4 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels = 512,out_channels=256,kernel_size=1,stride=1, padding=1), #Check for padding
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256,out_channels=512,kernel_size=3,stride=1, padding=0), #Check for padding
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            #2
            nn.Conv2d(in_channels = 512,out_channels=256,kernel_size=1,stride=1, padding=1), #Check for padding
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256,out_channels=512,kernel_size=3,stride=1, padding=0), #Check for padding
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            #3
            nn.Conv2d(in_channels = 512,out_channels=256,kernel_size=1,stride=1, padding=1), #Check for padding
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256,out_channels=512,kernel_size=3,stride=1, padding=0), #Check for padding
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            #4
            nn.Conv2d(in_channels = 512,out_channels=256,kernel_size=1,stride=1, padding=1), #Check for padding
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256,out_channels=512,kernel_size=3,stride=1, padding=0), #Check for padding
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            #5  
            nn.Conv2d(in_channels = 512,out_channels=512,kernel_size=1,stride=1, padding=1), #Check for padding
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512,out_channels=1024,kernel_size=3,stride=1, padding=0), #Check for padding
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            #maxpool
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.l5 = nn.Sequential(
            # 1
            nn.Conv2d(in_channels = 1024,out_channels=512,kernel_size=1,stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512,out_channels=1024,kernel_size=3,stride=1, padding=0), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            #2
            nn.Conv2d(in_channels = 1024,out_channels=512,kernel_size=1,stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512,out_channels=1024,kernel_size=3,stride=1, padding=0), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            #3
            nn.Conv2d(in_channels = 1024,out_channels=1024,kernel_size=3,stride=1, padding=0), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            #4
            nn.Conv2d(in_channels = 1024,out_channels=1024,kernel_size=3,stride=2, padding=2),  #Check for padding
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
        )
    
        self.l6 = nn.Sequential(
            
            nn.Conv2d(in_channels = 1024,out_channels=1024,kernel_size=1,stride=1, padding=0), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 1024,out_channels=1024,kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
        )

        self.l7 = nn.Sequential(
            nn.Flatten(), # We do this to Flatten the Input to 1D.This is better and PyTorchic Way.
            nn.Linear(in_features= 1024 * 7 * 7,out_features = 4096)
        )

        self.l8 = nn.Linear(in_features=4096, out_features = 7 * 7 * 30)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        # x = x.view(x.size(0), -1) # this to flatten the dimension find more pytorchic way
        x = self.l7(x)
        x = self.l8(x)
        return(x.reshape((7,7,30)))


if __name__ =="__main__":
    model = YoloV1(in_channels=3)
    input = torch.rand((1,3,448,448))
    print(model(input).shape)
