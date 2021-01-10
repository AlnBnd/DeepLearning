from torch import nn
from L2Norm import *

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3,  padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        
    def forward(self,image):
        out = self.conv_1(image)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        feature_map_4 = out

        return feature_map_4
        
       
class AuxiliaryConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = VGG()
        self.norm = L2Norm2d(20)
        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU()
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU()
        )
        self.conv_8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride = 2, padding=1),
            nn.ReLU()
        )
        self.conv_9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride = 2, padding=1),
            nn.ReLU()
        )
        self.conv_10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU()
        )
        self.conv_11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
        )   
    
    def forward(self, x):
        feature_maps = []
        out = self.features(x)
        feature_maps.append(self.norm(out))
        
        out = self.conv_5(out)
        feature_maps.append(out)

        out = self.conv_6(out)
        feature_maps.append(out)
        
        out = self.conv_7(out)
        feature_maps.append(out)

        out = self.conv_8(out)
        feature_maps.append(out)
        
        out = self.conv_9(out)
        feature_maps.append(out)

        out = self.conv_10(out)
        feature_maps.append(out)

        out = self.conv_11(out)
        feature_maps.append(out)

        return feature_maps
