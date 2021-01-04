import torch
from torch import nn
from torch.nn import functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_features), 
            nn.ReLU(),
            nn.Conv2d(in_features, growth_rate, kernel_size=1, stride=1), 
            nn.Dropout(p=0.2)
            )

    def forward(self, x):
        output = self.layers(x)  
        return output 


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_features, growth_rate):
        super().__init__()
        layer = []
        for i in range(num_layers):
            layer.append(DenseLayer(
                in_features + i * growth_rate, growth_rate))
        self.net = nn.Sequential(*layer)

    def forward(self, features):
        for blk in self.net:
            new_features = blk(features)
            features = torch.cat((features, new_features), dim=1)
        return features       


class TransitionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.input = nn.Sequential(
            nn.BatchNorm2d(in_features),  
            nn.ReLU(inplace = True), 
            nn.Conv2d(in_features, out_features, kernel_size = 1, stride = 1),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
        )

    def forward(self, x):
        output = self.input(x)
        return output 


class DenseNet(nn.Module):
    def __init__(self, in_channels, growth_rate, num_convs_in_blocks,
                 in_features, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, in_features, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        for i, num_convs in enumerate(num_convs_in_blocks):
            blks = DenseBlock(num_convs, in_features, growth_rate)
            self.features.add_module('denseblock%d' % (i + 1), blks)
            in_features += num_convs * growth_rate

            if i != len(num_convs_in_blocks) - 1:
                trans = TransitionLayer(in_features, in_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                in_features = in_features // 2

        self.fc = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )  

    def forward(self, x):
        features = self.features(x)
        out = self.fc(features)
        return out
