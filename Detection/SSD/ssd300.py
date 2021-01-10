from torch import nn
from base_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SSD300(nn.Module, class_num):
    def __init__(self):
        super(SSD300, self).__init__()
        self.eux_conv = AuxiliaryConv() 

        self.class_num = class_num  
        self.num_defaults_boxes = [4, 6, 6, 6, 4, 4]
        self.in_channels = [512, 1024, 512, 256, 256, 256]
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        for indx in range(len(self.in_channels)):
            self.loc_layers.append(nn.Conv2d(self.in_channels[indx], self.num_defaults_boxes[indx] * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.in_channels[indx], self.num_defaults_boxes[indx] * self.class_num, kernel_size=3, padding=1))


    def forward(self, x):
        xx = self.eux_conv(x)
        locs = []
        confs = []
        for i, x in enumerate(xx):
            locs = self.loc_layers[i](x)
            locs = locs.permute(0, 2, 3, 1).contiguous()
            locs.append(locs.view(locs.size(0), -1, 4))
            confs = self.conf_layers[i](x)
            confs = confs.permute(0, 2, 3, 1).contiguous()
            confs.append(confs.view(locs.size(0), -1, self.class_num))
        loc_pred = torch.cat(locs, dim=1)
        conf_pred = torch.cat(confs, dim=1)

        return loc_pred, conf_pred  
