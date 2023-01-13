import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
import torch.nn.functional as F
from .videoPain_seresnet18 import se_resnet_18
from .encoder import Encoder

def initialize_weights(m):
    for m in m.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            
class Encoder_token(nn.Module):
    def __init__(self):
        super(Encoder_token, self).__init__()
        self.encoder = Encoder(512, 2, 8, 64, 64,
                 512, 1024, dropout=0.1, pe_maxlen=5000)
                 
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, padded_input, input_lengths=None):
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        return encoder_padded_outputs

class Encoder_W_real(nn.Module):
    def __init__(self):
        super(Encoder_W_real, self).__init__()
        self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        #self.resnet18 = ResNet_18_real()
        self.seresnet = se_resnet_18(se=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(9)
        self.apply(initialize_weights)
        self.encoder = Encoder_token()
        #for p in self.parameters():
        #    p.requires_grad=False
        

    def forward(self, input):
        
        x = input.transpose(1,2)
        T = x.shape[2]
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        
        
        x= self.seresnet(x)
        #print("SEResNet18 output:", x.shape)
        
        x = x.view(x.size(0), x.size(1), -1)
        x = x.transpose(1,2).contiguous()
        #x = self.dropout(self.fc(x))
        #print("view and FC:", x.shape)
        Residual = x
        x = self.encoder(x)
        x += Residual
        x = self.bn(x)
        #print("Encoder and Residual:", x.shape)
        x = x.mean(dim=1)
     
        x = x.view(-1, T, 512)
        
        return x



