import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
__all__ = ['AllCNN']
  
class conv_bn(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3,stride=1,padding=1,bn=True,bn_weight_init=1.0,):
        super(conv_bn, self).__init__()
        
        self.bn=bn
        
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if self.bn:
            self.bn = nn.BatchNorm2d(c_out,track_running_stats=False)
            if bn_weight_init is not None:
                self.bn.weight.data.fill_(bn_weight_init)
                
    def forward(self, x):
        out = self.conv(x)
        if self.bn: out = self.bn(out)
        out = F.relu(out)
        return out
             
#class conv_bn(nn.Module):
#    def __init__(self, c_in, c_out, kernel_size=3,stride=1,padding=1,bn=True,):
#        super(conv_bn, self).__init__()
#        
#        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding)
#        self.bn=bn
#        if self.bn:
#            self.batchnorm2d = nn.BatchNorm2d(c_out,track_running_stats=False)
#                
#    def forward(self, x):
#        out = self.conv(x)
#        if self.bn: out = self.batchnorm2d(out)
#        out = F.relu(out)
#        return out
         
class AllCNN(nn.Module):

    def __init__(self, channels=None, weight=0.125, num_classes=10):
        super(AllCNN, self).__init__()
        channels = channels or {'cnn1': 96, 'cnn2': 96, 'cnn3': 96, 'cnn4': 192,'cnn5': 192, 'cnn6': 192, 'cnn7': 192,'cnn8': 192, 'cnn9': num_classes}
        
        self.conv1 = conv_bn(3, channels['cnn1'])
        self.conv2 = conv_bn(channels['cnn1'],channels['cnn2'])
        self.conv3 = conv_bn(channels['cnn2'],channels['cnn3'],stride=2)
        self.conv4 = conv_bn(channels['cnn3'],channels['cnn4'])
        self.conv5 = conv_bn(channels['cnn4'],channels['cnn5'])
        self.conv6 = conv_bn(channels['cnn5'],channels['cnn6'],stride=2)
        self.conv7 = conv_bn(channels['cnn6'],channels['cnn7'],padding=0)
        self.conv8 = conv_bn(channels['cnn7'],channels['cnn8'],stride=1,padding=1)
        self.conv9 = conv_bn(channels['cnn8'],channels['cnn9'],stride=1,padding=1)
        self.pool  = nn.AvgPool2d(6)
        
        self.weight = weight
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        #out = F.dropout(out,0.5)
 
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        #out = F.dropout(out,0.5)
        
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        
        out = self.pool(out)
        
        out = out.view(out.size(0), out.size(1))
       
        out = self.weight*out 
          
        return F.log_softmax(out, dim=1)


