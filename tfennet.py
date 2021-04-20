import torch.nn as nn
import torchvision.models as models
from resnet import resnet18
import torch
import math
# from ops.basic_ops import ConsensusModule


class TFEN_2(nn.Module):
    def __init__(self):
        super(TFEN_2,self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)
        self.resnet_p1=nn.Sequential(*list(self.resnet18.children())[:6])
        self.resnet_p2=nn.Sequential(*list(self.resnet18.children())[6:8])
        self.m2d = nn.AvgPool2d(7, stride=1)
        self.resnet18_3d_net = resnet18(sample_size=28, sample_duration=7, shortcut_type='A')      
        self.pretrain = torch.load('models/resnet-18-kinetics.pth')   
        self.resnet18_3d_net.load_state_dict({k.replace('module.',''):v for k,v in self.pretrain['state_dict'].items()})       
        self.resnet18_3d_net_part = torch.nn.Sequential(*list(self.resnet18_3d_net.children())[6:8])
        self.m3d = nn.AvgPool3d((2,7,7), stride=1)
        self.classifier_2 = nn.Linear(512+512,2)  
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x_re=x.view((-1,3)+x.size()[-2:])
        y1 = self.resnet_p1(x_re)
        y1_d = self.dropout(y1)
        y1_re = y1_d.view((-1, 7)+ y1_d.size()[1:]) 
        y1_tr = y1_re.transpose(1,2)
        y2 = self.resnet_p2(y1)
        y2_m = self.m2d(y2)
        y2_re = y2_m.view((-1, 7) + y2_m.size()[1:])
#         y2_mean = self.consensus(y2_re)
        y2_mean = y2_re.mean(dim=1, keepdim=True)
        y2_mean = y2_mean.squeeze(1)      
        y2_mean = self.dropout(y2_mean)
        y3 = self.resnet18_3d_net_part(y1_tr)
        y3_m = self.m3d(y3)
        y3_s = y3_m.squeeze(2)        
        y4 = torch.cat((y2_mean,y3_s),1)
        y4 = self.dropout(y4)
        y4_re = y4.view(-1,1024)
        y_final = self.classifier_2(y4_re)
        
        return y_final

class TFEN_6(nn.Module):
    def __init__(self):
        super(TFEN_6,self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)
        self.resnet_p1=nn.Sequential(*list(self.resnet18.children())[:6])
        self.resnet_p2=nn.Sequential(*list(self.resnet18.children())[6:8])
        self.m2d = nn.AvgPool2d(7, stride=1)
        self.resnet18_3d_net = resnet18(sample_size=28, sample_duration=7, shortcut_type='A')      
        self.pretrain = torch.load('models/resnet-18-kinetics.pth')   
        self.resnet18_3d_net.load_state_dict({k.replace('module.',''):v for k,v in self.pretrain['state_dict'].items()})       
        self.resnet18_3d_net_part = torch.nn.Sequential(*list(self.resnet18_3d_net.children())[6:8])
        self.m3d = nn.AvgPool3d((2,7,7), stride=1)
        self.classifier = nn.Linear(512+512,6)  
        self.dropout = nn.Dropout(0.5)
#         self.consensus = ConsensusModule()
#         self.consensus = ConsensusModule('avg')
        
    def forward(self,x):
        x_re=x.view((-1,3)+x.size()[-2:])
        y1 = self.resnet_p1(x_re)
        y1_d = self.dropout(y1)
        y1_re = y1_d.view((-1, 7)+ y1_d.size()[1:]) 
        y1_tr = y1_re.transpose(1,2)
        y2 = self.resnet_p2(y1)
        y2_m = self.m2d(y2)
        y2_re = y2_m.view((-1, 7) + y2_m.size()[1:])
#         y2_mean = self.consensus(y2_re)
        y2_mean = y2_re.mean(dim=1, keepdim=True)
#         print(y2_mean)
        y2_mean = y2_mean.squeeze(1)      
        y2_mean = self.dropout(y2_mean)
        y3 = self.resnet18_3d_net_part(y1_tr)
        y3_m = self.m3d(y3)
        y3_s = y3_m.squeeze(2)        
        y4 = torch.cat((y2_mean,y3_s),1)
        y4 = self.dropout(y4)
        y4_re = y4.view(-1,1024)
        y_final = self.classifier(y4_re)
        
        return y_final

class TFEN_1024(nn.Module):
    def __init__(self):
        super(TFEN_1024,self).__init__()
        self.resnet18 = models.resnet18(pretrained = False)
        self.resnet_p1=nn.Sequential(*list(self.resnet18.children())[:6])
        self.resnet_p2=nn.Sequential(*list(self.resnet18.children())[6:8])
        self.m2d = nn.AvgPool2d(7, stride=1)
        self.resnet18_3d_net = resnet18(sample_size=28, sample_duration=7, shortcut_type='A')          
        self.resnet18_3d_net_part = torch.nn.Sequential(*list(self.resnet18_3d_net.children())[6:8])
        self.m3d = nn.AvgPool3d((2,7,7), stride=1)
        self.classifier = nn.Sequential(
           nn.Linear(512+512 , 6),
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x_re=x.view((-1,3)+x.size()[-2:])
        y1 = self.resnet_p1(x_re)
        y1_d = self.dropout(y1)
        y1_re = y1_d.view((-1, 7)+ y1_d.size()[1:]) 
        y1_tr = y1_re.transpose(1,2)
        y2 = self.resnet_p2(y1)
        y2_m = self.m2d(y2)
        y2_re = y2_m.view((-1, 7) + y2_m.size()[1:])
        y2_mean = y2_re.mean(dim=1, keepdim=True)
        y2_mean = y2_mean.squeeze(1)      
        y2_mean = self.dropout(y2_mean)
        y3 = self.resnet18_3d_net_part(y1_tr)
        y3_m = self.m3d(y3)
        y3_s = y3_m.squeeze(2)      
        y4 = torch.cat((y2_mean,y3_s),1)
        y4_re = y4.view(-1,1024)
        return y4_re