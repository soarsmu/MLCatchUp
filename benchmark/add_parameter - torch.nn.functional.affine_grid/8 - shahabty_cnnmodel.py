import torch
from torch import nn
from torch.nn import init
from torchvision.models import resnet101
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,batch_size,input_height,input_width,seq_len):
        super(Encoder,self).__init__()
        resnet = resnet101(pretrained = True)
        self.conv1 = nn.Conv2d(seq_len, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.weight_init(self.conv1)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1,self.layer2,self.layer3,self.layer4 = resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2),(1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
            if 'conv1' in n or 'conv3' in n:
                m.dilation, m.stride = (2, 2),(1, 1)

        self.freeze_bn = True
        self.freeze_bn_affine = True
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x = self.layer1(x1)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3) 
        return x4,x3,x2,x1

    def weight_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(Encoder, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

###################################################################
class Action_Encoder(nn.Module):
    def __init__(self, in_size = 180):
        super(Action_Encoder, self).__init__()
        mid_size = 1000
                                  
        # Spatial transformer localization-network
        self.action_to_loc = nn.Sequential(
            nn.Linear(in_size,mid_size),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(True),
            nn.Linear(mid_size//2, mid_size),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(True),
            nn.Linear(mid_size//2,32),
            nn.ReLU(True),
            nn.Linear(32,3*2)
        )

        self.weight_init() 
    def forward(self, a,x): #,x4,x3,x2,x1):
        theta = self.stn(a)
        grid = F.affine_grid(theta,x)
        return F.grid_sample(x,grid)
        #grid4 = F.affine_grid(theta,x4.size())
        #x4 = F.grid_sample(x4,grid4)

        #grid3 = F.affine_grid(theta,x3.size())
        #x3 = F.grid_sample(x3,grid3)

        #grid2 = F.affine_grid(theta,x2.size())
        #x2 = F.grid_sample(x2,grid2)
        #grid1 = F.affine_grid(theta,x1.size())
        #x1 = F.grid_sample(x1,grid1)
        #return x4,x3,x2,x1

    def stn(self,x):
        theta = self.action_to_loc(x)
        theta = theta.view(-1,2,3)
        return theta

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)

class decode(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(decode, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample
        self.weight_init(self.conv)
        self.weight_init(self.bn)
    def forward(self, x):
        if self.will_ups:
            x = nn.functional.interpolate(x, 
                scale_factor=2, mode="bilinear", align_corners=True)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def weight_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
        if isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)

class Decoder(nn.Module):
    def __init__(self,batch_size,num_class = 12):
        super(Decoder,self).__init__()
        self.decode_1_1 = decode(2048,1024,upsample = False)
        self.decode_1_2 = decode(1024,1024,upsample = True)
        self.decode_2_1 = decode(1024,512,upsample = False)
        self.decode_2_2 = decode(512,512,upsample = True)

        self.decode_3_1 = decode(512,64,upsample = False)
        self.decode_3_2 = decode(64,64,upsample = True)
        self.decode_5 = decode(64,num_class,upsample = True)
    def forward(self,x4,x3,x2,x1):
        x = self.decode_1_1(x4) + x3
        x = self.decode_1_2(x)
        x = self.decode_2_1(x) + x2
        x = self.decode_2_2(x)
        x = self.decode_3_1(x) + x1
        x = self.decode_3_2(x)
        return self.decode_5(x)
####################################################################

class Model(nn.Module):
    def __init__(self,batch_size,input_height,input_width, seq_len,num_class):
        super(Model,self).__init__()
        self.save_dir = ''
        self.encoder = Encoder(batch_size,input_height,input_width,seq_len)
        self.decoder = Decoder(batch_size,num_class)
#        self.action_encoder = Action_Encoder()
#        self.speed_encoder = Speed_Encoder()

    def forward(self,x,action_future,speed_future):
#        x = self.action_encoder(a,x)
        x4,x3,x2,x1 = self.encoder(x)
        
#        x4,x3,x2,x1 = self.action_encoder(action_future,x4,x3,x2,x1)
#        s = self.speed_encoder(speed_current).reshape((-1,2048,8,16))
        return self.decoder(x4,x3,x2,x1)

    def save_model(self,exp_num,epoch_num,mean_iou,optimizer):
        self.save_dir = 'checkpoints/' + str(exp_num) + '_' + str(epoch_num) + '_' + str(mean_iou) + '.pth'
        state = {'epoch': epoch_num+1,
                 'encoder': self.encoder.state_dict(),
                 'decoder': self.decoder.state_dict(),
                 #'action_encoder': self.action_encoder.state_dict(),
                 #'speed_encoder': self.speed_encoder.state_dict(),
                 'optimizer': optimizer.state_dict(),
                }
        torch.save(state,self.save_dir)

    def load_model(self,optimizer):
        state = torch.load(self.save_dir)
        epoch = state['epoch']
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        self.action_encoder.load_state_dict(state['action_encoder'])
        #self.speed_encoder.load_state_dict(state['speed_encoder'])
        optimizer.load_state_dict(state['optimizer'])
        return epoch,optimizer


