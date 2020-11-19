import torch
import numpy as np
import matplotlib.pyplot as plt


def RecToPolar_3(RectData):
    '''
    Implement cartesian coordinate to polar coordinate
    imput:
        the array of cartesian coordinate
    output:
        the polar coodinate
        
    '''
#     RectData=yy
    #     print(RectData.size())
    SizeOfData=RectData.size()
    if(SizeOfData[2]==3):
        # print(RectData[0:3,:])
        ListSmall=1e-16#use a small num for illegal divition
        R=torch.norm(RectData,p=2,dim=2)+ListSmall
    #         print(R)
        Phi_Value=torch.addcdiv(torch.zeros_like(R),1,RectData[:,:,2],R)
        Phi=torch.acos(Phi_Value)#利用反余弦函数求出俯仰角
        r=torch.addcmul(torch.zeros_like(R),1,R,torch.sin(Phi))+ListSmall
        Theta_Value=torch.addcdiv(torch.zeros_like(r),1,RectData[:,:,0],r)
        SignalOfNum=torch.lt(RectData[:,:,1],torch.zeros_like(Theta_Value)).double()
        Flag_Signal_Coe=(-2*SignalOfNum+1)
        Flag_Fixed_Tail=np.pi*2*SignalOfNum
        Theta=torch.acos(Theta_Value).double()*Flag_Signal_Coe+Flag_Fixed_Tail
        result=torch.cat((torch.unsqueeze(R.double(),2),torch.unsqueeze(Theta.double(),2),torch.unsqueeze(Phi.double(),2)),dim=2)
        return(result)

def RecToPolar(RectData):
    # print(RectData.type())
    defaultType = RectData.dtype

    SizeOfData = RectData.size()
    if (SizeOfData[1] == 3):
        # print(RectData[0:3,:])
        ListSmall = 1e-20  # use a small num for illegal divition
        ListSmall = torch.tensor(1e-20, dtype=defaultType)
        R = torch.norm(RectData, p=2, dim=1) + ListSmall
        Phi_Value = torch.addcdiv(torch.zeros_like(R), 1, RectData[:, 2], R)
        
        return (torch.cat((R.reshape(-1, 1), Theta.reshape(-1, 1), Phi.reshape(-1, 1)), dim=1))
    elif (SizeOfData[1] == 2):
        ListSmall = 1e-20  # use a small num for illegal divition
        R = torch.norm(RectData, p=2, dim=1) + ListSmall
        Theta_Value = torch.addcdiv(torch.zeros_like(R), 1, RectData[:, 0], R).type_as(RectData)
        SignalOfNum = torch.lt(RectData[:, 1], torch.zeros_like(Theta_Value))
        Flag_Signal_Coe = (-2 * SignalOfNum + 1)
        Flag_Signal_Coe = Flag_Signal_Coe.type_as(RectData)
        Flag_Fixed_Tail = np.pi * 2 * SignalOfNum
        Flag_Fixed_Tail = Flag_Fixed_Tail.type_as(RectData)
        Theta = torch.acos(Theta_Value) * Flag_Signal_Coe + Flag_Fixed_Tail
        return (torch.cat((R.reshape(-1, 1), Theta.reshape(-1, 1)), dim=1))
    else:
        print('woring data format')
