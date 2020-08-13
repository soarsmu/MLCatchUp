import torch
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
x=Variable(torch.FloatTensor(torch.randn(200,1)),requires_grad=False)

y=torch.sin(x)

class SinNet(torch.nn.Module):
    def __init__(self):
        super(SinNet,self).__init__()
        self.conv=torch.nn.Linear(1,10)
        self.con2=torch.nn.Linear(10,10)
        self.conv3=torch.nn.Linear(10,10)
        self.con1=torch.nn.Linear(10,1)

    def forward(self,x):
        out1=torch.nn.functional.tanh(self.conv(x))

        out2=torch.nn.functional.tanh(self.con1(out1))
        return out2

nn=SinNet()
critizirion=torch.nn.BCELoss()
optimizer=torch.optim.Adam(nn.parameters(),lr=0.01)

for i in range(2000):
    out=nn(x)
    err=critizirion(out,y)
    err.backward()
    optimizer.step()
print(err.data)
# x=Variable(torch.FloatTensor(torch.linspace(-4,4,200)),requires_grad=False)
# y=torch.sin(x)
out=nn(x)
x=x.data.numpy()
out=out.data.numpy()
y=y.data.numpy()
plt.plot(x,out,'r.')
plt.plot(x,y,'bo')

plt.show()
