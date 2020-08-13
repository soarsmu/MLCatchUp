import models
from dataset import ChessMoveDataset_cp_it
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import progressbar
import git
import os

import chess
import chess.engine



def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.xavier_normal_(m.weight,2**0.5)
    m.bias.data.fill_(0.01)
  if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
    torch.nn.init.xavier_normal_(m.weight,2**0.5)
    m.bias.data.fill_(0.01)

torch.backends.cudnn.benchmark = True

device = ('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
startepoch = 0
epochs = 1000
batch_size = 1<<10
random_subset = None

githash = git.Repo(search_parent_directories=True).head.object.hexsha
log_dir = "output/" + githash + "/"

os.mkdir(log_dir)

log_file = open(log_dir+"out.csv", "w")

model = models.unet_simple().to(device)
model.apply(init_weights)
#model = torch.load("output/e4fee41f41ee88653738189b8c6a8c155ef96a78/model_ep23.nn",map_location=device)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-1, momentum=.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.67, patience=0, verbose=True, threshold=1e-2)

trainset,valset = ChessMoveDataset_cp_it(mirror=False),ChessMoveDataset_cp_it(mode='val')

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=True)
val_iter = iter(val_loader)
log_file.write("epoch,batch_count,train_cross_entropy_loss,val_cross_entropy_loss,train_acc,val_acc,train_grads,train_min_cp,val_min_cp,train_1cp_acc,val_1cp_acc\n")

def multi_cross_entropy(predicted, target, mask, legal):
  # maximum number of legal moves this batch
  maxlegal = int((legal*mask).sum(dim=1).max().item())
  mtarget = torch.masked_fill(target,(legal==0)|(mask==0),-float('inf'))
  ptarget = nn.functional.softmax(mtarget,dim=1)
  val,idx = ptarget.sort()
  # discard all non legal moves
  best_val,best_idx = val[:,-maxlegal:],idx[:,-maxlegal:]
  loss = 0
  for i in range(maxlegal):
    lloss = nn.functional.cross_entropy(predicted,best_idx[:,i],reduction='none')
    lloss = torch.masked_fill(lloss,lloss == float('inf'), 0)
    loss += (best_val[:,i]*lloss).mean()
  return loss
  '''
  loss = 0
  midx = np.argpartition(-(target+mask).cpu().numpy(),topn)[:,:topn]
  w = torch.nn.functional.softmax(torch.tensor(np.take_along_axis(target.cpu().numpy(),midx,axis=1),device=device), dim=1)
  for i in range(topn):
    loss += (w[:,i]* nn.functional.cross_entropy(predicted, torch.tensor(midx[:,i],device=device),reduction='none')).mean()
  return loss
  '''



def loss_fcn(predicted, target, mask, legal):
  #mse = nn.functional.mse_loss(torch.flatten(predicted*mask),torch.flatten(target*mask*0.05),reduction='sum') / mask.sum()
  #hinge = (nn.functional.relu((predicted-target*0.05)*(1-mask))**2).sum() / (1-mask).sum()
  #cross_entropy = nn.functional.cross_entropy(predicted, (target+mask).argmax(dim=1),reduction='mean')
  #avg_cp_loss = -(nn.functional.softmax(predicted)*target).view(len(target),-1).sum(1).mean()
  #return avg_cp_loss
  m_cross_entropy = multi_cross_entropy(predicted, target, mask, legal)
  return m_cross_entropy

total_batch_count = 0
running_train_loss = None

def sum_grads(model):
  train_params = filter(lambda p: p.requires_grad, model.parameters())
  return sum(ti.grad.detach().cpu().abs().sum().numpy() for ti in train_params)

def acc_fnc(predicted,cplosses,mask):
 return (predicted.argmax(dim=1) == (cplosses+mask).argmax(dim=1)).cpu().numpy().mean()

def min_cp_loss_fnc(predicted,cplosses):
  return cplosses[torch.arange(predicted.size(0)),predicted.argmax(dim=1)].mean().cpu().numpy()

def top_1cp_acc_fnc(predicted,cplosses):
  return (cplosses[torch.arange(predicted.size(0)),predicted.argmax(dim=1)] > -1).float().mean().cpu().numpy()

def validate_batch():
  global val_iter
  x,c,m,l = None,None,None,None
  try:
    x,c,m,l = next(val_iter)
  except:
    val_iter = iter(val_loader)
    x,c,m,l = next(val_iter)
  
  x,c,m,l = x.to(device),c.to(device),m.to(device),l.to(device)
  with torch.no_grad():
    predicted = model(x)
    predicted = predicted.masked_fill(l==0,-float('inf'))
    val_loss = loss_fcn(predicted,c,m,l)
    val_acc = acc_fnc(predicted.detach(),c,m)
    min_cp_loss = min_cp_loss_fnc(predicted.detach(),c)
    top_1cp_acc = top_1cp_acc_fnc(predicted.detach(),c)

    return val_loss.detach().data.cpu().numpy(), val_acc, min_cp_loss, top_1cp_acc


def train():
  global total_batch_count
  global running_train_loss
  for x,c,m,l in progressbar.progressbar(train_loader,max_value=len(trainset)//batch_size):
    x,c,m,l = x.to(device),c.to(device),m.to(device),l.to(device)
    model.train()
    optimizer.zero_grad()
    #x,y = x.type(torch.float), y.type(torch.float)

    predicted = model(x)
    predicted = predicted.masked_fill(l==0,-float('inf'))
    train_loss = loss_fcn(predicted,c,m,l)
    train_loss.backward()
    train_grad = sum_grads(model)
    optimizer.step()

    train_acc = acc_fnc(predicted.detach(),c,m)
    train_min_cp_loss = min_cp_loss_fnc(predicted.detach(),c)
    train_top_1cp_acc = top_1cp_acc_fnc(predicted.detach(),c)
    val_loss = ''
    val_acc = ''
    val_min_cp_loss = ''
    val_top_1cp_acc = ''

    if (total_batch_count % 10 == 0):
      val_loss,val_acc, val_min_cp_loss, val_top_1cp_acc = validate_batch()

    log_file.write(','.join(map(str,[e,total_batch_count, train_loss.detach().data.cpu().numpy(), val_loss, train_acc, val_acc, train_grad, train_min_cp_loss, val_min_cp_loss, train_top_1cp_acc, val_top_1cp_acc]))+'\n')
    log_file.flush()

    total_batch_count += 1
    if running_train_loss is None:
      running_train_loss = train_loss.detach().data.cpu().numpy()
    running_train_loss = running_train_loss*0.9 + train_loss.detach().data.cpu().numpy()*0.1
    

def validate():
  with torch.no_grad:
    pass

for e in range(startepoch,epochs):
  torch.save(model, log_dir+'model_ep%d.nn'%e)
  print ("Epoch %d of %d:"%(e,epochs))

  train()
  #val_loss = validate()
  #print(val_loss)
  print(running_train_loss)

  scheduler.step(running_train_loss)

torch.save(model, 'output/model_ep%d.nn'%epochs)


log_file.close()
