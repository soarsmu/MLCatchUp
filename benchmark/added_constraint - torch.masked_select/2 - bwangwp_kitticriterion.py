from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn

def kitti_criterion(
    outputs,
    labels,
    num_classes=4,
    objectness_criterion=nn.CrossEntropyLoss(),
    classification_criterion=nn.CrossEntropyLoss(),
    location_criterion=nn.MSELoss(),
    dimension_criterion=nn.MSELoss()
):
    
    # book keeping
    batch_size = outputs.shape[0]
    
    # objectness loss
    outputs_objectness_logits = outputs[:,:2,:]
    labels_objectness = labels[:,0,:]
    objectness_loss = objectness_criterion(outputs_objectness_logits, labels_objectness.long())
    
    # objectness mask
    objectness_mask = labels_objectness.byte()
    
    # classification loss
    outputs_class_logits = outputs[:,2+num_classes*0:2+num_classes*1,:]
    outputs_class_logits = torch.masked_select(outputs_class_logits, torch.unsqueeze(objectness_mask, 1))
    outputs_class_logits = outputs_class_logits.view(batch_size, num_classes, -1)
    
    labels_class = labels[:,1,:]
    labels_class = torch.masked_select(labels_class, objectness_mask)
    labels_class = labels_class.view(batch_size, -1)

    classification_loss = classification_criterion(outputs_class_logits, labels_class.long())
    
    # class_mask
    class_mask = torch.zeros_like(outputs_class_logits)
    class_mask = class_mask.scatter_(1, torch.unsqueeze(labels_class, 1).long(), 1.0)
    class_mask = class_mask.byte()
    
    # dx loss
    outputs_dx = outputs[:,2+num_classes*1:2+num_classes*2,:]
    outputs_dx = torch.sigmoid(outputs_dx)
    outputs_dx = torch.masked_select(outputs_dx, torch.unsqueeze(objectness_mask, 1))
    outputs_dx = torch.masked_select(outputs_dx, class_mask.view(-1))
    outputs_dx = outputs_dx.view(batch_size, -1)
    
    labels_dx = labels[:,2,:]
    labels_dx = torch.masked_select(labels_dx, objectness_mask)
    labels_dx = labels_dx.view(batch_size, -1)
    
    dx_loss = location_criterion(outputs_dx, labels_dx)
    
    # dy loss
    outputs_dy = outputs[:,2+num_classes*2:2+num_classes*3,:]
    outputs_dy = torch.sigmoid(outputs_dy)
    outputs_dy = torch.masked_select(outputs_dy, torch.unsqueeze(objectness_mask, 1))
    outputs_dy = torch.masked_select(outputs_dy, class_mask.view(-1))
    outputs_dy = outputs_dy.view(batch_size, -1)
    
    labels_dy = labels[:,3,:]
    labels_dy = torch.masked_select(labels_dy, objectness_mask)
    labels_dy = labels_dy.view(batch_size, -1)
    
    dy_loss = location_criterion(outputs_dy, labels_dy)
    
    # dz loss
    outputs_dz = outputs[:,2+num_classes*3:2+num_classes*4,:]
    outputs_dz = torch.sigmoid(outputs_dz)
    outputs_dz = torch.masked_select(outputs_dz, torch.unsqueeze(objectness_mask, 1))
    outputs_dz = torch.masked_select(outputs_dz, class_mask.view(-1))
    outputs_dz = outputs_dz.view(batch_size, -1)
    
    labels_dz = labels[:,4,:]
    labels_dz = torch.masked_select(labels_dz, objectness_mask)
    labels_dz = labels_dz.view(batch_size, -1)
    
    dz_loss = location_criterion(outputs_dz, labels_dz)
    
    # h loss
    outputs_h = outputs[:,2+num_classes*4:2+num_classes*5,:]
    outputs_h = torch.masked_select(outputs_h, torch.unsqueeze(objectness_mask, 1))
    outputs_h = torch.masked_select(outputs_h, class_mask.view(-1))
    outputs_h = outputs_h.view(batch_size, -1)
    
    labels_h = labels[:,5,:]
    labels_h = torch.masked_select(labels_h, objectness_mask)
    labels_h = labels_h.view(batch_size, -1)
    
    h_loss = dimension_criterion(outputs_h, labels_h)
    
    # w loss
    outputs_w = outputs[:,2+num_classes*5:2+num_classes*6,:]
    outputs_w = torch.masked_select(outputs_w, torch.unsqueeze(objectness_mask, 1))
    outputs_w = torch.masked_select(outputs_w, class_mask.view(-1))
    outputs_w = outputs_w.view(batch_size, -1)
    
    labels_w = labels[:,6,:]
    labels_w = torch.masked_select(labels_w, objectness_mask)
    labels_w = labels_w.view(batch_size, -1)
       
    w_loss = dimension_criterion(outputs_w, labels_w)
    
    # l loss
    outputs_l = outputs[:,2+num_classes*6:2+num_classes*7,:]
    outputs_l = torch.masked_select(outputs_l, torch.unsqueeze(objectness_mask, 1))
    outputs_l = torch.masked_select(outputs_l, class_mask.view(-1))
    outputs_l = outputs_l.view(batch_size, -1)
    
    labels_l = labels[:,7,:]
    labels_l = torch.masked_select(labels_l, objectness_mask)
    labels_l = labels_l.view(batch_size, -1)
    
    l_loss = dimension_criterion(outputs_l, labels_l)
    
    # total loss
    loss = (
        objectness_loss +
        classification_loss +
        dx_loss +
        dy_loss + 
        dz_loss +
        h_loss +
        w_loss +
        l_loss
    )
    
    return loss
