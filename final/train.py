import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import os
import shutil
from PIL import Image
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import wandb
import sys


def test_m(device, model, loader):
    loss_log = []
    acc_log = []
    model.eval()
    
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        
        
        logits = model(data)
        loss = nn.CrossEntropyLoss()(logits, target)
        acc = (logits.argmax(dim=1) == target).sum() / target.shape[0]
        
        loss_log.append(loss.item())
        acc_log.append(acc.item()) 
        
    return np.mean(loss_log), np.mean(acc_log)

def train_epoch(device, model, optimizer, train_loader):
    loss_log = []
    acc_log = []
    model.train()
    s=0
    for data, target in tqdm(train_loader, file=sys.stdout):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = nn.CrossEntropyLoss()(logits, target)
        
        loss.backward()
        acc = (logits.argmax(dim=1) == target).sum() / target.shape[0]
        optimizer.step()
        
        loss_log.append(loss.item())
        acc_log.append(acc.item())  

    return loss_log, acc_log

def train_m(device, model, optimizer, n_epochs, train_loader, val_loader, scheduler=None):
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(device, model, optimizer, train_loader)
        val_loss, val_acc = test_m(device, model, val_loader)
        
        train_loss_log.extend(train_loss)
        train_acc_log.extend(train_acc)
        
        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)
        
        wandb.log({
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_loss': np.mean(train_loss),
            'train_acc': np.mean(train_acc),
        })

#         print(f"Epoch {epoch}")
#         print(f" train loss: {np.mean(train_loss)}, train acc: {np.mean(train_acc)}")
#         print(f" val loss: {val_loss}, val acc: {val_acc}\n")
        
        if epoch==1:
            torch.save(model.state_dict(), f'mobilenet_sem_plus1.pt')
        if epoch==3:
            torch.save(model.state_dict(), f'mobilenet_sem_plus3.pt')
        if epoch==5:
            torch.save(model.state_dict(), f'mobilenet_sem_plus5.pt')
        if epoch==7:
            torch.save(model.state_dict(), f'mobilenet_sem_plus7.pt')
        if epoch==9:
            torch.save(model.state_dict(), f'mobilenet_sem_plus9.pt')
        if epoch==11:
            torch.save(model.state_dict(), f'mobilenet_sem_plus11.pt')
        if epoch==13:
            torch.save(model.state_dict(), f'mobilenet_sem_plus13.pt')
        if epoch==15:
            torch.save(model.state_dict(), f'mobilenet_sem_plus15.pt')
        if epoch==19:
            torch.save(model.state_dict(), f'mobilenet_sem_plus19.pt')
            
        if scheduler is not None:
            scheduler.step(val_loss)

    return train_loss_log, train_acc_log, val_loss_log, val_acc_log