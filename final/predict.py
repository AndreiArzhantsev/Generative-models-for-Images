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

def predict(device, model, loader):
    model.eval()
    predicts = []
    s=0
    for data, target in loader:
        if s%100==0:
            print(s)
        s+=1
        data = data.to(device)
        with torch.no_grad():
            logits = model(data)
            predicts.extend(logits.argmax(dim=1))
    return predicts