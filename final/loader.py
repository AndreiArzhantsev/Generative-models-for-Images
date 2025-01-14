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

def get_data(batch_size, data, shuffle):
    torch.manual_seed(0)
    np.random.seed(0)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    
    return data_loader