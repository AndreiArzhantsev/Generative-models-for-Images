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

class MyDataset(Dataset):
    SPLIT_RANDOM_SEED = 42
    TEST_SIZE = 0.25

    def __init__(self, path, df, transform=None, load_to_ram=True):
        self.path = path
        self.transform = transform
        self.all_files = []
        self.labels = []
        self.images = []
        self.load_to_ram = load_to_ram

        if df is not None:
            for _, row in df.iterrows():
                self.labels.append(row["Label"])
                self.all_files.append(row["Id"])
                if self.load_to_ram:
                    print(228)
                    self.images.append(Image.open(os.path.join(self.path, row["Id"])).convert("RGB"))
        else:
            self.all_files = sorted(os.listdir(self.path))
            if self.load_to_ram:
                for f in self.all_files:
                    self.images.append(Image.open(os.path.join(self.path, self.all_files[item])).convert("RGB"))
        
    
    def __len__(self):
        return len(self.all_files)
        
    def __getitem__(self, item):
        if self.labels:
            label = self.labels[item]
        else:
            label = 0

        if self.load_to_ram:
            self.images[item]
        else:
            image = Image.open(os.path.join(self.path, self.all_files[item])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label