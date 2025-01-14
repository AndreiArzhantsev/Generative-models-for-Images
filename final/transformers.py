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
from torchvision.transforms.functional import InterpolationMode

transform = transforms.Compose(
        [
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform2 = transforms.Compose(
        [
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_train = transforms.Compose(
        [
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.AutoAugment(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

transform_train2 = transforms.Compose(
        [
        transforms.Resize((32,32)),
        transforms.Random(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
