import sys
import os

sys.path.append(os.path.abspath('..'))
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from core.utils import seed_everything, get_device, build_dataloaders, train_one_epoch, evaluate, count_parameters, plot_history
from core.models import CNN
seed_everything()
get_device()

##################################################################
class SimpleCNN(nn.modules):
    def __init__(self) -> None:
        super.__init__()
def _build_dataloaders():
    return
def _train_one_epoch():
    return
def _evaluate():
    return


