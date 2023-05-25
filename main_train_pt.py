from __future__ import print_function
import os
import argparse
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
from torchvision import datasets, transforms
import utils
from utils import *
# from tqdm import tqdm
import pickle
from utils_1 import *
import logging
import random
import pandas as pd



##### Settings #########################################################################                      3x3에서는 patdnn을 따라가도록?
parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--model',      default='ResNet_Q',          help = 'select model : VGG9_Q, ResNet_Q')
parser.add_argument('--dataset',    default='cifar10',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=1e-3, type=float,   help='set learning rate') # 6e-5
parser.add_argument('--epoch',      default=120, type=int,      help='set epochs') # 60
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=2)
parser.add_argument('--original', type=int, default=1, help = '1.: Conv2D,   2.: Switched Conv2D')
args = parser.parse_args()
print(args)

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print ('Current cuda device ', device)

args.workers = 2

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#########################################################################################################
# ResNet code modified from original of [https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py]
# Modified version for our experiment.


'''
ResNet-20 Quantization
'''

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class BasicBlock_Q(nn.Module):
    expansion = 1
    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act1 = Activate(self.a_bit)

        if stride == 1 :
            if args.original == 1 : # conv2d
                self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
                self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
            else :
                self.conv1 = SwitchedConv2d(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=(2,1), bias=False)
                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
                self.conv2 = SwitchedConv2d(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=(2,1), bias=False)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

        else :
            if args.original == 1 : # conv2d
                self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
                self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
            else :
                self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
                self.conv2 = SwitchedConv2d(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=(2,1), bias=False)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                '''
                For CIFAR10 ResNet paper uses option A.
                '''
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    #Conv2d_Q_(self.w_bit, in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    #SwitchBatchNorm2d(self.w_bit, self.expansion * planes)
                    ## Full-precision
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        self.act2 = Activate(self.a_bit)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # x used here
        out = self.act2(out)
        return out

# ResNet code modified from original of [https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py]
# Modified version for our experiment.
class ResNet20_Q(nn.Module):
    def __init__(self, a_bit, w_bit, block, num_blocks, num_classes=10, expand=1): 
        super().__init__()
        self.in_planes = 16 # Resnet

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act = Activate(self.a_bit)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False),
            SwitchBatchNorm2d(self.w_bit, 16),
            Activate(self.a_bit),
            
            *self._make_layer(block, 16, num_blocks[0], stride=1),
            *self._make_layer(block, 32, num_blocks[1], stride=2),
            *self._make_layer(block, 64, num_blocks[2], stride=2),
        )

        # mask_prune(self.layers)
        self.fc = nn.Linear(64, num_classes) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # Full precision
            # option is 'A': Use F.pad
            # option is 'B': Use Conv+BN
            layers.append(block(self.a_bit, self.w_bit, self.in_planes, planes, stride, option='B'))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 



def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total


def train_model(model, train_loader, test_loader):
    # Clear acc_list
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1


    print(f'TRAINING START!')
    for epoch in range(args.epoch):
        model.train()
        cnt = 0
        loss_sum = 0
        for i, (img, target) in enumerate(train_loader):
            cnt += 1
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        
        loss_sum = loss_sum / cnt
        model.eval()
        acc = eval(model, test_loader)

        print(f'Epochs : {epoch+1}, Accuracy : {acc}')
        

        
        if acc > best_acc :
            best_acc = acc
            print('Best accuracy is updated! at epoch %d/%d: %.4f '%(epoch+1, args.epoch, best_acc))

    torch.save(model.state_dict(), './pre_trained_0729.pt')
    

def main():
    train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)

    model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=10).cuda()


    train_model(model, train_loader, test_loader)

if __name__=='__main__':
  main()





