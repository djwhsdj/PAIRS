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
from tqdm import tqdm
import pickle
from function import vwsdk, counting, SDK
from utils_1 import *
import logging
import random

import torch.backends.cudnn as cudnn


##### train, test, retrain function #######################################################
def train(args, model, device, pattern_set, train_loader, test_loader, optimizer, Z, Y, U, V):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = admm_loss(args, device, model, Z, Y, U, V, output, target)
        loss.backward()
        optimizer.step()


def ttest(args, model, device, test_loader, pattern_set, Z, Y, U, V):
    model.eval()
    test_loss = 0
    loss_c = 0
    loss_z = 0
    loss_y = 0
    correct = 0
    with torch.no_grad():   # No Training
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.cross_entropy(output, target).item()
            loss_c += admm_lossc(args, device, model, Z, Y, U, V, output, target).item()
            loss_z += admm_lossz(args, device, model, Z, Y, U, V, output, target).item()
            loss_y += admm_lossy(args, device, model, Z, Y, U, V, output, target).item()
           
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    loss_c /= len(test_loader.dataset)
    loss_z /= len(test_loader.dataset)
    loss_y /= len(test_loader.dataset)
    prec = 100. * correct / len(test_loader.dataset)
    print('Accuracy : {:.2f}, Cross Entropy: {:f}, Z loss: {:f}, Y loss: {:f}'.format(prec, loss_c, loss_z, loss_y))
    #print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset), prec))

    return prec


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    prec = 100. * correct / len(test_loader.dataset)
    print('Accuracy: {:.2f}, Cross Entropy: {:f}'.format(prec, test_loss))

    return prec


def retrain(args, model, mask, device, train_loader, test_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = regularized_nll_loss(args, model, output, target)
        loss.backward()
        optimizer.prune_step(mask)



parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
# parser.add_argument('--model',      default='resnet18',         help='select model')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--dataset',    default='cifar10',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=0.001, type=float,   help='set learning rate') # 6e-5
# parser.add_argument('--re_lr',      default=1e-3, type=float,   help='set fine learning rate')
parser.add_argument('--alpha',      default=5e-4, type=float,   help='set l2 regularization alpha')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='adam epsilon')
parser.add_argument('--rho',        default=1, type=float,   help='set rho') # original 6e-1?
#parser.add_argument('--rho',        default=1000, type=float,   help='set rho')
#parser.add_argument('--rho',        default=1000, type=float,   help='set rho')
parser.add_argument('--connect_perc',  default=1, type=float, help='connectivity pruning ratio')
parser.add_argument('--epoch',      default=80, type=int,      help='set epochs') # 60
parser.add_argument('--re_epoch',   default=120, type=int,       help='set retrain epochs') # 100
parser.add_argument('--num_sets',   default='4', type=int,      help='# of pattern sets')
parser.add_argument('--exp',        default='test', type=str,   help='test or not')
parser.add_argument('--l2',         default=False, action='store_true', help='apply l3 regularization')
parser.add_argument('--scratch',    default=False, action='store_true', help='start from pretrain/scratch')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--method', default = 'random', type = str, help='patdnn, ours, random, click')
parser.add_argument('--withoc', type=int, default=1) 
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--mask', type=int, default=5)
parser.add_argument('--ar', type=int, default=512)
parser.add_argument('--ac', type=int, default=512)
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=2)
parser.add_argument('--seed', type=int, default=1992)
args = parser.parse_args()
print(args)

# candidate = 9 - args.mask + 1
candidate = args.mask
args.ac = int(args.ac/args.wb)
print(f'Actual used array columns = {args.ac}')

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print ('Current cuda device ', device)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_name = './log/resnet20/%s/1118/%s/%dx%d_epoch%d_reepoch%d_%s_wb%d_mask%d_numset%d_withoc%d_test.log'%(args.dataset, args.method, args.ar, args.ac, args.epoch, args.re_epoch, args.method, args.wb, args.mask, args.num_sets, args.withoc)

file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



if args.exp == 'test':
    args.exp = f'{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'

args.workers = 2

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
##########################################################################################################
'''
ResNet-20
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

        self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
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


print('Preparing pre-trained model...')
pre_model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=10).cuda()
if args.method == 'patdnn' :
    print('pretrained model uploaded...')
    pre_model.load_state_dict(torch.load('./pre_trained_0729.pt'), strict=True)
    # print("pre-trained model:\n", pre_model)

elif args.method == 'pruned_sparse' :
    print('pretrained model uploaded...')
    pre_model.load_state_dict(torch.load('./pre_trained_1119_sparse.pt'), strict=True)
    # print("pre-trained model:\n", pre_model)
else :
    print('new model uploaded...')


##### Load Dataset ####
print('\nPreparing Dataset...')
train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)
print('Dataset Loaded')

##### Find Pattern Set #####
print('\nFinding Pattern Set...')
if args.method == 'patdnn' :
    if os.path.isfile('pattern_set_'+ args.dataset + '_mask_' + str(args.mask) + '_0729.npy') is False:
        pattern_set = pattern_setter(pre_model, candidate)
        np.save('pattern_set_'+ args.dataset + '_mask_' + str(args.mask) + '_0729.npy', pattern_set)
    else:
        pattern_set = np.load('pattern_set_'+ args.dataset + '_mask_' + str(args.mask) + '_0729.npy')

    pattern_set = pattern_set[:args.num_sets, :]

elif args.method == 'pruned_sparse' :
    if os.path.isfile('pattern_set_'+ args.dataset + '_mask_' + str(args.mask) + '_0729_v.npy') is False:
        pattern_set = pattern_setter_v(pre_model, candidate)
        np.save('pattern_set_'+ args.dataset + '_mask_' + str(args.mask) + '_0729_v.npy', pattern_set)
    else:
        pattern_set = np.load('pattern_set_'+ args.dataset + '_mask_' + str(args.mask) + '_0729_v.npy')

    pattern_set = pattern_set[:args.num_sets, :]


elif args.method == 'random' :
    candi_list = []
    ran_list = [0,1,2,3,4,5,6,7,8]
    for i in range(args.num_sets) :
        one_list = [1,1,1,1,1,1,1,1,1]
        sample = random.sample(ran_list, args.mask)
        for sam in sample :
            one_list[sam] = 0
        candi_list.append(one_list)
    
    pattern_set = np.array(candi_list)

elif args.method == 'original' :
    pattern_set = np.array([[1,1,1,1,1,1,1,1,1]])

elif args.method == 'pconv' :
    pattern_set = np.array([[0,1,0,1,1,1,0,0,0], [0,1,0,1,1,0,0,1,0], [0,1,0,0,1,1,0,1,0], [0,0,0,1,1,1,0,1,0]])



if args.method == 'ours' :
    if args.mask == 5 : # 4
        pattern_set = np.array([[0,0,0,0,1,1,0,1,1], [0,0,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,0,0], [0,1,1,0,1,1,0,0,0]])
    elif args.mask == 3 : # 4
        pattern_set = np.array([[0,0,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,0,0], [1,1,0,1,1,0,1,1,0], [0,1,1,0,1,1,0,1,1]])
    elif args.mask == 1: # 4
        pattern_set = np.array([[0,1,1,1,1,1,1,1,1], [1,1,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,1,1], [1,1,1,1,1,1,1,1,0]])
    elif args.mask == 2 : # 14
        pattern_set = np.array([
                                [0,0,1,1,1,1,1,1,1], [1,0,0,1,1,1,1,1,1], [0,1,1,0,1,1,1,1,1], [1,1,0,1,1,0,1,1,1],
                                [1,1,1,0,1,1,0,1,1], [1,1,1,1,1,1,0,0,1], [1,1,1,1,1,1,1,0,0], [1,1,1,1,1,0,1,1,0],
                                [0,1,0,1,1,1,1,1,1], [1,1,0,1,1,1,1,1,0], [1,1,1,1,1,1,0,1,0], [0,1,1,1,1,1,0,1,1],
                                [0,1,1,1,1,1,1,1,0], [1,1,0,1,1,1,0,1,1]
                                ])
    elif args.mask == 4 :# 16
        pattern_set = np.array([
                                [0,0,0,0,1,1,1,1,1], [0,0,0,1,1,0,1,1,1], [0,0,0,1,1,1,0,1,1], [0,0,0,1,1,1,1,1,0],
                                [1,1,1,1,1,0,0,0,0], [1,1,1,0,1,1,0,0,0], [0,1,1,1,1,1,0,0,0], [1,1,0,1,1,1,0,0,0],
                                [0,1,1,0,1,1,0,0,1], [0,0,1,0,1,1,0,1,1], [0,1,0,0,1,1,0,1,1], [0,1,1,0,1,1,0,1,0],
                                [1,1,0,1,1,0,1,0,0], [1,0,0,1,1,0,1,1,0], [0,1,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,1,0]
                                ])

pattern_set_1 = pattern_set

print(pattern_set)
print('pattern_set loaded')
                    

##### Load Model #####
# model = utils.__dict__[args.model]()

## new model
model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=10).cuda()


print('lr:', args.lr)
print('\nTraining...') ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
best_prec = 0
Z, Y, U, V = initialize_Z_Y_U_V(model)

#WarmUp                                            
optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("-"*50)
images = [32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8]
pwr = []
pwh = []

idxx = -1
for name, param in model.named_parameters(): # name : weight, bias... # params : value list
    if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
        if name[:5] == 'conv1' :
            continue
        idxx += 1 
        output_c, input_c, kkr, kkh = param.shape
        _, pw_r, pw_h = SDK(images[idxx], images[idxx], kkr, kkh, input_c, output_c, args.ar, args.ac)
        pwr.append(pw_r)
        pwh.append(pw_h)


print('*'*100)
logger.info(f"Pattern set : {pattern_set}")
if args.method == 'ours' :
    logger.info(f"Pattern set_1 : {pattern_set_1}")


print("-"*50)
print('WarmUp for some epochs...')
print("-"*50)


for epoch in range(5):
    train(args, model, device, 0, train_loader, test_loader, optimizer, Z, Y, U, V)
       
    X = update_X(model)
    for ppww in range(len(pwr)) :
        if args.method == 'ours' :
            if ppww > 3 :
                Z=update_Z(X, U, pattern_set_1, args.withoc)
            else :
                Z=update_Z(X, U, pattern_set, args.withoc)
        else :
            Z=update_Z(X, U, pattern_set, args.withoc)
    Y = update_Y(X, V, args)

    prec = ttest(args, model, device, test_loader, pattern_set, Z, Y, U, V)



# Optimizer
# optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-4)
print("-"*50)
print('Lego training!!!')
print("-"*50)

for epoch in range(args.epoch+1):
    print(f"Current epochs : {epoch}")
    # if epoch in [args.epoch//2, (args.epoch * 3)//4]: ############################################################################
    # if epoch in [args.epoch//4, args.epoch//2, args.epoch//4*3]:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
    
    train(args, model, device, 0, train_loader, test_loader, optimizer, Z, Y, U, V)
       
    X = update_X(model)
    for ppww in range(len(pwr)) :
        if args.method == 'ours' :
            if ppww > 3 :
                Z=update_Z(X, U, pattern_set_1, args.withoc)
            else :
                Z=update_Z(X, U, pattern_set, args.withoc)
        else :
            Z=update_Z(X, U, pattern_set, args.withoc)

    Y = update_Y(X, V, args)
    U = update_U(U, X, Z)
    V = update_V(V, X, Y)

    prec = ttest(args, model, device, test_loader, 0, Z, Y, U, V)
    logger.info("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.epoch, prec))


# Real Pruning ! ! !
print("\nApply Pruning with connectivity & pattern set...")
mask = apply_prune_pat(args, model, device, pattern_set, pattern_set_1, pwr, args.withoc, args.method)
print_prune(model)

# Fine-tuning...
print("Retraining for fine tuning...")
logger.info("="*100)
logger.info("Retraining for fine tuning...")
best_prec = -1

# optimizer 2
optimizer_1 = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)

for epoch in range(args.re_epoch):
    print("Epoch: {} with re_lr: {}".format(epoch+1, args.lr))
    # if epoch in [args.re_epoch//4, args.re_epoch//2, args.re_epoch//4*3]:
    #     for param_group in optimizer_1.param_groups:
    #         param_group['lr'] *= 0.1

    retrain(args, model, mask, device, train_loader, test_loader, optimizer_1)

    print("\ntesting...")
    prec = test(args, model, device, test_loader)
    logger.info("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.re_epoch, prec))
        
    if prec > best_prec :
        best_prec = prec


idx = -1
total_skipped_rows = 0
layer6464_skipped_rows = 0
layer128128_skipped_rows = 0
layer256256_skipped_rows = 0
for name, param in model.named_parameters():
    if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
        if name[:5] == 'conv1' :
            continue
        rows = 0
        idx += 1
        if idx != -1 :
            oc, ic, kr, kh = mask[name].shape
            _, pw_r, pw_h = SDK(images[idx], images[idx], kr, kh, ic, oc, args.ar, args.ac)
            print(f'idx : {idx}, pwr : {pw_r}, pwh : {pw_h}')
        
            mode_ = True
            rows = counting(mask[name], mask[name].shape, pw_r, pw_h, mode_)
            print(f'layer name = {name}, # offed rows = {rows}')
            if 1 <= idx and idx <= 5 :
                layer6464_skipped_rows += rows
            elif 6 <= idx and idx <= 10:
                layer128128_skipped_rows += rows
            elif 11 <= idx and idx <= 15 :
                layer256256_skipped_rows += rows

            total_skipped_rows += rows


            param.data.mul_(mask[name])


logger.info("="*60)
logger.info(f'pwr = {pwr[5]}, pwh = {pwh[5]}, # total offed rows = {layer6464_skipped_rows}')
logger.info(f'pwr = {pwr[10]}, pwh = {pwh[10]}, # total offed rows = {layer128128_skipped_rows}')
logger.info(f'pwr = {pwr[15]}, pwh = {pwh[15]}, # total offed rows = {layer256256_skipped_rows}')
logger.info(f'# total offed rows = {total_skipped_rows}')
logger.info("-"*70)
logger.info(f'best acc = {best_prec}')
logger.info("-"*70)

print("="*100)
print(f'pwr = {pwr[5]}, pwh = {pwh[5]}, # total offed rows = {layer6464_skipped_rows}')
print(f'pwr = {pwr[10]}, pwh = {pwh[10]}, # total offed rows = {layer128128_skipped_rows}')
print(f'pwr = {pwr[15]}, pwh = {pwh[15]}, # total offed rows = {layer256256_skipped_rows}')
print(f'# total offed rows = {total_skipped_rows}')
print("-"*70)
print(f'best acc = {best_prec}')
print("-"*70)
print("="*100)


# my mistake 1 - making mask.pickle
"""
with open('mask.pickle', 'wb') as fw:
    pickle.dump(mask, fw)

with open('mask.pickle', 'rb') as fr:
    mask = pickle.load(fr)
    print("mask loaded")
"""

# my mistake 2
"""
for module in model.named_modules():
    if isinstance(module[1], nn.Conv2d):
        print("module:", module[0])
        prune.custom_from_mask(module, 'weight', mask=mask[module[0] +'.weight'])
"""


# torch.save(pre_model.state_dict(), 'pre_trained_1119_sparse.pt')



