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
from function import vwsdk, counting
from utils_1 import *


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
        #loss = regularized_nll_loss(args, model, F.log_softmax(output, dim=1), target)
        loss = regularized_nll_loss(args, model, output, target)
        loss.backward()
        optimizer.prune_step(mask)


##### Settings #########################################################################                      3x3에서는 patdnn을 따라가도록?
parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--model',      default='resnet18',         help='select model')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--dataset',    default='cifar10',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=1e-3, type=float,   help='set learning rate') # 6e-5
parser.add_argument('--re_lr',      default=1e-3, type=float,   help='set fine learning rate')
parser.add_argument('--alpha',      default=5e-4, type=float,   help='set l2 regularization alpha')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='adam epsilon')
parser.add_argument('--rho',        default=6e-1, type=float,   help='set rho')
#parser.add_argument('--rho',        default=1000, type=float,   help='set rho')
parser.add_argument('--connect_perc',  default=1, type=float, help='connectivity pruning ratio')
parser.add_argument('--epoch',      default=1, type=int,      help='set epochs') # 30
parser.add_argument('--re_epoch',   default=40, type=int,       help='set retrain epochs') # 10
parser.add_argument('--num_sets',   default='8', type=int,      help='# of pattern sets')
parser.add_argument('--exp',        default='test', type=str,   help='test or not')
parser.add_argument('--l2',         default=False, action='store_true', help='apply l3 regularization')
parser.add_argument('--scratch',    default=False, action='store_true', help='start from pretrain/scratch')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--method', default = 'patdnn', type = str, help='patdnn, ours, ours_fixed')
parser.add_argument('--withoc', type=int, default=1) 
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--mask', type=int, default=3)
parser.add_argument('--gradual', type=int, default=0) 
parser.add_argument('--ar', type=int, default=512)
parser.add_argument('--ac', type=int, default=512)
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=2)
args = parser.parse_args()
print(args)
comment = "check11_patdnn"

candidate = args.mask
args.ac = int(args.ac/args.wb)
print(f'Actual used array columns = {args.ac}')

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print ('Current cuda device ', torch.cuda.current_device()) # check

if args.exp == 'test':
    args.exp = f'{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
args.save = f'logs/{args.dataset}/{args.model}/{args.exp}_lr{str(args.lr)}_rho{str(args.rho)}_{comment}_candidate{candidate}'

args.workers = 4

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
    def __init__(self, a_bit, w_bit, block, num_blocks, num_classes=10, expand=2): 
        super().__init__()
        self.in_planes = 64 # Resnet

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act = Activate(self.a_bit)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=False),
            SwitchBatchNorm2d(self.w_bit, 64),
            Activate(self.a_bit),
            
            *self._make_layer(block, 64, num_blocks[0], stride=1),
            *self._make_layer(block, 128, num_blocks[1], stride=2),
            *self._make_layer(block, 256, num_blocks[2], stride=2),
        )

        # mask_prune(self.layers)
        self.fc = nn.Linear(256, num_classes) 

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
if args.gradual == 1 :
    pre_model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=10).cuda()
else :
    if args.dataset == 'imagenet':
        pre_model = torchvision.models.vgg16(pretrained=True)
    elif args.dataset == 'cifar10' or args.dataset == 'svhn':
        pre_model = torchvision.models.resnet18(pretrained=True)
        # pre_model = utils.__dict__[args.model]()
        # pre_model.load_state_dict(torch.load('./cifar10_pretrain/vgg16_bn.pt'), strict=True)
print("pre-trained model:\n", pre_model)


### Find Pattern Set #####                      
if args.gradual == 0 :        
    print('\nFinding Pattern Set...')
    if args.method == 'patdnn' :
        if os.path.isfile('pattern_set_'+ args.dataset + str(candidate) + '.npy') is False:
            pattern_set = pattern_setter(pre_model, candidate)
            np.save('pattern_set_'+ args.dataset +str(candidate) + '.npy', pattern_set)
        else:
            pattern_set = np.load('pattern_set_'+ args.dataset + str(candidate) + '.npy')

        pattern_set = pattern_set[:args.num_sets, :]

    elif args.method == "ours" :
        if args.mask == 3 :
            # pattern_set = np.array([[0,0,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,0,0]])
            pattern_set = np.array([[0,0,0,1,1,1,1,1,1]])
        elif args.mask == 5 :
            pattern_set = np.array([[0,0,0,0,1,1,0,1,1]])

    print(pattern_set)
    print('pattern_set loaded')


##### Load Dataset ####
print('\nPreparing Dataset...')
train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)
print('Dataset Loaded')


##### Load Model #####
# model = utils.__dict__[args.model]()
model = pre_model

# if pre-trained... load pre-trained weight
if args.gradual == 0 :
    if not args.scratch:
        state_dict = pre_model.state_dict()
        torch.save(state_dict, 'tmp_pretrained.pt')

        model.load_state_dict(torch.load('tmp_pretrained.pt'), strict=True)
    model.cuda()
    pre_model.cuda()


# History collector
history_score = np.zeros((200, 2))
his_idx = 0

print('patdnn')
print('lr:', args.lr, 'rho:', args.rho)
print('\nTraining...') ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
best_prec = 0
Z, Y, U, V = initialize_Z_Y_U_V(model)

#WarmUp                                            
# optimizer = PruneAdam(model.named_parameters(), lr=1e-6, eps=args.adam_epsilon)
optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
print("-"*50)
####################################################################################################### 내가 추가한 부분
images = [32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8]
pwr = []
pwh = []

if args.gradual == 1 :
    print("Gradually finding the masks")
    print("-"*50)
    # for setting
    mask_list = []
    idxx = -1
    for name, param in model.named_parameters(): # name : weight, bias... # params : value list
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            if name[:5] == 'conv1' :
                continue
            idxx += 1 
            print(name, idxx, param[0,:,:,:].shape, param[:,0,:,:].shape)
            output_c, input_c, kkr, kkh = param.shape
            candi_mask = np.ones((output_c, input_c, kkr, kkh))
            mask_list.append(candi_mask)
            _, _, _, _, pw_r, pw_h = vwsdk(images[idxx], images[idxx], kkr, kkh, input_c, output_c, args.ar, args.ac)
            pwr.append(pw_r)
            pwh.append(pw_h)


    # for training
    epo = 2 #######################################
    masking_epoch = args.mask * epo
    for epoch in range(masking_epoch):
        print("-"*50)
        print(f"Current epochs : {epoch}")
        print("-"*50)
        train(args, model, device, epo, train_loader, test_loader, optimizer, Z, Y, U, V) # 여기서 epo는 없어도 되는 값
        if epoch % epo == 0 :
            idxx = -1
            for name, param in model.named_parameters(): # name : weight, bias... # params : value list
                if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
                    if name[:5] == 'conv1' :
                        continue
                    idxx += 1 
                    out_c, in_c, k_r, k_h = mask_list[idxx].shape
                    # mask_list[idxx] = mask_list[idxx].reshape(-1, k_r*k_h)
                    # weight_reshape = param.reshape(-1,k_r*k_h)

                    if args.method == 'patdnn' :
                        for ic in range(in_c) :
                            importance = []
                            for j in range(k_r*k_h) :
                                importance.append(0)

                            for idx in range(len(importance)) : 
                                divider = idx // 3
                                residue = idx % 3
                                if np.sum(mask_list[idxx][:, ic, divider, residue]) != 0 :
                                    # print(torch.sum(abs(param[:,ic, divider, residue])))
                                    importance[idx] = torch.sum(abs(param[:,ic, divider, residue])).item()
                                else :
                                    importance[idx] = 1000

                            importance[4] = 1000
                            min_candi = min(importance)
                            del_idx_candi = list(filter(lambda x: importance[x] == min_candi, range(len(importance))))
                            
                            divider = del_idx_candi[0] // 3
                            residue = del_idx_candi[0] % 3
                            
                            mask_list[idxx][:, ic, divider, residue] = 0

                    elif args.method == 'ours' : 
                        _, _, _, _, pw_r, pw_h = vwsdk(images[idxx], images[idxx], k_r, k_h, in_c, out_c, args.ar, args.ac)
                        masking_sq_pwrow = [0, 1, 2, 3, 5, 6, 7, 8]
                        masking_rec_pwrow = [0, 1, 2, 6, 7, 8]
                        masking_sq_edge_squre = [0, 2, 6, 8]
                        # masking_all = [0,1,2,3,4,5,6,7,8]

                        diction_squ = {
                            0 : [1, 3],
                            1 : [0, 2],
                            2 : [1, 5],
                            3 : [0, 6],
                            4 : [],
                            5 : [2, 8],
                            6 : [3, 7],
                            7 : [6, 8],
                            8 : [5, 7]
                            }
                        diction_rec = {
                            0 : [1, 3],
                            1 : [0, 2, 4],
                            2 : [1, 5],
                            3 : [0, 4, 6],
                            4 : [1, 3, 5, 7],
                            5 : [2, 4, 8],
                            6 : [3, 7],
                            7 : [6, 4, 8],
                            8 : [5, 7]
                        }

                        for ic in range(in_c) :
                            importance = []
                            score = []
                            for j in range(k_r*k_h) :
                                importance.append(0)
                                score.append(0)

                            for idx in range(len(importance)) : 
                                divider = idx // 3
                                residue = idx % 3
                                if np.sum(mask_list[idxx][:, ic, divider, residue]) == 0 :
                                    score[idx] = -10
                                    if pw_r > pw_h :
                                        for k_item in diction_rec[idx] :
                                            score[k_item] += 1
                                    else :
                                        for k_item in diction_squ[idx] :
                                            score[k_item] += 1
                                    importance[idx] = 1000
                                
                                else :
                                    importance[idx] = torch.sum(abs(param[:,ic,divider,residue])).item()

                            if pw_r > k_r :
                                # if pw_h == k_h : 
                                for i in masking_rec_pwrow :
                                    score[i] += 1
                                for i in masking_sq_edge_squre :
                                    score[i] += 1
                                        
                            score[4] = -10
                            importance[4] = 1000


                            min_candi = max(score)
                            del_idx_candi = list(filter(lambda x: score[x] == min_candi, range(len(score))))
                            del_idx_cand = []
                            if ic == 0 :
                                print(score)

                            for del_cand_ in del_idx_candi:
                                divider = del_cand_ // 3
                                residue = del_cand_ % 3
                                if importance[del_cand_] != 1000 : 
                                    del_idx_cand.append(del_cand_)

                            del_candi_score = []
                            for del_cand in del_idx_cand :
                                del_candi_score.append(score[del_cand])
    

                            max_score = del_candi_score[0]
                            max_index = list(filter(lambda x: score[x] == max_score, range(len(score))))
                            divider = max_index[0] // 3
                            residue = max_index[0] % 3
                            mask_list[idxx][:, ic, divider, residue] = 0

    # gradual 고정 패턴은 여기
        # print(mask_list)
    if args.gradual == 1 :
        if args.method == 'ours' :
            if candidate == 5 :
                num = 4 
            elif candidate == 3 :
                num = 2
            pattern_set = pattern_setter_gradual(pre_model, mask_list, candidate, num)
        elif args.method == 'patdnn':
            pattern_set = pattern_setter_gradual_pat(pre_model, mask_list, candidate, args.num_sets)
    
    elif args.gradual == 0 :
        if args.method == 'patdnn' :
            pattern_set = pattern_setter(pre_model, candidate, args.num_sets)
    

    print('*'*100)
    for i in range(len(pattern_set)) :
        print(pattern_set[i])



if args.gradual == 0 and args.method == 'ours_fixed' : 
    mask_list = []

    idxx = -1
    for name, param in model.named_parameters(): # name : weight, bias... # params : value list
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            if name[:5] == 'conv1' :
                continue
            print(name)
            idxx += 1 
            output_c, input_c, kkr, kkh = param.shape
            candi_mask = np.ones((output_c, input_c, kkr, kkh))
            mask_list.append(candi_mask)

    
    for i in range(len(mask_list)) :
        oc, ic, kr, kh = mask_list[i].shape 
        _, _, _, _, pw_r, pw_h = vwsdk(images[i], images[i], kr, kh, ic, oc, args.ar, args.ac)
        if args.mask == 3 :
            if pw_r > pw_h :
                if pw_h >= kh :
                    patten_set = np.array([[0,0,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,0,0]])
            else :
                patten_set = np.array([[0,0,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,0,0]])

        elif args.mask == 5 :
            if pw_r > pw_h :
                if pw_h > kh :
                    patten_set = np.array([[0,0,0,0,1,1,0,1,1], [0,0,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,0,0], [0,1,1,0,1,1,0,0,0]])
                elif pw_h == kh :
                    patten_set = np.array([[0,0,0,0,1,0,1,1,1], [0,0,0,1,1,1,0,1,0], [1,1,1,0,1,0,0,0,0], [0,1,0,1,1,1,0,0,0]])
            else :
                patten_set = np.array([[0,0,0,0,1,0,1,1,1], [0,0,0,1,1,1,0,1,0], [1,1,1,0,1,0,0,0,0], [0,1,0,1,1,1,0,0,0]])

    pattern_set = pattern_setter_gradual(pre_model, mask_list, candidate)



print("-"*50)
print('WarmUp for some epochs...')
print("-"*50)

# print(pattern_set)
# print("-"*50)
# print(len(pattern_set))

for epoch in range(5):
    train(args, model, device, 0, train_loader, test_loader, optimizer, Z, Y, U, V)
       
    X = update_X(model)
    if args.method == 'ours' :
        for pat in range(len(pattern_set)) :
            Z=update_Z(X, U, pattern_set[pat], args.withoc)
    elif args.method == 'patdnn' :
        Z = update_Z(X, U, pattern_set, args.withoc)
    Y = update_Y(X, V, args)

    prec = ttest(args, model, device, test_loader, pattern_set, Z, Y, U, V)
    history_score[his_idx][0] = epoch
    history_score[his_idx][1] = prec
    his_idx += 1




# Optimizer
optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
print("-"*50)
print('Lego training!!!')
print("-"*50)

best_acc = -1
for epoch in range(args.epoch):
    print(f"Current epochs : {epoch}")
    if epoch in [args.epoch//2, (args.epoch * 3)//4]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    
    train(args, model, device, 0, train_loader, test_loader, optimizer, Z, Y, U, V)
       
    X = update_X(model)
    if args.method == 'ours' :
        for pat in range(len(pattern_set)) :
            Z=update_Z(X, U, pattern_set[pat], args.withoc)
    elif args.method == 'patdnn' :
        Z = update_Z(X, U, pattern_set, args.withoc)
    Y = update_Y(X, V, args)
    U = update_U(U, X, Z)
    V = update_V(V, X, Y)

    prec = ttest(args, model, device, test_loader, 0, Z, Y, U, V)
    history_score[his_idx][0] = epoch
    history_score[his_idx][1] = prec
    his_idx += 1

    if prec > best_acc :
        best_acc = prec
    
create_exp_dir(args.save)
if args.dataset == 'cifar10' :
    name = 'cifar_before.pth.tar'
elif args.dataset == 'svhn' :
    name = 'svhn_before.pth.tar'

torch.save(model.state_dict(), os.path.join(args.save, name))




# Real Pruning ! ! !
print("\nApply Pruning with connectivity & pattern set...")
if args.method == 'patdnn' :
    mask = apply_prune_pat(args, model, device, pattern_set, args.withoc)
elif args.method == 'ours' :
    mask = apply_prune(args, model, device, pattern_set, args.withoc)

print_prune(model)

if args.dataset == 'cifar10' :
    name_ = 'cifar_after.pth.tar'
elif args.dataset == 'svhn' :
    name_ = 'svhn_after.pth.tar'
    
torch.save(model.state_dict(), os.path.join(args.save, name_))

idx = -1
total_skipped_rows = 0
layer6464_skipped_rows = 0
layer128128_skipped_rows = 0
layer256256_skipped_rows = 0
# layer512512_skipped_rows = 0
for name, param in model.named_parameters():
    if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
        if name[:5] == 'conv1' :
            continue
        rows = 0
        idx += 1
        if idx != -1 :
            oc, ic, kr, kh = mask[name].shape
            _, _, _, _, pw_r, pw_h = vwsdk(images[idx], images[idx], kr, kh, ic, oc, args.ar, args.ac)
            print(f'idx : {idx}, pwr : {pw_r}, pwh : {pw_h}')
            print(mask[name].shape)
        
            if pw_r > kr :
                mode_ = True
                rows = counting(mask[name], mask[name].shape, pw_r, pw_h, mode_)
                print(f'layer name = {name}, # offed rows = {rows}')
                if 0 <= idx and idx <= 5 :
                    layer6464_skipped_rows += rows
                elif 6 <= idx and idx <= 10:
                    layer128128_skipped_rows += rows
                elif 11 <= idx and idx <= 15 :
                    layer256256_skipped_rows += rows
                # elif 13 <= idx and idx <= 16 :
                #     layer512512_skipped_rows += rows

                total_skipped_rows += rows


            param.data.mul_(mask[name])



print("="*100)
print(f'final ecpoh = {epoch+1}')
print(f'pwr = {pwr[5]}, pwh = {pwh[5]}, # total offed rows = {layer6464_skipped_rows}')
print(f'pwr = {pwr[10]}, pwh = {pwh[10]}, # total offed rows = {layer128128_skipped_rows}')
print(f'pwr = {pwr[15]}, pwh = {pwh[15]}, # total offed rows = {layer256256_skipped_rows}')
# print(f'pwr = {pwr[16]}, pwh = {pwh[16]}, # total offed rows = {layer512512_skipped_rows}')      
print(f'# total offed rows = {total_skipped_rows}')
print("-"*70)
print(f'best acc = {best_acc}')
print("-"*70)
print("="*100)


print("\ntesting...")
test(args, model, device, test_loader)


# Optimizer for Retrain
optimizer = PruneAdam(model.named_parameters(), lr=args.re_lr, eps=args.adam_epsilon)


############################### removed
# # Fine-tuning...
# print("\nfine-tuning...")
# best_prec = 0
# for epoch in range(args.re_epoch):
#     if epoch in [args.re_epoch//4, args.re_epoch//2, args.re_epoch//4*3]:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] *= 0.1

#     retrain(args, model, mask, device, train_loader, test_loader, optimizer)

#     prec = test(args, model, device, test_loader)
#     history_score[his_idx][0] = epoch
#     history_score[his_idx][1] = prec
#     his_idx += 1
    
#     if prec > best_prec:
#         best_prec = prec
#         print(f'Best accuracy is updated!!! : {best_prec}')
#         torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_pruned.pth.tar'))

# np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')

# print('patdnn lr:', args.lr, 'rho:', args.rho)

############################################

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






