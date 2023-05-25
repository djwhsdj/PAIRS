import os
import torch
import numpy as np
import math
import copy
import random

def pattern_setter_gradual_pat(model, mask_list, candidate, method, num_sets=8):
    patterns = [[0,0,0,0,0,0,0,0,0,  0]]


    idxx = -1
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            if name[:5] == 'conv1' :
                continue
            
            print(f'name:{name}')
            idxx += 1
            par=param.cpu().detach().numpy()

            par1 = np.multiply(mask_list[idxx], np.abs(par))
            par1 = np.where(par1>0, 1, 0)
            par1 = par1.reshape(-1,1,9)
            patterns = get_pattern(patterns, par1)

            # par = np.multiply(mask_list[idxx], par)
            # patterns = get_pattern(patterns, top_4(par, candidate, method))

    patterns = np.array(patterns, dtype='int')
    patterns = patterns[patterns[:,9].argsort(kind='mergesort')]
    patterns = np.flipud(patterns)

    pattern_set = patterns[:num_sets,:9]

    
    return pattern_set

# def pattern_setter_gradual(model, mask_list, candidate, num_sets=8):
#     # patterns = [[0,0,0,0,0,0,0,0,0,  0]]
#     # patterns = [[]]
    
#     idxx = -1
#     pattern_pool = []
#     for name, param in model.named_parameters():
#         patterns = [[0,0,0,0,0,0,0,0,0,  0]]
#         # patterns = [[]]

#         if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
#             if name[:5] == 'conv1' :
#                 continue
#             idxx += 1
#             print(f'name:{name}')
#             par=param.cpu().detach().numpy()
#             par = np.multiply(mask_list[idxx],par)
#             # par=param.detach().numpy()

#             # patterns=get_pattern_gradual(patterns, top_4(par, candidate))
#             patterns=get_pattern_gradual(patterns, par)
            
 
#             patterns = np.array(patterns, dtype='int')
#             patterns = patterns[patterns[:,9].argsort(kind='mergesort')]
#             patterns = np.flipud(patterns)
#             # print(patterns)

#             pattern_set = patterns[:num_sets,:9]
#             # print(pattern_set)
#             # pattern_pool.append(pattern_set)

#             pattern_set_new = []
            
#             for pat in pattern_set :
#                 # print(pat)
#                 if sum(pat) != 0 :
#                     # pattern_set = np.delete(pattern_set, pat)
#                     pattern_set_new.append(pat)
#             # print(pattern_set_new)
#             # if patterns not in pattern_set : 
#             pattern_pool.append(pattern_set_new)
    
#             # print(pattern_set)
#     print('*'*100)
#     # print(pattern_pool)
#     return pattern_pool

def get_pattern_gradual(patterns, arr):               # input : (?, 1, 9) / output : (?, 10) 
    l = len(arr)


    for j in range(l):
        found_flag = 0
        for i in range(len(patterns)):
            # print(arr[j].tolist())
            if np.array_equal([patterns[i][0:9]], arr[j].tolist()):
                patterns[i][9] = patterns[i][9]+1
                found_flag = 1
                break
    
        if(found_flag == 0):
            y = np.c_[arr[j], [1]]
            if sum(y.tolist()[0]) != 0 :
                patterns.append(y.tolist()[0])

    return patterns


def get_pattern(patterns, arr):               # input : (?, 1, 9) / output : (?, 10) 
    l = len(arr)

    for j in range(l):
        found_flag = 0
        for i in range(len(patterns)):
            if np.array_equal([patterns[i][0:9]], arr[j].tolist()):
                patterns[i][9] = patterns[i][9]+1
                found_flag = 1
                break;

        if(found_flag == 0):
            y = np.c_[arr[j], [1]]
            patterns.append(y.tolist()[0])
    return patterns    



def top_4(arr, candidate):                     # input : (d, ch, 1, 9) / output : (d*ch, 1, 9)
    arr = arr.reshape(-1,1,9)
    arr = abs(arr)
    for i in range(len(arr)):
        arr[i][0][4] = 0 
        x = arr[i].copy()
        x.sort()
        arr[i]=np.where(arr[i]<x[0][candidate+1], 0, 1) 
        arr[i][0][4] = 1 
    return arr      

def top_4_v(arr, candidate):                     # input : (d, ch, 1, 9) / output : (d*ch, 1, 9)
    arr = arr.reshape(-1,1,9)
    arr = abs(arr)
    for i in range(len(arr)):
        # arr[i][0][4] = 0 
        x = arr[i].copy()
        x.sort()
        arr[i]=np.where(arr[i]<x[0][candidate], 0, 1) 
        # arr[i][0][4] = 1 
    return arr       
      


def pattern_setter(model, candidate, num_sets=8):
    patterns = [[0,0,0,0,0,0,0,0,0,  0]]
    
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            print(f'name:{name}')
            par=param.cpu().detach().numpy() 
            patterns=get_pattern(patterns, top_4(par, candidate))
 
    patterns = np.array(patterns, dtype='int')
    patterns = patterns[patterns[:,9].argsort(kind='mergesort')]
    patterns = np.flipud(patterns)

    pattern_set = patterns[:num_sets,:9]
    # print(pattern_set)
    
    return pattern_set

def pattern_setter_v(model, candidate, num_sets=8):
    patterns = [[0,0,0,0,0,0,0,0,0,  0]]
    
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            print(f'name:{name}')
            par=param.cpu().detach().numpy() 
            patterns=get_pattern(patterns, top_4_v(par, candidate))
 
    patterns = np.array(patterns, dtype='int')
    patterns = patterns[patterns[:,9].argsort(kind='mergesort')]
    patterns = np.flipud(patterns)

    pattern_set = patterns[:num_sets,:9]
    # print(pattern_set)
    
    return pattern_set



def top_4_pat(arr, pattern_set, withoc):    # input arr : (d, ch, 3, 3) or (d, ch, 1, 1)   pattern_set : (6~8, 9)
    if arr.shape[2] == 3:
        if withoc == 0 :
            cpy_arr = arr.reshape(-1, 9) 
            new_arr = np.zeros(cpy_arr.shape)
            pat_set = np.array(pattern_set).reshape(-1, 9)
            for i in range(len(cpy_arr)):
                pat_arr = cpy_arr[i] * pat_set
                pat_arr = np.linalg.norm(pat_arr, axis=1)
                pat_idx = np.argmax(pat_arr)

                new_arr[i] = cpy_arr[i] * pat_set[pat_idx]

            new_arr = new_arr.reshape(arr.shape)

        elif withoc == 1 : # KxKxICtxOCt
            cpy_arr = arr.copy()
            new_arr = np.zeros(cpy_arr.shape)
            tmp_pattern = []
    
            for patt in pattern_set :
                patt = np.array(patt).reshape(3,3)
                patt = np.tile(patt, reps = [arr.shape[0],1,1,1]) 
                patt = patt.reshape(-1,9)
                tmp_pattern.append(patt)
        
            pat_set = np.array(tmp_pattern)

            for i in range(len(cpy_arr[1])):
                pat_set = pat_set.reshape(len(pat_set), -1)
                pat_arr = cpy_arr[:,i].reshape(-1) * pat_set
                pat_arr = np.linalg.norm(pat_arr, axis=1)
                pat_idx = np.argmax(pat_arr)
                new_arr[:,i] = (cpy_arr[:,i].reshape(-1) * pat_set[pat_idx]).reshape(cpy_arr[1].shape)

                '''
                cpy_arr = arr[0].reshape(-1, 9) 
                new_arr = np.zeros(cpy_arr.shape)
                pat_set = np.array(pattern_set).reshape(-1, 9)
                for i in range(len(cpy_arr)):
                    pat_arr = cpy_arr[i] * pat_set
                    pat_arr = np.linalg.norm(pat_arr, axis=1)
                    pat_idx = np.argmax(pat_arr)
                    new_arr[i] = cpy_arr[i] * pat_set[pat_idx]
                '''

        else : # KxKxICxOC structured
            cpy_arr = arr.copy()
            new_arr = np.zeros(cpy_arr.shape)
            tmp_pattern = []
    
            for patt in pattern_set :
                patt = np.array(patt).reshape(3,3)
                patt = np.tile(patt, reps = [arr.shape[0],arr.shape[1],1,1]) 
                patt = patt.reshape(-1,9)
                tmp_pattern.append(patt)
        
            pat_set = np.array(tmp_pattern)

            pat_set = pat_set.reshape(len(pat_set), -1)
            pat_arr = cpy_arr.reshape(-1) * pat_set
            pat_arr = np.linalg.norm(pat_arr, axis=1)
            pat_idx = np.argmax(pat_arr)
            new_arr = (cpy_arr.reshape(-1) * pat_set[pat_idx]).reshape(arr.shape)
        return new_arr
    
    else:
        return arr
        
        

def top_k_kernel(arr, perc):    # input (d, ch, 3, 3)
    if arr.shape[2] == 1:
        new_arr = arr.copy().reshape(-1, 1)    # (d*ch, 1)
    elif arr.shape[2] == 3:
        new_arr = arr.copy().reshape(-1, 9)    # (d*ch, 9)
    else:
        return arr

    k = math.ceil(arr.shape[0] * arr.shape[1] / perc)
    l2_arr = np.linalg.norm(new_arr, axis=1)
    threshold = l2_arr[np.argsort(-l2_arr)[k-1]]
    l2_arr = l2_arr >= threshold
    
    if arr.shape[2] == 1:
        new_arr = new_arr.reshape(-1) * l2_arr

    elif arr.shape[2] == 3:
        l2_arr = l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr
        l2_arr = np.transpose(np.array(l2_arr))
        new_arr = new_arr * l2_arr
   
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


##### for 'main_swp.py' #####
def top_4_pat_swp(arr, pattern_set):   # input arr : (d, ch, 3, 3) or (d, ch, 1, 1)  pattern_set : (6~8, 9)
    if arr.shape[2] == 3:
        cpy_arr = arr.copy().reshape(len(arr), -1, 9)
        new_arr = np.zeros(cpy_arr.shape)
        pat_set = pattern_set.copy().reshape(-1, 9)
        pat_rst = np.zeros(len(pat_set))

        pat_arr = 0
        for i in range(len(cpy_arr)):
            for j in range(len(pat_set)):
                pat_arr = cpy_arr[i] * pat_set[j]
                pat_rst[j] = np.linalg.norm(pat_arr.reshape(-1))
        
            pat_idx = np.argmax(pat_rst)
            new_arr[i] = cpy_arr[i] * pat_set[pat_idx]

        new_arr = new_arr.reshape(arr.shape)
        return new_arr
    else:
        return arr



""" my mistake1... should use tensor / torch calculation! (for speed)

def top_4_pat(arr, pattern_set):    # input arr : (d, ch, 3, 3)   pattern_set : (6~8, 9) (9 is 3x3)
    cpy_arr = arr.copy().reshape(-1, 1, 9)
    new_arr = np.zeros(cpy_arr.shape)

    for i in range(len(cpy_arr)):
        max = -1
        for j in range(len(pattern_set)):
            pat_arr = cpy_arr[i] * pattern_set[j]
            pat_l2 = np.linalg.norm(cpy_arr[i])
            
            if pat_l2 > max:
                max = pat_l2
                new_arr[i] = pat_arr
        
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def top_k_kernel(arr, perc):    # input (d, ch, 3, 3)
    k = math.ceil(arr.shape[0] * arr.shape[1] / perc)
    new_arr = arr.copy().reshape(-1, 1, 9)    # (d*ch, 1, 9)
    l2_arr = np.zeros(len(new_arr))

    for i in range(len(new_arr)):
        l2_arr[i] = np.linalg.norm(new_arr[i]) 
        
    threshold = l2_arr[np.argsort(-l2_arr)[k-1]]    # top k-th l2-norm

    for i in range(len(new_arr)):
        new_arr[i] = new_arr[i] * (l2_arr[i] >= threshold)
    
    new_arr = new_arr.reshape(arr.shape)
    return new_arr
"""





