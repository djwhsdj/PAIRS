import math

import numpy as np

def SDK (image_col, image_row, filter_col, filter_row, in_channel, out_channel, array_row, array_col) :
    
    row_vector = filter_row * filter_col * in_channel
    col_vector = out_channel
    
    used_row = math.ceil(row_vector/array_row)
    used_col = math.ceil(col_vector/array_col)
    
    new_array_row = array_row * used_row
    new_array_col = array_col * used_col

    # initialize
    cycle = []
    w = []
    w.append(filter_row*filter_col)
    cycle.append(used_row*used_col*(image_row-filter_row+1)*(image_col-filter_col+1))
    
    i=0
    while True :
        i += 1
        pw_row = filter_row + i - 1 
        pw_col = filter_col + i - 1
        pw = pw_row * pw_col
        if pw*in_channel <= new_array_row and i * i * out_channel <= new_array_col :
            parallel_window_row = math.ceil((image_row - (filter_row + i) + 1)/i) + 1
            parallel_window_col = math.ceil((image_col - (filter_col + i) + 1)/i) + 1
            
            if parallel_window_row * parallel_window_row * used_row * used_col <= cycle[0] :
                del cycle[0]
                del w[0]
                cycle.append(parallel_window_row * parallel_window_col * used_row * used_col)
                w.append(pw)
            
        else :
            break
        
    
    return cycle[0], int(math.sqrt(w[0])), int(math.sqrt(w[0]))


def counting (mask, layer_shape, pwr, pwh, mode = False) :
    mask = mask.cpu().numpy()
    OC, IC, kr, kh = layer_shape

    cnt = 0

    kernel = []
    for i in range(kr*kh) :
        kernel.append(i)
    
    for i in range(IC) :
        pw = []
        for j in range(pwr*pwh) :
            pw.append([])
            
        for a in range(pwh-kh+1) :
            for b in range(pwr-kr+1) :
                for c in range(len(kernel)) :
                    divider = c // 3
                    residue = c % 3
                    pw_idx = (divider+a)*pwr+(residue+b)
                    pw[pw_idx].append(kernel[c])
        
        zero_list = []
        for j in range(kr) :
            for k in range(kh) :
                cal = mask[:, i, j, k].sum()
                if cal == 0 :
                    idx = j*kr + k
                    zero_list.append(idx)

        for q in range(len(pw)) :
            for j in zero_list :
                if j in pw[q] :
                    pw[q].remove(j)

        for m in pw :
            if m == [] :
                cnt+=1

    if mode == True :
        print("="*60)
        for iccc in range(IC) :
            if iccc < 3 :
                print(mask[0][iccc])

    return cnt
