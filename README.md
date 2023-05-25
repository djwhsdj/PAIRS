# PAIRS: Pruning-AIded Row-Skipping for In-Memory Convolutional Weight Mapping

## Abstract
Due to their energy-efficient computation, processing-in-memory (PIM) architecture is becoming a promising candidate for convolutional neural network (CNN) inference. Utilizing the weight sparsity of CNNs, computation of rows with zero-valued weights can be skipped to reduce the computing cycles. However, inherent sparsity of CNNs does not produce a high row-skipping ratio, since row-skipping requires the entire cells in a specific row to be zeros. In this paper, we propose pairing row-skipping with pattern-based pruning to skip more rows of CNN inference computation on a PIM array. The proposed PAIRS (pruning-aided row-skipping) method first determines the pattern dimension that shares a certain pattern shape, considering the PIM array and channel size. Then, for each pattern dimension, PAIRS determines the pattern shape that maximizes the row-skip ratio and minimizes the accuracy loss. When the 6-entry pattern is used, the simulation with a 512×512 PIM array in ResNet-20 shows that compared to no pruning, our proposed method achieves the cycle reduction
by 21.1% within 2% accuracy loss, while the prior work fails to reduce the computing cycles. Codes are available at the following address (https://github.com/918273917234982734/PAIRS)


### This code is based on https://github.com/7bvcxz/PatDNN

### !!Please check run_script.py before the training!!
  #### main_loss3_v1.py 
    * base code
  #### run_script.py 
    * running for main_loss_v1.py with defining the parameters
  ### pattern_setter.py
    * several functions for generating the pattern
    
    * define the pattern dimension (kernel-wise (KW), block-wise (BW) and array-wise (AW) )
