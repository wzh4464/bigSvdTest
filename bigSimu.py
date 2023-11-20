'''
File: /bigSimu.py
Created Date: Monday November 20th 2023
Author: Zihan
-----
Last Modified: Monday, 20th November 2023 10:00:18 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

# use gpu to svd matrix A

import numpy as np
import torch
import time

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

# preprocessing
from sklearn import preprocessing

A = np.random.rand(10000, 10000)
A = preprocessing.normalize(A, norm='l2')
A = torch.from_numpy(A)

# using 3 gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
A = A.to(device)
A = Variable(A)

# svd
start_gpu = time.time()
u, s, v = torch.svd(A)
end_gpu = time.time()
print("GPU time: ", end_gpu - start_gpu) # GPU time:  511.34471321105957
