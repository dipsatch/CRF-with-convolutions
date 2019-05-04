# from conv import Conv   # SJ
from conv_dg import Conv  # DG
import torch
import torch.nn as nn

X = torch.tensor([[[[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]]]], dtype=torch.float)
K1 = torch.nn.Parameter(torch.tensor([[[[1,0,1],[0,1,0],[1,0,1]]]], dtype=torch.float))
print("Please enter size of padding (enter 0 for no padding): ")
padding=int(input().strip())
conv2 = Conv(K1,padding,stride=1)
print("Output:")
print(conv2.forward(X))

