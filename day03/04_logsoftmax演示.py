import torch
from torch import nn

log_softmax = nn.LogSoftmax(dim=-1)
input=torch.randn(2,3)
print(input)
print(torch.softmax(input,dim=-1))
print(log_softmax(input))
