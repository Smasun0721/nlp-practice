import torch
from torch import nn

# 输入维度  输出维度    隐藏层数
rnn = nn.RNN(5, 6, 1)
# 序列长度  批次数 输入维度
input = torch.randn(1, 3, 5)
# 隐藏层数  批次数 隐藏层维度
h0 = torch.randn(1, 3, 6)

output, h1 = rnn(input, h0)

print(output)
print(output.shape)
print(h1)
print(h1.shape)
