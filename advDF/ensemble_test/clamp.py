import torch
import os
x=torch.Tensor([[1,2,3],[2,1,4]])
# print(torch.clamp.__file__)
y=torch.max(x,torch.Tensor([[2,2,3],[2,3,1]]))
z=torch.min(y,torch.Tensor([[3,3,3],[4,4,4]]))
print(z)