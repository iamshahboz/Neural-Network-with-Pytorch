import torch 

x = torch.Tensor([5,3])  
y = torch.Tensor([2,1])

'''
Pytorch in simple words is a numpy which runs on gpu

Tensor is a multi dimentional array
'''

zeros = torch.zeros([2,5])

random = torch.rand([2,5])

print(random)
print(zeros)
print(x*y)