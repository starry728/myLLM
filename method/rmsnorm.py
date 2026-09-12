import torch

#开方求倒数
t = torch.rsqrt(torch.tensor(4.0))
print(t)

#创建一个全1张量
t2 = torch.ones(3,4)
print(t2)

#计算均值
mean = t2.mean()
print(mean)