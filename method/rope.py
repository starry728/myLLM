import torch

# x = torch.tensor([1,2,3,4,5])
# y = torch.tensor([10,20,30,40,50])
# conditon = x > 3
# result = torch.where(conditon, x, y)#x里面不满足条件的、用对应位置y上的元素补充
# print(result) 

t = torch.arange(0, 10, 2)
print(t)
t2 = torch.arange(5,0,-1)
print(t2)
v1 = torch.tensor([1,2,3])
v2 = torch.tensor([4,5,6])
result = torch.outer(v1, v2)#外积
print(result)

# t1 = torch.tensor([[[1,2,3],[4,5,6]],[[13,14,15],[16,17,18]]])
# t2 = torch.tensor([[[7,8,9],[10,11,12]],[[19,20,21],[22,23,24]]])
# print(t1.shape)#【2，2，3】
# result = torch.cat((t1,t2),dim=0)#在第0个维度上结合【4，2，3】
# print(result)
# result = torch.cat((t1,t2),dim=1)#在第1个维度上结合【2，4，3】
# print(result)
# result = torch.cat((t1,t2),dim=2)#在第2个维度上结合【2，2，6】
# print(result)

# t1 = torch.Tensor([1,2,3])
# t2 = t1.unsqueeze(0)#自适应增加一个维度
# print(t1)
# print(t1.shape)
# print(t2)
# print(t2.shape)