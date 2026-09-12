import torch
import torch.nn as nn

# dropout_layer = nn.Dropout(p=0.5)

# t1 = torch.Tensor([1,2,3])
# t2 = dropout_layer(t1)
# # 这里dropout丢弃为了保持期望不变，将其他部分扩大两倍
# print(t2)

# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.Tensor([1, 2, 3])  # shape: (3,)
# t2 = torch.Tensor([[1, 2, 3]])  # shape: (1, 3)
# # 这里应用的w和b是随机的，真正训练里会在optimizer上更新
# output2 = layer(t2)  # shape: (1, 5)
# print(output2)

# t = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])# 【2，6】
# t_view1 = t.view(3, 4)#【3，4】
# print(t_view1)
# t_view2 = t.view(4, 3)
# print(t_view2)#变形状的

#交换（转置）
# t1 = torch.Tensor([[1, 2, 3], [4, 5, 6]])#[2,3]
# t1 = t1.transpose(0, 1)
# print(t1)

# 主对角线下置零--掩码用到
# x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(torch.triu(x))
# print(torch.triu(x,diagonal=1))

#重新构建形状
x = torch.arange(1, 7)  # tensor([1, 2, 3, 4, 5, 6])

y = torch.reshape(x, (2, 3))
print(y)
# 输出:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# 使用 -1 自动推断
z = torch.reshape(x, (3, -1))
print(z)
