import torch

x = torch.randn(5, 5, 10)
print(x)
print(torch.mean(x, dim=1))
# torch gather
# indice = torch.tensor([[0], [1]])
# print(torch.gather(x, 1, indice))