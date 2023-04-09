import torch

x = torch.randn(2, 2)
print(x)
indice = torch.tensor([[0], [1]])
print(torch.gather(x, 1, indice))