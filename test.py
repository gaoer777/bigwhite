import numpy as np
import torch

a = torch.rand((5, 7))
conf, j = a[:, 5:].max(1)
print(a.numpy())