#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)
from train import model


import torch.optim as optim


optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)