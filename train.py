import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from make_data import make_data
from make_data import *

enc_input , dec_input , dec_output = make_data()
loader = Data.DataLoader(MyDataSet(enc_input, dec_input, dec_output), 2, True)
model = Transformer()

learning_rate = 1e-3
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

loss_fn = nn.CrossEntropyLoss()

n_epochs = 50

for e_poch in range(n_epochs):
    for enc_input, dec_input, dec_output in loader:
        output = model(enc_input, dec_input)
        loss = loss_fn(output, dec_output.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d , Loss: %f" % (e_poch,float(loss)));
torch.save(model, 'model.pth')
print("保存模型")