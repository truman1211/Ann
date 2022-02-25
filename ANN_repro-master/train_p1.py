import torch
from Ann import ann
from Dataloader_p1 import dataset
from function_p1 import coe_right_function
from loss_p1 import problem1_loss
import os

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14, 14]
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ANN = ann()
epochs = 1000
batch_size = 10
lr = 0.003
ic = 1
"""
dataloader
"""
right_hand_function = coe_right_function.right_hand_function
coefficient_function = coe_right_function.coefficient_function

X = torch.linspace(0, 1, 1000).unsqueeze(1)  #in paper, they just use 10 point to train, but the performance is poor in my code.
dataset = dataset(X)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(ANN.parameters(), lr=lr)

loss_function = problem1_loss()

for epochs in range(epochs):

    for i, x in enumerate(dataloader):
        x = torch.autograd.Variable(x, requires_grad=True)

        optimizer.zero_grad()
        loss = loss_function(ANN, right_hand_function, coefficient_function, ic, x)

        # print("epoch:", epochs , ",batch:",i,"loss:",loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

x = torch.linspace(0, 2, 200, requires_grad=True).unsqueeze(1)
y_approx = (ic + x * ANN(x)).squeeze(0).detach().numpy().transpose()[0]
y_exact = ((torch.exp(-(x ** 2) / 2) / (1 + x + x ** 3)) + x ** 2).squeeze(0).detach().numpy().transpose()[0]
x = x.squeeze(0).detach().numpy().transpose()[0]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# Plot the analytical solution with the approximation
ax1.plot(x, y_approx, label='NN Approximation')
ax1.plot(x, y_exact, label='Exact Solution')
ax1.legend()

# Plot the residual
ax2.plot(x, y_exact - y_approx, label='Residual')
ax2.legend()

ax2.set_xlabel('x', fontsize=18)
ax1.set_ylabel("$\Psi(x)$", fontsize=18)
ax2.set_ylabel("Error", fontsize=18)
