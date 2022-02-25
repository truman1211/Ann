import torch
"""
it's the network module in Artificial Neural Networks for Solving Ordinary and Partial Differential Equations
"""
class ann(torch.nn.Module):
    def __init__(self):
        super(ann, self).__init__()

        self.hidden_layer = torch.nn.Linear(in_features=1, out_features=10, bias=True)
        self.out_layer = torch.nn.Linear(in_features=10, out_features=1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):

        Y = self.hidden_layer(x)
        Y = self.sigmoid(Y)
        Y = self.out_layer(Y)

        return Y

