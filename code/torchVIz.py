import torch
import torch.nn as nn
from torchviz import make_dot


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


model = Net()
x = torch.randn(1, 10)
y = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_graph", format="png")
