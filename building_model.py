import torch.nn as nn 
import torch.nn.functional as F 
import torch



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # fc = fully connected
        self.fc1 = nn.Linear(28*28, 64) #784 it is flattened image 28*28
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # F.relu - this is our activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)
    

net = Net()

X = torch.rand(28,28)
X = X.view(-1, 28*28)
output = net(X)
print(output)

# the result we get is the actual predictions






