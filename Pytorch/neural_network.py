import torch
from torch import nn

x1 = torch.tensor([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 1],
    [1, 3],
    [3, 2],
    [2, 3]
], dtype=torch.float32)

y1 = torch.tensor([
    4,
    6,
    7,
    9,
    12,
    8,
    14,
    11
], dtype=torch.float32).reshape(-1, 1)


layers = []

# Input -> first hidden layer
layers.append(nn.Linear(2, 5))
layers.append(nn.ReLU())

# Remaining 2 hidden layers
for _ in range(4):
    layers.append(nn.Linear(5, 5))
    layers.append(nn.ReLU())
    


network = nn.Sequential(*layers)


class NeuralNetowrk(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = network
        self.output = nn.Linear(5, 1)
        
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        
        return x
        

n_network = NeuralNetowrk()

cost_function = nn.MSELoss()

optimizer = torch.optim.SGD(n_network.parameters(), lr = 0.001)

for epoch in range(700):
    
    # Clear previous gradients and set them to 0, otherwise they accumulate and disrupt calculations
    optimizer.zero_grad()
    
    # forward propagation
    prediction = n_network(x1)
    
    
    cost = cost_function(prediction, y1)
    
    # Back propagation, only calculates the graidents, doesn't update the params (weights & biases)
    cost.backward() 
    
    # update the weights and biases
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {cost.item()}")
    
    
x_new = torch.tensor([[1, 1]], dtype=torch.float32)
predict = n_network(x_new)

print(predict)

# for name, param in network.named_parameters():
#     print(name, param.size())