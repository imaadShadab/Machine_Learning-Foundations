import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn

x1 = torch.tensor(
    [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
        [4, 1],
        [4, 2],
        [4, 3],
        [5, 1],
        [5, 2],
        [5, 3],
        [6, 1],
        [6, 2],
        [6, 3],
        [7, 1],
        [7, 2],
        [7, 3],
        [8, 1],
        [8, 2],
        [8, 3],
    ],
    dtype=torch.float32,
)

y1 = torch.tensor(
    [
        6,
        8,
        10,
        8,
        11,
        13,
        11,
        12,
        16,
        14,
        15,
        17,
        18,
        20,
        22,
        22,
        23,
        25,
        25,
        27,
        29,
        30,
        32,
        34,
    ],
    dtype=torch.float32,
).reshape(-1, 1)


class MyDataSet(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


dataset = MyDataSet(x1, y1)

train_size = int(0.8 * len(dataset))
validate_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, validate_size])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=2)


layers = []
layers.append(nn.Linear(2, 4))
layers.append(nn.ReLU())

for _ in range(3):
    layers.append(nn.Linear(4, 4))
    layers.append(nn.ReLU())

network = nn.Sequential(*layers)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = network
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)

        return x


neural_network = NeuralNetwork()

cost_function = nn.MSELoss()

optimizer = torch.optim.SGD(neural_network.parameters(), lr=0.001)

neural_network.train()

train_losses = []
val_losses = []

for epoch in range(701):

    total_train_cost = 0

    for x_train, y_train in train_loader:

        optimizer.zero_grad()

        prediction = neural_network(x_train)

        cost = cost_function(prediction, y_train)

        cost.backward()

        optimizer.step()

        total_train_cost += cost.item()

    average_train_cost = total_train_cost / len(train_loader)

    neural_network.eval()

    total_val_cost = 0

    with torch.inference_mode():

        for x_val, y_val in val_loader:

            prediction = neural_network(x_val)

            cost = cost_function(prediction, y_val)

            total_val_cost += cost.item()

    average_val_cost = total_val_cost / len(val_loader)

    train_losses.append(average_train_cost)
    val_losses.append(average_val_cost)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"Train Loss = {average_train_cost:.4f}, "
            f"Validation Loss = {average_val_cost:.4f}"
        )

    neural_network.train()


x_new = torch.tensor([[1, 4]], dtype=torch.float32)

neural_network.eval()

with torch.inference_mode():
    prediction = neural_network(x_new)

print(f"Prediction: {prediction}")
