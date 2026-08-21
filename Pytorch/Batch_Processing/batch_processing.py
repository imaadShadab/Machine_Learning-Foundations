import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# DATA
# ============================================================

x1 = torch.tensor(
    [
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [3, 1],
        [1, 3],
        [3, 2],
        [2, 3]
    ],
    dtype=torch.float32
)

y1 = torch.tensor(
    [4, 6, 7, 9, 12, 8, 14, 11],
    dtype=torch.float32
).reshape(-1, 1)


# ============================================================
# DATASET
# ============================================================

class MyDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def create_dataloader(x, y, batch_size=2, shuffle=True):

    dataset = MyDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


# ============================================================
# MODEL
# ============================================================

def create_model(
    input_features,
    hidden_neurons,
    hidden_layers,
    output_features):

    layers = []

    # Input -> first hidden layer
    layers.append(nn.Linear(input_features, hidden_neurons))
    layers.append(nn.ReLU())

    # Remaining hidden layers
    for _ in range(hidden_layers - 1):
        layers.append(nn.Linear(hidden_neurons, hidden_neurons))
        layers.append(nn.ReLU())

    network = nn.Sequential(*layers)

    class NeuralNetwork(nn.Module):

        def __init__(self):
            super().__init__()

            self.hidden = network
            self.output = nn.Linear(hidden_neurons, output_features)

        def forward(self, x):

            x = self.hidden(x)
            x = self.output(x)

            return x

    return NeuralNetwork()


# ============================================================
# TRAINING
# ============================================================

def fit(model, dataloader, cost_function, optimizer, epochs):

# Switch pytorch to training mode. Some ML algos behave differently during training and evaluation.
    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for x_batch, y_batch in dataloader:

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward propagation
            prediction = model(x_batch)

            # Calculate loss
            cost = cost_function(prediction, y_batch)

            # Backpropagation
            cost.backward()

            # Update parameters
            optimizer.step()

            # Accumulate batch loss
            total_loss += cost.item()

        # Average loss across all batches
        average_loss = total_loss / len(dataloader)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {average_loss}")


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, dataloader, cost_function):
# Switches pytorch to evaluation (prediction) mode when testing the model on new data
    model.eval()

    total_loss = 0

# Tells pytorch to not store gradients
    with torch.no_grad():

        for x_batch, y_batch in dataloader:

            prediction = model(x_batch)

            cost = cost_function(prediction, y_batch)

            total_loss += cost.item()

    average_loss = total_loss / len(dataloader)

    return average_loss


# ============================================================
# MAIN
# ============================================================

'''In this example out training_data_loader and evaluation_data_loader will be the same
since we are treating our single dataset as both, however when training and evaluation data
will be different we will make different data_loaders for both.'''
train_data_loader = create_dataloader(
    x1,
    y1,
    batch_size=2,
    shuffle=True
)

model = create_model(2, 5, 4, 1)

cost_function = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)


# Training the model
fit(
    model,
    train_data_loader,
    cost_function,
    optimizer,
    epochs=700
)


# Evaluate
loss = evaluate(
    model,
    train_data_loader,
    cost_function
)

print(f"Evaluation loss: {loss}")


# Predict
x_new = torch.tensor([[1, 1]], dtype=torch.float32)

model.eval()

with torch.no_grad():
    prediction = model(x_new)

print(f"Prediction: {prediction}")