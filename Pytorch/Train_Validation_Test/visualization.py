import matplotlib.pyplot as plt

from train_validation import train_losses, val_losses


epochs = range(len(train_losses))


plt.figure(figsize=(10, 6))

plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()