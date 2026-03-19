"""
PHASE 5 — CNNs & TorchVision
Day 3: Training a CNN

Concepts:
- CNN model training
- CrossEntropyLoss
- forward pass
- backpropagation
- optimizer step
- prediction & accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("PHASE 5 — DAY 3")
print("Training a CNN")
print("-" * 50)


# --------------------------------------------------
# 1. Create fake dataset
# --------------------------------------------------

print("\n1. Creating dataset")

# 10 images, RGB, 64x64
X = torch.rand(10, 3, 64, 64)

# 10 labels (0–9 classes)
y = torch.randint(0, 10, (10,))

print("Input shape:", X.shape)
print("Labels:", y)


# --------------------------------------------------
# 2. Build CNN model
# --------------------------------------------------

print("\n2. Building CNN model")

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(16 * 60 * 60, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


model = CNN()

print(model)


# --------------------------------------------------
# 3. Define loss function
# --------------------------------------------------

print("\n3. Loss function")

loss_fn = nn.CrossEntropyLoss()

print("Using CrossEntropyLoss")


# --------------------------------------------------
# 4. Define optimizer
# --------------------------------------------------

print("\n4. Optimizer")

optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Using SGD")


# --------------------------------------------------
# 5. Training loop
# --------------------------------------------------

print("\n5. Training loop")

for epoch in range(10):

    # Forward pass
    outputs = model(X)

    # Compute loss
    loss = loss_fn(outputs, y)

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    # Reset gradients
    optimizer.zero_grad()

    print("Epoch:", epoch, "Loss:", loss.item())


# --------------------------------------------------
# 6. Predictions
# --------------------------------------------------

print("\n6. Predictions")

outputs = model(X)

_, predicted = torch.max(outputs, 1)

print("Predicted:", predicted)
print("Actual:", y)


# --------------------------------------------------
# 7. Accuracy
# --------------------------------------------------

print("\n7. Accuracy")

correct = (predicted == y).sum().item()
accuracy = correct / len(y)

print("Accuracy:", accuracy)


print("\nDay 3 completed successfully.")