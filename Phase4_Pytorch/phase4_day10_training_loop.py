"""
PHASE 4 — PyTorch Fundamentals
Day 10: Loss Functions & Optimizers

Concepts:
- neural network model
- loss functions
- optimizers
- forward pass
- backpropagation
- training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("PHASE 4 — DAY 10")
print("Loss Functions & Optimizers")
print("-" * 50)


# --------------------------------------------------
# 1. Create dummy dataset
# --------------------------------------------------

print("\n1. Creating dataset")

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])

print("Inputs:")
print(X)

print("Targets:")
print(y)


# --------------------------------------------------
# 2. Create model
# --------------------------------------------------

print("\n2. Creating linear model")

model = nn.Linear(1,1)

print(model)


# --------------------------------------------------
# 3. Define loss function
# --------------------------------------------------

print("\n3. Loss function")

loss_fn = nn.MSELoss()

print("Using Mean Squared Error")


# --------------------------------------------------
# 4. Define optimizer
# --------------------------------------------------

print("\n4. Optimizer")

optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Optimizer: SGD")


# --------------------------------------------------
# 5. Forward pass
# --------------------------------------------------

print("\n5. Forward pass")

predictions = model(X)

print("Predictions:")
print(predictions)


# --------------------------------------------------
# 6. Compute loss
# --------------------------------------------------

print("\n6. Compute loss")

loss = loss_fn(predictions, y)

print("Loss:", loss.item())


# --------------------------------------------------
# 7. Backpropagation
# --------------------------------------------------

print("\n7. Backpropagation")

loss.backward()

print("Gradients computed")


# --------------------------------------------------
# 8. Update model parameters
# --------------------------------------------------

print("\n8. Optimizer step")

optimizer.step()

print("Model weights updated")


# --------------------------------------------------
# 9. Reset gradients
# --------------------------------------------------

print("\n9. Zero gradients")

optimizer.zero_grad()

print("Gradients cleared")


# --------------------------------------------------
# 10. Full training loop
# --------------------------------------------------

print("\n10. Training loop")

for epoch in range(50):

    predictions = model(X)

    loss = loss_fn(predictions, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    print("Epoch:", epoch, "Loss:", loss.item())


# --------------------------------------------------
# 11. Test trained model
# --------------------------------------------------

print("\n11. Testing trained model")

test = torch.tensor([[5.0]])

prediction = model(test)

print("Input:", test.item())
print("Prediction:", prediction.item())

print("\nDay 10 completed successfully.")