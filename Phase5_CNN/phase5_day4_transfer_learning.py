"""
PHASE 5 — CNNs & TorchVision
Day 4: Pretrained Models & Transfer Learning

Concepts:
- pretrained models (ResNet)
- evaluation mode
- modifying final layer
- freezing layers
- transfer learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

print("PHASE 5 — DAY 4")
print("Pretrained Models & Transfer Learning")
print("-" * 50)


# --------------------------------------------------
# 1. Load pretrained ResNet18
# --------------------------------------------------

print("\n1. Loading pretrained ResNet18")

model = models.resnet18(pretrained=True)

print("Model loaded")


# --------------------------------------------------
# 2. Set to evaluation mode
# --------------------------------------------------

print("\n2. Setting evaluation mode")

model.eval()


# --------------------------------------------------
# 3. Forward pass with fake image
# --------------------------------------------------

print("\n3. Testing forward pass")

img = torch.rand(1, 3, 224, 224)

output = model(img)

print("Output shape:", output.shape)


# --------------------------------------------------
# 4. Get predicted class
# --------------------------------------------------

print("\n4. Prediction")

_, predicted = torch.max(output, 1)

print("Predicted class index:", predicted.item())


# --------------------------------------------------
# 5. Modify final layer (2 classes)
# --------------------------------------------------

print("\n5. Modifying final layer")

model.fc = nn.Linear(512, 2)

print("New final layer:", model.fc)


# --------------------------------------------------
# 6. Freeze all layers
# --------------------------------------------------

print("\n6. Freezing layers")

for param in model.parameters():
    param.requires_grad = False

# Unfreeze final layer
for param in model.fc.parameters():
    param.requires_grad = True

print("Only final layer will be trained")


# --------------------------------------------------
# 7. Create fake dataset
# --------------------------------------------------

print("\n7. Creating dataset")

X = torch.rand(20, 3, 224, 224)
y = torch.randint(0, 2, (20,))

print("Dataset shape:", X.shape)


# --------------------------------------------------
# 8. Loss & optimizer
# --------------------------------------------------

print("\n8. Loss & optimizer")

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.fc.parameters(), lr=0.01)


# --------------------------------------------------
# 9. Training loop (only FC layer)
# --------------------------------------------------

print("\n9. Training")

for epoch in range(5):

    outputs = model(X)

    loss = loss_fn(outputs, y)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print("Epoch:", epoch, "Loss:", loss.item())


# --------------------------------------------------
# 10. Predictions & accuracy
# --------------------------------------------------

print("\n10. Evaluation")

outputs = model(X)

_, predicted = torch.max(outputs, 1)

correct = (predicted == y).sum().item()
accuracy = correct / len(y)

print("Accuracy:", accuracy)


print("\nDay 4 completed successfully.")