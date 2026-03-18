"""
PHASE 5 — CNNs & TorchVision
Day 1: Conv2d + ReLU

Concepts:
- convolutional layers
- feature maps
- channels
- kernel/filter
- ReLU activation
- stacking CNN layers
"""

import torch
import torch.nn as nn

print("PHASE 5 — DAY 1")
print("CNN Layers (Conv2d + ReLU)")
print("-" * 50)


# --------------------------------------------------
# 1. Create fake image tensor
# --------------------------------------------------

print("\n1. Creating fake image")

# (batch, channels, height, width)
img = torch.rand(1, 3, 64, 64)

print("Image shape:", img.shape)


# --------------------------------------------------
# 2. Create convolution layer
# --------------------------------------------------

print("\n2. Creating Conv2D layer")

conv = nn.Conv2d(
    in_channels=3,
    out_channels=8,
    kernel_size=3
)

print(conv)


# --------------------------------------------------
# 3. Apply convolution
# --------------------------------------------------

print("\n3. Applying convolution")

output = conv(img)

print("Output shape:", output.shape)


# --------------------------------------------------
# 4. Apply ReLU activation
# --------------------------------------------------

print("\n4. Applying ReLU")

relu = nn.ReLU()

activated = relu(output)

print("Activated shape:", activated.shape)


# --------------------------------------------------
# 5. Combine Conv + ReLU
# --------------------------------------------------

print("\n5. Conv + ReLU block")

conv = nn.Conv2d(3, 8, 3)
relu = nn.ReLU()

x = conv(img)
x = relu(x)

print("Block output shape:", x.shape)


# --------------------------------------------------
# 6. Stack multiple layers
# --------------------------------------------------

print("\n6. Stacking layers")

conv1 = nn.Conv2d(3, 8, 3)
conv2 = nn.Conv2d(8, 16, 3)

relu = nn.ReLU()

x = conv1(img)
x = relu(x)

x = conv2(x)
x = relu(x)

print("Stacked output shape:", x.shape)


# --------------------------------------------------
# 7. Using nn.Sequential
# --------------------------------------------------

print("\n7. Using Sequential")

model = nn.Sequential(
    nn.Conv2d(3, 8, 3),
    nn.ReLU(),
    nn.Conv2d(8, 16, 3),
    nn.ReLU()
)

output = model(img)

print("Sequential output shape:", output.shape)


# --------------------------------------------------
# 8. Inspect convolution weights
# --------------------------------------------------

print("\n8. Inspecting weights")

conv = nn.Conv2d(3, 8, 3)

print("Weight shape:", conv.weight.shape)


# --------------------------------------------------
# 9. Build simple CNN class
# --------------------------------------------------

print("\n9. Building custom CNN class")

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


model = SimpleCNN()

output = model(img)

print("Custom CNN output shape:", output.shape)


print("\nDay 1 completed successfully.")