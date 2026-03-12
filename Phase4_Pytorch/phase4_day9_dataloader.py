"""
PHASE 4 — PyTorch Fundamentals
Day 9: DataLoader

Concepts:
- batching data
- shuffling datasets
- iterating through batches
- simulating training loops
"""

import torch
from torch.utils.data import TensorDataset, DataLoader

print("PHASE 4 — DAY 9")
print("DataLoader - Batching & Shuffling")
print("-" * 50)

# --------------------------------------------------
# 1. Create input features
# --------------------------------------------------

print("\n1. Creating dataset")

X = torch.tensor([
    [1,2],
    [2,3],
    [3,4],
    [4,5]
], dtype=torch.float32)

y = torch.tensor([3,5,7,9], dtype=torch.float32)

dataset = TensorDataset(X, y)

print("Dataset length:", len(dataset))


# --------------------------------------------------
# 2. Create DataLoader
# --------------------------------------------------

print("\n2. Creating DataLoader")

loader = DataLoader(dataset, batch_size=2)

print("Batch size:", 2)


# --------------------------------------------------
# 3. Iterate through batches
# --------------------------------------------------

print("\n3. Iterating through batches")

for batch_X, batch_y in loader:
    print("Batch features:", batch_X)
    print("Batch labels:", batch_y)


# --------------------------------------------------
# 4. Shuffle data
# --------------------------------------------------

print("\n4. DataLoader with shuffling")

loader_shuffled = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_X, batch_y in loader_shuffled:
    print("Batch features:", batch_X)
    print("Batch labels:", batch_y)


# --------------------------------------------------
# 5. Batch shapes
# --------------------------------------------------

print("\n5. Batch shapes")

for batch_X, batch_y in loader:
    print("Feature shape:", batch_X.shape)
    print("Label shape:", batch_y.shape)


# --------------------------------------------------
# 6. Simulated training loop
# --------------------------------------------------

print("\n6. Simulated training loop")

for epoch in range(2):

    print("\nEpoch:", epoch)

    for batch_X, batch_y in loader:
        print("Training batch:", batch_X)


# --------------------------------------------------
# 7. Image dataset example
# --------------------------------------------------

print("\n7. Image dataset example")

images = torch.rand(20,3,64,64)
labels = torch.randint(0,10,(20,))

image_dataset = TensorDataset(images, labels)

image_loader = DataLoader(image_dataset, batch_size=4)

for imgs, labs in image_loader:

    print("Image batch shape:", imgs.shape)
    print("Label batch shape:", labs.shape)

print("\nDay 9 completed successfully.")