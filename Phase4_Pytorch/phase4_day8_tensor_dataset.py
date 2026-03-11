"""
PHASE 4 — PyTorch Fundamentals
Day 8: TensorDataset

Concepts:
- building datasets
- pairing inputs and labels
- dataset indexing
- dataset iteration
"""

import torch
from torch.utils.data import TensorDataset

print("PHASE 4 — DAY 8")
print("TensorDataset - Building Training Datasets")
print("-" * 50)

# --------------------------------------------------
# 1. Create input data
# --------------------------------------------------

print("\n1. Creating input features")

X = torch.tensor([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
], dtype=torch.float32)

print("Features:")
print(X)


# --------------------------------------------------
# 2. Create labels
# --------------------------------------------------

print("\n2. Creating labels")

y = torch.tensor([3, 5, 7, 9], dtype=torch.float32)

print("Labels:")
print(y)


# --------------------------------------------------
# 3. Create TensorDataset
# --------------------------------------------------

print("\n3. Creating dataset")

dataset = TensorDataset(X, y)

print(dataset)


# --------------------------------------------------
# 4. Access dataset element
# --------------------------------------------------

print("\n4. Access first sample")

sample = dataset[0]

print("Sample:", sample)
print("Input:", sample[0])
print("Label:", sample[1])


# --------------------------------------------------
# 5. Iterate through dataset
# --------------------------------------------------

print("\n5. Iterating through dataset")

for data, label in dataset:
    print("Input:", data, "Label:", label)


# --------------------------------------------------
# 6. Dataset length
# --------------------------------------------------

print("\n6. Dataset size")

print("Length:", len(dataset))


# --------------------------------------------------
# 7. Image-style dataset example
# --------------------------------------------------

print("\n7. Image dataset example")

images = torch.rand(10, 3, 64, 64)  # 10 images
labels = torch.randint(0, 2, (10,))  # binary labels

image_dataset = TensorDataset(images, labels)

print("Dataset size:", len(image_dataset))

print("First sample shape:", image_dataset[0][0].shape)
print("First label:", image_dataset[0][1])


print("\nDay 8 completed successfully.")