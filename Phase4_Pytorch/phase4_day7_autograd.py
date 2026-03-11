"""
PHASE 4 — PyTorch Fundamentals
Day 7: Autograd (Automatic Gradients)

Concepts:
- requires_grad
- computation graphs
- backward()
- gradient accumulation
- disabling gradients
"""

import torch

print("PHASE 4 — DAY 7")
print("Autograd - Automatic Differentiation")
print("-" * 50)


# --------------------------------------------------
# 1. Create Tensor with Gradient Tracking
# --------------------------------------------------

print("\n1. Tensor with requires_grad")

x = torch.tensor(2.0, requires_grad=True)

print("Tensor x:", x)
print("Requires grad:", x.requires_grad)


# --------------------------------------------------
# 2. Perform Mathematical Operation
# --------------------------------------------------

print("\n2. Simple operation")

y = x ** 2

print("y = x^2")
print("y:", y)


# --------------------------------------------------
# 3. Compute Gradient
# --------------------------------------------------

print("\n3. Backpropagation")

y.backward()

print("Gradient dy/dx:", x.grad)


# --------------------------------------------------
# 4. More Complex Function
# --------------------------------------------------

print("\n4. Complex function")

x = torch.tensor(3.0, requires_grad=True)

y = x**3 + 2*x

y.backward()

print("x:", x)
print("Gradient:", x.grad)


# --------------------------------------------------
# 5. Chain Operations
# --------------------------------------------------

print("\n5. Chain operations")

x = torch.tensor(4.0, requires_grad=True)

y = x * 2
z = y ** 2

z.backward()

print("Result z:", z)
print("Gradient dz/dx:", x.grad)


# --------------------------------------------------
# 6. Vector Gradient Example
# --------------------------------------------------

print("\n6. Vector gradients")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x * 2

z = y.sum()

z.backward()

print("Tensor x:", x)
print("Gradients:", x.grad)


# --------------------------------------------------
# 7. Disable Gradient Tracking
# --------------------------------------------------

print("\n7. Disable gradients with no_grad")

x = torch.tensor(5.0, requires_grad=True)

with torch.no_grad():
    y = x * 3

print("Result:", y)
print("Requires grad:", y.requires_grad)


print("\nDay 7 completed successfully.")