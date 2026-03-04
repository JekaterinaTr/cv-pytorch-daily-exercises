"""
PHASE 4 — PyTorch Fundamentals
Day 3: Tensor Math Operations

Concepts:
- element-wise operations
- sum
- mean
- broadcasting
- row/column reduction
"""

import torch

print("=== PHASE 4 - DAY 3: TENSOR MATH ===\n")

# --------------------------------------------------
# 1. Element-wise Addition
# --------------------------------------------------

print("1. Element-wise Addition")

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = a + b

print("a:", a)
print("b:", b)
print("a + b:", c)
print()

# --------------------------------------------------
# 2. Element-wise Multiplication
# --------------------------------------------------

print("2. Element-wise Multiplication")

c2 = a * b

print("a * b:", c2)
print()

# --------------------------------------------------
# 3. Tensor Sum
# --------------------------------------------------

print("3. Sum of Tensor")

t = torch.tensor([1, 2, 3, 4])

total = torch.sum(t)

print("Tensor:", t)
print("Sum:", total)
print()

# --------------------------------------------------
# 4. Tensor Mean (Average)
# --------------------------------------------------

print("4. Mean (Average)")

mean = torch.mean(t.float())

print("Mean:", mean)
print()

# --------------------------------------------------
# 5. Broadcasting Example
# --------------------------------------------------

print("5. Broadcasting")

t2 = torch.tensor([1, 2, 3])

result = t2 + 10

print("Original:", t2)
print("After +10:", result)
print()

# --------------------------------------------------
# 6. Matrix Sum and Mean
# --------------------------------------------------

print("6. Matrix Math")

matrix = torch.tensor([
    [1, 2],
    [3, 4]
])

print("Matrix:\n", matrix)
print("Sum:", torch.sum(matrix))
print("Mean:", torch.mean(matrix.float()))
print()

# --------------------------------------------------
# 7. Row and Column Operations
# --------------------------------------------------

print("7. Row and Column Reduction")

mat2 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

# Sum per row
row_sums = torch.sum(mat2, dim=1)

# Sum per column
col_sums = torch.sum(mat2, dim=0)

print("Matrix:\n", mat2)
print("Row sums:", row_sums)
print("Column sums:", col_sums)
print()

print("Day 3 Completed Successfully ✅")