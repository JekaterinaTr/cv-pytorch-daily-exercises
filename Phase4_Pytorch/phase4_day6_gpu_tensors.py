"""
PHASE 4 — PyTorch Fundamentals
Day 6: GPU Tensors (CUDA)

Concepts:
- CUDA availability
- device management
- moving tensors CPU ↔ GPU
- GPU tensor math
"""

import torch
import time

print("PHASE 4 — DAY 6")
print("GPU Tensors with CUDA")
print("-" * 50)

# --------------------------------------------------
# 1. Check if CUDA is available
# --------------------------------------------------

print("\n1. Checking CUDA availability")

print("CUDA available:", torch.cuda.is_available())


# --------------------------------------------------
# 2. Define device
# --------------------------------------------------

print("\n2. Defining device")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# 3. Create tensor on CPU
# --------------------------------------------------

print("\n3. Tensor on CPU")

tensor = torch.tensor([1, 2, 3, 4])

print("Tensor:", tensor)
print("Device:", tensor.device)


# --------------------------------------------------
# 4. Move tensor to GPU
# --------------------------------------------------

print("\n4. Move tensor to GPU")

tensor_gpu = tensor.to(device)

print("Tensor device:", tensor_gpu.device)


# --------------------------------------------------
# 5. Create tensor directly on device
# --------------------------------------------------

print("\n5. Create tensor on device")

tensor2 = torch.rand(5).to(device)

print("Tensor:", tensor2)
print("Device:", tensor2.device)


# --------------------------------------------------
# 6. GPU tensor math
# --------------------------------------------------

print("\n6. GPU Tensor Operations")

a = torch.rand(3).to(device)
b = torch.rand(3).to(device)

c = a + b

print("a:", a)
print("b:", b)
print("a + b:", c)
print("Device:", c.device)


# --------------------------------------------------
# 7. Move tensor back to CPU
# --------------------------------------------------

print("\n7. Move tensor back to CPU")

tensor_cpu = c.cpu()

print("Device after move:", tensor_cpu.device)


# --------------------------------------------------
# 8. Convert to NumPy
# --------------------------------------------------

print("\n8. Convert tensor to NumPy")

np_array = tensor_cpu.numpy()

print("NumPy array:", np_array)
print("Type:", type(np_array))


# --------------------------------------------------
# 9. Timing example CPU
# --------------------------------------------------

print("\n9. CPU vs GPU timing example")

size = 2000

cpu_tensor = torch.rand(size, size)

start = time.time()
cpu_result = cpu_tensor * 2
cpu_time = time.time() - start

print("CPU time:", cpu_time)


# --------------------------------------------------
# 10. GPU timing
# --------------------------------------------------

gpu_tensor = cpu_tensor.to(device)

start = time.time()
gpu_result = gpu_tensor * 2

if device.type == "cuda":
    torch.cuda.synchronize()

gpu_time = time.time() - start

print("GPU time:", gpu_time)

print("\nDay 6 completed successfully.")