import torch
import time

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU instead.")

# Create random tensors on GPU
print("Allocating tensors on GPU...")
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

# Perform matrix multiplication
print("Performing matrix multiplication on GPU...")
for i in range(100):
    c = torch.mm(a, b)
    torch.cuda.synchronize()  # Ensure computation is finished

print("Computation done! Check tegrastats now.")

# Keep the script running so you can monitor GPU usage
time.sleep(30)  # Keeps GPU allocated for 30 seconds
