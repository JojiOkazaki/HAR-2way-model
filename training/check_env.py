import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.version.cuda.startswith("12"), "Unexpected CUDA version"
