import traceback
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.models.deeplabv3 import DeepLabV3

m = DeepLabV3(21, pretrained=False)

# Try different input sizes
for size in [256, 512, 224]:
    try:
        o = m(torch.randn(1, 3, size, size))
        print(f"Size {size}: OK -> {o.shape}")
    except Exception as e:
        print(f"Size {size}: FAILED -> {e}")
        if size == 256:
            traceback.print_exc()
