import torch
import traceback
import torchvision.models.segmentation as seg

m = seg.deeplabv3_resnet101(weights=None, num_classes=21)
x = torch.randn(1, 3, 256, 256)

try:
    o = m(x)
    print("OK:", o["out"].shape)
except Exception as e:
    tb = traceback.format_exc()
    with open("error.txt", "w") as f:
        f.write(tb)
    print("Error written to error.txt")
    print("Error:", type(e).__name__, str(e)[:200])
