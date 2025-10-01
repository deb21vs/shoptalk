import torch, platform
print("cuda?", torch.cuda.is_available())
print("mps?", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
print("platform:", platform.platform())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
