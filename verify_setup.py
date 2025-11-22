import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f'  Number of GPUs: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        
        # Test GPU computation
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f'\n✓ GPU computation successful!')
        print(f'  Result shape: {z.shape}')
        print(f'  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')

except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")

try:
    import timm
    print(f"✓ timm version: {timm.__version__}")
except ImportError as e:
    print(f"✗ timm not installed: {e}")

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy not installed: {e}")

try:
    import scipy
    print(f"✓ SciPy version: {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy not installed: {e}")

try:
    import yaml
    print(f"✓ PyYAML installed")
except ImportError as e:
    print(f"✗ PyYAML not installed: {e}")

try:
    from PIL import Image
    print(f"✓ Pillow installed")
except ImportError as e:
    print(f"✗ Pillow not installed: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib not installed: {e}")

print("Setup verification complete!")