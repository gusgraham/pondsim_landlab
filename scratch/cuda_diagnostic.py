import numba
from numba import cuda
import sys

print(f"Numba version: {numba.__version__}")
print(f"Python version: {sys.version}")

try:
    available = cuda.is_available()
    print(f"CUDA available: {available}")
    
    if available:
        devices = cuda.list_devices()
        for i, dev in enumerate(devices):
            print(f"Device {i}: {dev.name}")
            
        @cuda.jit
        def test_kernel():
            pass
            
        test_kernel[1, 1]()
        print("Kernel launch: SUCCESS")
    else:
        print("CUDA not available. Checking for missing DLLs or drivers...")
        # Check if we can find NVVM
        try:
            from numba.cuda.cudadrv import nvvm
            print(f"NVVM Lib found: {nvvm.NVVM().is_available()}")
        except Exception as e:
            print(f"NVVM check failed: {e}")
            
except Exception as e:
    print(f"Unexpected error during CUDA check: {e}")
    import traceback
    traceback.print_exc()
