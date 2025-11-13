import torch
import time
import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import threading

# GPU monitoring
try:
    # Attempt to import pynvml, which is necessary for power reading
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, Exception) as e:
    GPU_AVAILABLE = False
    print(f"Warning: GPU monitoring (pynvml) not available: {e}")

@dataclass
class PowerSample:
    """High-frequency power reading entry."""
    timestamp: float  # System time (e.g., time.perf_counter())
    power_w: float    # Power usage in watts

class GPUMonitor:
    """Handles GPU power monitoring using NVML (synchronous power fetch)."""
    
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.available = GPU_AVAILABLE
        self.handle = None
        
        if self.available:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception as e:
                print(f"Warning: Could not get GPU handle: {e}")
                self.available = False
    
    def get_power(self) -> float:
        """Get current GPU power usage in watts."""
        if not self.available or self.handle is None:
            return 0.0
        
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return power_mw / 1000.0  # Convert to watts
        except Exception as e:
            # Handle permissions errors gracefully
            if 'NVML_ERROR_NO_PERMISSION' in str(e):
                 print("Error: NVML permission denied. Try running with 'sudo'.")
            return 0.0
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get static GPU information."""
        if not self.available or self.handle is None:
            return {}
        
        try:
            name = pynvml.nvmlDeviceGetName(self.handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            return {
                'name': name,
                'memory_total_gb': memory_info.total / (1024**3),
                'memory_used_gb': memory_info.used / (1024**3),
            }
        except Exception as e:
            return {}


class PowerLoggerThread(threading.Thread):
    """
    Runs in a separate thread to continuously poll and log power usage 
    at high frequency. Used by KernelPowerProfiler.
    """
    def __init__(self, device_index: int = 0, poll_interval_ms: int = 1):
        super().__init__()
        self.monitor = GPUMonitor(device_index)
        self.power_log: List[PowerSample] = []
        self._stop_event = threading.Event()
        self.poll_interval_s = poll_interval_ms / 1000.0
        self.daemon = True # Allow program to exit if this thread is running

    def run(self):
        """High-frequency polling loop."""
        print(f"PowerLoggerThread started (Interval: {self.poll_interval_s*1000:.1f}ms)...")
        while not self._stop_event.is_set():
            # Ensure we only poll if monitoring is available
            if self.monitor.available:
                try:
                    # Capture power and a high-resolution timestamp
                    power = self.monitor.get_power()
                    timestamp = time.perf_counter()
                    self.power_log.append(PowerSample(timestamp, power))
                except Exception as e:
                    # Suppress constant logging during thread run
                    pass
            
            time.sleep(self.poll_interval_s)

    def stop(self):
        """Signals the thread to stop."""
        self._stop_event.set()
        print("PowerLoggerThread stopped.")

    def get_log(self) -> List[PowerSample]:
        """Returns the collected power log."""
        return self.power_log