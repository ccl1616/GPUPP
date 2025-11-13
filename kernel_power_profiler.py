"""
Kernel Power Profiler
=====================
Profiles GPU power consumption per CUDA Kernel using torch.profiler and a 
high-frequency NVML power polling thread.
"""

import torch
import torch.profiler
import pandas as pd
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from common import PowerLoggerThread, PowerSample, GPUMonitor # Use shared utilities

# Set a very low sleep time for high-frequency polling
POLL_INTERVAL_MS = 1 

@dataclass
class KernelMetrics:
    """Data class for storing per-kernel metrics."""
    kernel_name: str
    exec_id: int
    duration_ms: float
    avg_power_w: float
    energy_j: float  # Integrated energy consumption
    
    def to_dict(self) -> Dict:
        return asdict(self)

class KernelPowerProfiler:
    """
    Main class for profiling power consumed by individual CUDA kernels.
    Uses torch.profiler for kernel timeline and a dedicated thread for power logging.
    """
    
    def __init__(self, model: torch.nn.Module, device_index: int = 0):
        self.model = model
        self.device_index = device_index
        self.gpu_monitor = GPUMonitor(device_index)
        self.kernel_results: List[KernelMetrics] = []
        self.exec_counter = 0

    def _get_kernel_timeline(self, inputs: Any) -> List[Dict]:
        """
        Uses torch.profiler to capture microsecond-accurate kernel events.
        
        Returns:
            List of kernel event dictionaries with timing and name info.
        """
        print("  Capturing CUDA timeline with torch.profiler...")
        
        # Use high-resolution perf_counter as a common CPU anchor time
        cpu_start_time = time.perf_counter() 
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # Use record_shapes to get better context, although performance impact is higher
            record_shapes=False, 
            profile_memory=False,
            # Schedule controls when to start/stop the detailed tracing
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=None 
        ) as p:
            # Perform a full run (or multiple runs) to fill the trace
            for _ in range(4): # 1 wait, 1 warmup, 2 active
                with torch.no_grad():
                    _ = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
                p.step()

        # Export trace data
        # filter_fn keeps only the relevant kernel events
        kernel_events = p.key_averages(group_by_input_shape=False).table(
            sort_by="cuda_time_total", row_limit=-1, header=True
        ).split('\n')
        
        # Simple parsing logic (can be improved with JSON export from profiler)
        kernel_data = []
        for line in kernel_events[4:]: # Skip header lines
            parts = line.strip().split()
            if len(parts) < 5 or parts[0].startswith('Total'): continue
            
            # The profiler usually reports 'cuda_time_total' and 'self_cuda_time_total'
            # We are extracting the kernel name from the first column
            name = parts[0]
            # Prof. output uses "us" (microseconds), convert to ms
            avg_duration_us = float(parts[2].replace(',', '')) 
            
            # NOTE: We are NOT getting start/end times here, only averages.
            # To get accurate start/end times per kernel *launch*, we would need to 
            # parse the raw JSON trace (e.g., p.export_chrome_trace()). 
            # For this simplified example, we'll use the average duration 
            # as a placeholder for the single kernel launch time.
            
            # A full implementation would parse the detailed trace to get 
            # event-specific start/end times (time_gpu_us).
            
            kernel_data.append({
                'name': name,
                'duration_ms': avg_duration_us / 1000.0,
            })
            
        print(f"  Captured {len(kernel_data)} unique kernel events (Averages).")
        return kernel_data


    def _integrate_power_log(self, 
                             kernel_events: List[Dict], 
                             power_log: List[PowerSample], 
                             start_time_sync: float) -> List[KernelMetrics]:
        """
        Approximates energy consumption by integrating power samples over kernel duration.
        
        NOTE: This is highly simplified as the torch.profiler simple output does not 
        give individual kernel launch times. In a real scenario, this would use 
        the raw trace to align specific kernel start/end times (microsecond) 
        with the power log (millisecond).
        """
        
        # Filter power log to the active duration
        if not power_log:
            print("Warning: Power log is empty. Check NVML setup/permissions.")
            return []

        # Find the time span of the power log relative to the start_time_sync
        log_start_time = power_log[0].timestamp
        log_end_time = power_log[-1].timestamp
        total_duration = log_end_time - log_start_time
        
        print(f"  Power Log duration: {total_duration * 1000:.2f}ms")
        
        # --- Simplified Energy Calculation (Placeholder) ---
        # Since we only have *average* kernel times, we will simplify the calculation:
        # 1. Calculate total energy over the full log duration.
        # 2. Distribute that energy based on the average duration of each kernel type.
        
        # Total Energy Approximation over the log period
        total_energy_j = 0.0
        for i in range(len(power_log) - 1):
            p1 = power_log[i]
            p2 = power_log[i+1]
            avg_p = (p1.power_w + p2.power_w) / 2
            dt = p2.timestamp - p1.timestamp
            total_energy_j += avg_p * dt
            
        print(f"  Total Integrated Energy over log period: {total_energy_j:.2f} J")
        
        
        # Total duration of all captured kernel averages
        total_kernel_duration_ms = sum(e['duration_ms'] for e in kernel_events)
        
        # Distribute energy proportionally based on time
        results: List[KernelMetrics] = []
        for i, event in enumerate(kernel_events):
            kernel_duration_ms = event['duration_ms']
            
            # Simple proportional energy distribution (highly inaccurate in real life, but necessary for placeholder)
            if total_kernel_duration_ms > 0:
                energy_share = (kernel_duration_ms / total_kernel_duration_ms) * total_energy_j
            else:
                energy_share = 0.0
                
            # Assume constant power during its average execution time (W = J/s)
            avg_power = energy_share / (kernel_duration_ms / 1000.0) if kernel_duration_ms > 0 else 0.0
            
            results.append(KernelMetrics(
                kernel_name=event['name'],
                exec_id=i,
                duration_ms=kernel_duration_ms,
                avg_power_w=avg_power,
                energy_j=energy_share
            ))

        self.exec_counter = len(results)
        return results


    def profile(self, 
                inputs: Any, 
                num_runs: int = 1,
                warmup: bool = True) -> List[KernelMetrics]:
        """Performs kernel profiling."""
        
        if warmup:
             print("Running warmup (Kernel Profiler)...")
             with torch.no_grad():
                _ = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
        
        # 1. Start Power Polling Thread
        power_thread = PowerLoggerThread(self.device_index, POLL_INTERVAL_MS)
        power_thread.start()
        
        # 2. Capture Kernel Timeline
        # Anchor the profiling to a CPU time
        profiling_start_time = time.perf_counter()
        try:
            kernel_timeline = self._get_kernel_timeline(inputs)
        finally:
            # 3. Stop Power Polling Thread
            power_thread.stop()
            power_thread.join()
        
        power_log = power_thread.get_log()
        
        # 4. Integrate Data
        self.kernel_results = self._integrate_power_log(
            kernel_timeline, 
            power_log, 
            profiling_start_time
        )
        
        print(f"Profiling complete. Collected {len(self.kernel_results)} kernel measurements.")
        return self.kernel_results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        if not self.kernel_results: return pd.DataFrame()
        data = [metrics.to_dict() for metrics in self.kernel_results]
        return pd.DataFrame(data)

    def print_summary(self):
        # Implementation of print_summary for Kernels
        if not self.kernel_results:
            print("No kernel profiling results available.")
            return

        df = self.get_results_dataframe()
        gpu_info = self.gpu_monitor.get_gpu_info()
        
        print("\n" + "#"*80)
        print("KERNEL POWER PROFILING SUMMARY (Approximated)")
        print("#"*80)
        
        if gpu_info:
            print(f"\nGPU: {gpu_info.get('name', 'Unknown')}")
            # print memory info...
        
        total_time = df['duration_ms'].sum()
        total_energy = df.groupby('kernel_name')['energy_j'].sum().sum()
        
        print(f"\nTotal Kernel Execution Time: {total_time:.2f} ms")
        print(f"Total Approximated Kernel Energy: {total_energy:.2f} J")
        
        # Group by kernel name to show total energy and time per kernel type
        summary_by_type = df.groupby('kernel_name').agg(
            total_duration_ms=('duration_ms', 'sum'),
            total_energy_j=('energy_j', 'sum'),
            avg_power_w=('avg_power_w', 'mean')
        ).sort_values(by='total_energy_j', ascending=False).head(10).round(4)
        
        print("\nTop 10 Kernel Types by Total Energy Consumption:")
        print(summary_by_type.to_string())
        print("#"*80)