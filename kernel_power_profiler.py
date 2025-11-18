"""
Kernel Power Profiler
=====================
Profiles GPU power consumption per CUDA Kernel using torch.profiler and a 
high-frequency NVML power polling thread.
"""

import torch
import torch.profiler
from torch.autograd import DeviceType
import pandas as pd
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import os
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
    # ... (init, profiling, and integration methods remain the same)
    def __init__(self, model: torch.nn.Module, device_index: int = 0):
        self.model = model
        self.device_index = device_index
        self.gpu_monitor = GPUMonitor(device_index)
        self.kernel_results: List[KernelMetrics] = []
        self.exec_counter = 0

    def _get_kernel_timeline(self, inputs: Any) -> List[Dict]:
        """
        Uses torch.profiler to capture microsecond-accurate kernel events (Averaged for simplicity).
        """
        print("  Capturing CUDA timeline with torch.profiler...")
        
        cpu_start_time = time.perf_counter() 
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False, 
            profile_memory=False,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=None 
        ) as p:
            for _ in range(4):
                with torch.no_grad():
                    _ = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
                p.step()

        # Get events sorted by CUDA time
        events = p.key_averages(group_by_input_shape=False)
        sorted_events = sorted(events, key=lambda evt: evt.cuda_time_total, reverse=True)
        
        kernel_data = []
        for evt in sorted_events:
            # Only process CUDA events (kernels) - check if CUDA time is non-zero
            if evt.cuda_time_total > 0:
                # cuda_time_total is in microseconds
                duration_us = evt.cuda_time_total
                kernel_data.append({
                    'name': evt.key,
                    'duration_ms': duration_us / 1000.0,
                })
        
        print(f"  Captured {len(kernel_data)} unique kernel events (Averages).")
        return kernel_data


    def _integrate_power_log(self, 
                             kernel_events: List[Dict], 
                             power_log: List[PowerSample], 
                             start_time_sync: float) -> List[KernelMetrics]:
        """
        Approximates energy consumption by integrating power samples over kernel duration 
        (Simplified proportional distribution).
        """
        if not power_log:
            print("Warning: Power log is empty. Check NVML setup/permissions.")
            return []

        log_start_time = power_log[0].timestamp
        log_end_time = power_log[-1].timestamp
        total_duration = log_end_time - log_start_time
        
        total_energy_j = 0.0
        for i in range(len(power_log) - 1):
            p1 = power_log[i]
            p2 = power_log[i+1]
            avg_p = (p1.power_w + p2.power_w) / 2
            dt = p2.timestamp - p1.timestamp
            total_energy_j += avg_p * dt
            
        total_kernel_duration_ms = sum(e['duration_ms'] for e in kernel_events)
        
        results: List[KernelMetrics] = []
        for i, event in enumerate(kernel_events):
            kernel_duration_ms = event['duration_ms']
            
            if total_kernel_duration_ms > 0:
                energy_share = (kernel_duration_ms / total_kernel_duration_ms) * total_energy_j
            else:
                energy_share = 0.0
                
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
        # ... (profiling orchestration remains the same)
        
        if warmup:
             print("Running warmup (Kernel Profiler)...")
             with torch.no_grad():
                _ = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
        
        power_thread = PowerLoggerThread(self.device_index, POLL_INTERVAL_MS)
        power_thread.start()
        
        profiling_start_time = time.perf_counter()
        try:
            kernel_timeline = self._get_kernel_timeline(inputs)
        finally:
            power_thread.stop()
            power_thread.join()
        
        power_log = power_thread.get_log()
        
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

    def save_results(self, filename: str = "kernel_power_profile.csv"):
        """Save detailed results to CSV file and automatically save top 10 energy consumers."""
        df = self.get_results_dataframe()
        
        if df.empty:
            print("No results to save.")
            return
        
        df.to_csv(filename, index=False)
        print(f"Kernel results saved to {filename}")
        
        # Automatically save top 10
        self.save_top10_energy_consumers(filename)
    
    def save_top10_energy_consumers(self, base_filename: str = "kernel_power_profile.csv"):
        """Save top 10 energy consumers to a separate CSV file."""
        df = self.get_results_dataframe()
        
        if df.empty:
            print("No results to save for top 10 energy consumers.")
            return
        
        # Group by kernel name and sum up energy
        type_summary = df.groupby('kernel_name').agg(
            total_energy_j=('energy_j', 'sum'),
            total_duration_ms=('duration_ms', 'sum'),
            avg_power_w=('avg_power_w', 'mean')
        ).reset_index()
        
        # Get top 10 energy consumers
        top_energy = type_summary.nlargest(10, 'total_energy_j')[['kernel_name', 'total_duration_ms', 'avg_power_w', 'total_energy_j']]
        
        # Generate filename for top 10 CSV
        base_name = os.path.splitext(base_filename)[0]
        top10_filename = f"{base_name}_top10_energy.csv"
        
        # Save to CSV
        top_energy.to_csv(top10_filename, index=False)
        print(f"Top 10 kernel energy consumers saved to {top10_filename}")

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