"""
GPU Layer Power Profiler
========================
A clean pipeline for monitoring and recording GPU power usage for each layer 
during model inference.

Features:
- Per-layer power consumption tracking
- Execution time measurement
- CSV export with detailed metrics
- Support for any PyTorch model
- Easy-to-use API

Requirements:
    pip install torch transformers pynvml pandas

Usage:
    python layer_power_profiler.py
    
Or use as a library:
    from layer_power_profiler import LayerPowerProfiler
    profiler = LayerPowerProfiler(model)
    results = profiler.profile(inputs)
"""

import torch
import time
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, Exception) as e:
    GPU_AVAILABLE = False
    print(f"Warning: GPU monitoring not available: {e}")


@dataclass
class LayerMetrics:
    """Data class for storing per-layer metrics"""
    layer_name: str
    exec_id: int
    start_time: float
    end_time: float
    duration_ms: float
    start_power_w: float
    end_power_w: float
    avg_power_w: float
    energy_j: float  # Approximate energy consumption
    
    def to_dict(self) -> Dict:
        return asdict(self)


class GPUMonitor:
    """Handle GPU power monitoring using NVML"""
    
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
        """Get current GPU power usage in watts"""
        if not self.available or self.handle is None:
            return 0.0
        
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return power_mw / 1000.0  # Convert to watts
        except Exception as e:
            print(f"Warning: Could not read power: {e}")
            return 0.0
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        if not self.available or self.handle is None:
            return {}
        
        try:
            name = pynvml.nvmlDeviceGetName(self.handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            return {
                'name': name,
                'memory_total_gb': memory_info.total / (1024**3),
                'memory_used_gb': memory_info.used / (1024**3),
                'memory_free_gb': memory_info.free / (1024**3),
            }
        except Exception as e:
            print(f"Warning: Could not get GPU info: {e}")
            return {}


class LayerPowerProfiler:
    """Main profiler class for tracking per-layer power consumption"""
    
    def __init__(self, model: torch.nn.Module, device_index: int = 0):
        self.model = model
        self.gpu_monitor = GPUMonitor(device_index)
        self.exec_counter = 0
        self.layer_results: List[LayerMetrics] = []
        self.layer_start_data: Dict[str, Dict] = {}
        self.module_names: Dict[torch.nn.Module, str] = {}
        self.hook_handles: List = []
        
        # Map modules to names
        self._build_module_map()
    
    def _build_module_map(self):
        """Build mapping from module objects to their names"""
        for name, module in self.model.named_modules():
            self.module_names[module] = name
    
    def _pre_forward_hook(self, module: torch.nn.Module, input: Any):
        """Hook called before each layer's forward pass"""
        layer_name = self.module_names.get(module, "unknown")
        
        # Skip the root model module
        if module == self.model:
            return
        
        start_time = time.time()
        start_power = self.gpu_monitor.get_power()
        
        self.layer_start_data[layer_name] = {
            'exec_id': self.exec_counter,
            'start_time': start_time,
            'start_power': start_power,
        }
        
        self.exec_counter += 1
    
    def _post_forward_hook(self, module: torch.nn.Module, input: Any, output: Any):
        """Hook called after each layer's forward pass"""
        layer_name = self.module_names.get(module, "unknown")
        
        # Skip the root model module
        if module == self.model:
            return
        
        if layer_name not in self.layer_start_data:
            return
        
        end_time = time.time()
        end_power = self.gpu_monitor.get_power()
        
        start_data = self.layer_start_data.pop(layer_name)
        
        # Calculate metrics
        duration = end_time - start_data['start_time']
        duration_ms = duration * 1000
        avg_power = (start_data['start_power'] + end_power) / 2
        energy_j = avg_power * duration  # Approximate energy in joules
        
        # Store metrics
        metrics = LayerMetrics(
            layer_name=layer_name,
            exec_id=start_data['exec_id'],
            start_time=start_data['start_time'],
            end_time=end_time,
            duration_ms=duration_ms,
            start_power_w=start_data['start_power'],
            end_power_w=end_power,
            avg_power_w=avg_power,
            energy_j=energy_j
        )
        
        self.layer_results.append(metrics)
    
    def _register_hooks(self):
        """Register forward hooks on all modules"""
        self.hook_handles.clear()
        
        for name, module in self.model.named_modules():
            if module == self.model:
                continue
            
            pre_handle = module.register_forward_pre_hook(self._pre_forward_hook)
            post_handle = module.register_forward_hook(self._post_forward_hook)
            
            self.hook_handles.extend([pre_handle, post_handle])
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def profile(self, 
                inputs: Any,
                num_runs: int = 1,
                warmup: bool = True) -> List[LayerMetrics]:
        """
        Profile the model with given inputs
        
        Args:
            inputs: Model inputs (typically dict with input_ids, attention_mask, etc.)
            num_runs: Number of profiling runs (results will include all runs)
            warmup: Whether to do a warmup run before profiling
            
        Returns:
            List of LayerMetrics for all layers across all runs
        """
        self.layer_results.clear()
        self.exec_counter = 0
        
        # Warmup run
        if warmup:
            print("Running warmup...")
            with torch.no_grad():
                if isinstance(inputs, dict):
                    _ = self.model(**inputs)
                else:
                    _ = self.model(inputs)
        
        # Profiling runs
        print(f"Running {num_runs} profiling run(s)...")
        for run in range(num_runs):
            self._register_hooks()
            
            try:
                with torch.no_grad():
                    if isinstance(inputs, dict):
                        _ = self.model(**inputs)
                    else:
                        _ = self.model(inputs)
            finally:
                self._remove_hooks()
            
            if num_runs > 1:
                print(f"  Completed run {run + 1}/{num_runs}")
        
        print(f"Profiling complete. Collected {len(self.layer_results)} layer measurements.")
        return self.layer_results
    
    def profile_generation(self,
                          inputs: Any,
                          max_new_tokens: int = 50,
                          warmup: bool = True,
                          **generate_kwargs) -> List[LayerMetrics]:
        """
        Profile the model during text generation
        
        Args:
            inputs: Model inputs
            max_new_tokens: Maximum number of tokens to generate
            warmup: Whether to do a warmup run
            **generate_kwargs: Additional arguments for model.generate()
            
        Returns:
            List of LayerMetrics for all layers
        """
        self.layer_results.clear()
        self.exec_counter = 0
        
        # Check if model has generate method
        if not hasattr(self.model, 'generate'):
            raise ValueError("Model does not have a generate() method")
        
        # Warmup
        if warmup:
            print("Running warmup...")
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=1)
        
        # Profiling
        print("Profiling generation...")
        self._register_hooks()
        
        try:
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **generate_kwargs)
        finally:
            self._remove_hooks()
        
        print(f"Profiling complete. Collected {len(self.layer_results)} layer measurements.")
        return self.layer_results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        if not self.layer_results:
            return pd.DataFrame()
        
        data = [metrics.to_dict() for metrics in self.layer_results]
        return pd.DataFrame(data)
    
    def save_results(self, filename: str = "layer_power_profile.csv"):
        """Save results to CSV file and automatically save top 10 power consumers"""
        df = self.get_results_dataframe()
        
        if df.empty:
            print("No results to save.")
            return
        
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        # Automatically save top 10 power consumers
        self.save_top10_power_consumers(filename)
    
    def save_top10_power_consumers(self, base_filename: str = "layer_power_profile.csv"):
        """Save top 10 power consumers to a separate CSV file"""
        df = self.get_results_dataframe()
        
        if df.empty:
            print("No results to save for top 10 power consumers.")
            return
        
        # Get top 10 power consumers with exec_id included
        top_power = df.nlargest(10, 'avg_power_w')[['layer_name', 'exec_id', 'avg_power_w', 'duration_ms', 'energy_j']]
        
        # Generate filename for top 10 CSV
        import os
        base_name = os.path.splitext(base_filename)[0]
        top10_filename = f"{base_name}_top10.csv"
        
        # Save to CSV
        top_power.to_csv(top10_filename, index=False)
        print(f"Top 10 power consumers saved to {top10_filename}")
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics per layer"""
        df = self.get_results_dataframe()
        
        if df.empty:
            return pd.DataFrame()
        
        summary = df.groupby('layer_name').agg({
            'duration_ms': ['mean', 'std', 'min', 'max'],
            'avg_power_w': ['mean', 'std', 'min', 'max'],
            'energy_j': ['sum', 'mean', 'std'],
        }).round(4)
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of profiling results"""
        if not self.layer_results:
            print("No profiling results available.")
            return
        
        df = self.get_results_dataframe()
        
        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        
        # GPU info
        gpu_info = self.gpu_monitor.get_gpu_info()
        if gpu_info:
            print(f"\nGPU: {gpu_info.get('name', 'Unknown')}")
            print(f"Memory Used: {gpu_info.get('memory_used_gb', 0):.2f} GB / "
                  f"{gpu_info.get('memory_total_gb', 0):.2f} GB")
        
        # Overall stats
        total_time = df['duration_ms'].sum()
        total_energy = df['energy_j'].sum()
        avg_power = df['avg_power_w'].mean()
        
        print(f"\nTotal Execution Time: {total_time:.2f} ms")
        print(f"Total Energy Consumed: {total_energy:.2f} J")
        print(f"Average Power Draw: {avg_power:.2f} W")
        
        # Top power consumers
        print_top_power = True
        if print_top_power:
            print("\nTop 10 Power Consumers:")
            top_power = df.nlargest(10, 'avg_power_w')[['layer_name', 'exec_id', 'avg_power_w', 'duration_ms', 'energy_j']]
            print(top_power.to_string(index=False))
        
        # Top time consumers
        print_top_time = False
        if print_top_time:
            print("\nTop 10 Time Consumers:")
            top_time = df.nlargest(10, 'duration_ms')[['layer_name', 'exec_id', 'duration_ms', 'avg_power_w', 'energy_j']]
            print(top_time.to_string(index=False))
        
        print("\n" + "="*80)


# Example usage
def main():
    """Example usage with LLaMA model"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("Loading model...")
    model_id = "meta-llama/Llama-2-7b-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Prepare input
    prompt = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Create profiler
    profiler = LayerPowerProfiler(model)
    
    # Profile generation
    results = profiler.profile_generation(inputs, max_new_tokens=50)
    
    # Print summary
    profiler.print_summary()
    
    # Save detailed results
    profiler.save_results("layer_power_profile.csv")
    
    # Get summary statistics
    summary = profiler.get_summary_stats()
    print("\nPer-Layer Summary Statistics:")
    print(summary)
    summary.to_csv("layer_power_summary.csv")


if __name__ == "__main__":
    main()
