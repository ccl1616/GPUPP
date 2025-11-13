"""
Layer Power Profiler
====================
Profiles GPU power consumption per PyTorch Module (Layer) using synchronous 
forward hooks and the shared GPUMonitor.
"""

import torch
import time
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from common import GPUMonitor # Use shared utility

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


class LayerPowerProfiler:
    """Main profiler class for tracking per-layer power consumption using hooks."""
    
    def __init__(self, model: torch.nn.Module, device_index: int = 0):
        self.model = model
        # Use shared GPUMonitor
        self.gpu_monitor = GPUMonitor(device_index) 
        self.exec_counter = 0
        self.layer_results: List[LayerMetrics] = []
        self.layer_start_data: Dict[str, Dict] = {}
        self.module_names: Dict[torch.nn.Module, str] = {}
        self.hook_handles: List = []
        
        self._build_module_map()
    
    def _build_module_map(self):
        """Build mapping from module objects to their names"""
        for name, module in self.model.named_modules():
            self.module_names[module] = name
    
    def _pre_forward_hook(self, module: torch.nn.Module, input: Any):
        """Hook called before each layer's forward pass"""
        layer_name = self.module_names.get(module, "unknown")
        
        if module == self.model: return
        
        # Synchronous power read at layer start
        start_power = self.gpu_monitor.get_power()
        start_time = time.time()
        
        self.layer_start_data[layer_name] = {
            'exec_id': self.exec_counter,
            'start_time': start_time,
            'start_power': start_power,
        }
        self.exec_counter += 1
    
    def _post_forward_hook(self, module: torch.nn.Module, input: Any, output: Any):
        """Hook called after each layer's forward pass"""
        layer_name = self.module_names.get(module, "unknown")
        
        if module == self.model or layer_name not in self.layer_start_data: return
        
        # Synchronous power read at layer end
        end_time = time.time()
        end_power = self.gpu_monitor.get_power()
        
        start_data = self.layer_start_data.pop(layer_name)
        
        # Calculate metrics
        duration = end_time - start_data['start_time']
        duration_ms = duration * 1000
        avg_power = (start_data['start_power'] + end_power) / 2
        energy_j = avg_power * duration  # Approximate energy
        
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
        # ... (hook registration methods remain the same)
        self.hook_handles.clear()
        for name, module in self.model.named_modules():
            if module == self.model: continue
            pre_handle = module.register_forward_pre_hook(self._pre_forward_hook)
            post_handle = module.register_forward_hook(self._post_forward_hook)
            self.hook_handles.extend([pre_handle, post_handle])
    
    def _remove_hooks(self):
        # ... (hook removal method remains the same)
        for handle in self.hook_handles: handle.remove()
        self.hook_handles.clear()
    
    def profile(self, inputs: Any, num_runs: int = 1, warmup: bool = True) -> List[LayerMetrics]:
        self.layer_results.clear()
        self.exec_counter = 0
        
        if warmup:
            print("Running warmup (Layer Profiler)...")
            with torch.no_grad():
                _ = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
        
        print(f"Running {num_runs} profiling run(s) (Layer Profiler)...")
        for run in range(num_runs):
            self._register_hooks()
            try:
                with torch.no_grad():
                    _ = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
            finally:
                self._remove_hooks()
        
        return self.layer_results
    
    # --- Other utility methods (profile_generation, get_results_dataframe, etc.) ---
    # These methods remain largely the same, using self.layer_results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        if not self.layer_results: return pd.DataFrame()
        data = [metrics.to_dict() for metrics in self.layer_results]
        return pd.DataFrame(data)

    def print_summary(self):
        # Implementation of print_summary remains the same
        if not self.layer_results:
            print("No profiling results available.")
            return
        
        df = self.get_results_dataframe()
        gpu_info = self.gpu_monitor.get_gpu_info()
        
        print("\n" + "="*80)
        print("LAYER PROFILING SUMMARY")
        print("="*80)
        
        if gpu_info:
            print(f"\nGPU: {gpu_info.get('name', 'Unknown')}")
            # print memory info...
        
        # Overall stats
        total_time = df['duration_ms'].sum()
        total_energy = df['energy_j'].sum()
        avg_power = df['avg_power_w'].mean()
        
        print(f"\nTotal Execution Time: {total_time:.2f} ms")
        print(f"Total Energy Consumed: {total_energy:.2f} J")
        print(f"Average Power Draw: {avg_power:.2f} W")
        
        print("\nTop 10 Energy Consumers:")
        top_energy = df.nlargest(10, 'energy_j')[['layer_name', 'exec_id', 'duration_ms', 'avg_power_w', 'energy_j']]
        print(top_energy.to_string(index=False))
        print("="*80)


def main():
    """Example usage for LayerPowerProfiler."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("LayerPowerProfiler: Loading LLaMA 2...")
    model_id = "meta-llama/Llama-2-7b-hf"
    
    # Dummy load for demonstration if actual GPU is not available
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
    except Exception as e:
        print(f"Could not load large model for demo: {e}. Using dummy model.")
        model = torch.nn.Sequential(torch.nn.Linear(10, 10)).cuda()
        inputs = torch.randn(1, 10).cuda()

    profiler = LayerPowerProfiler(model)
    profiler.profile(inputs, num_runs=1, warmup=True)
    profiler.print_summary()

if __name__ == "__main__":
    main()