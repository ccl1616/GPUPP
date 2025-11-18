# GPUPP - GPU Power Profiler

A clean, modular pipeline for monitoring and recording GPU power usage for each layer during PyTorch model inference.

## Features

✅ **Per-layer or Per-kernel power consumption tracking** - Monitor energy consumption per unit   
✅ **CSV export** - Save detailed metrics for analysis  
✅ **Summary statistics** - Get aggregated stats per layer 

## Quick Start

### Installation

This project was tested on Cloudlab c240g5 with P100 GPU. The env file can be used to quickly create a conpatible environment. NVIDIA GPU with NVML support is required.
```bash
# make sure you have conda installed
$ conda env create -f environment_p100.yml
$ conda activate p100

# login Huggingface if running example 2 & 4
$ hf auth login
```

### Repo Structure

```bash
GPUPP/
├── environment_p100.yml        # env file for conda
├── common.py                   # shared by both layer and kernel profilers
├── layer_power_profiler.py     # layer profiler
├── kernel_power_profiler.py    # kernel profiler
├── examples.py                 # 📌 examples of how to use profilers
└── README.md
```

### How to use
```bash
python examples.py <example ID> # available example ID: 1-4
```

### Example options
1. Profile a dummy model with LayerPowerProfiler
2. Profile a dummy model with KernelPowerProfiler
3. Profile LLaMA model during text generation with LayerPowerProfiler 📌 Huggingface credentials required
4. Profile LLaMA model during text generation with KernelPowerProfiler 📌 Huggingface credentials required

### Example results
```
$ python examples.py 1
============================================================
Example 1: Layer Profiler (Simple Forward Pass)
============================================================
Running warmup (Layer Profiler)...
Running 1 profiling run(s) (Layer Profiler)...

================================================================================
LAYER PROFILING SUMMARY
================================================================================

GPU: Tesla P100-PCIE-12GB

Total Execution Time: 0.26 ms
Total Energy Consumed: 0.01 J
Average Power Draw: 30.21 W

Top 10 Energy Consumers:
layer_name  exec_id  duration_ms  avg_power_w  energy_j
         0        0     0.114679       30.207  0.003464
         2        2     0.071764       30.207  0.002168
         1        1     0.069380       30.207  0.002096
================================================================================
Layer results saved to layer_demo_profile.csv
Top 10 layer energy consumers saved to layer_demo_profile_top10_energy.csv
```

## Methodology
### LayerPowerProfiler

The `LayerPowerProfiler` tracks GPU power consumption at the **PyTorch module (layer) level** during model inference. It uses PyTorch's forward hook mechanism to instrument each layer in the model and measure power usage synchronously.

**How it works:**
- Registers pre-forward and post-forward hooks on each module in the model
- Takes synchronous power readings using NVML at the start and end of each layer's execution
- Calculates metrics including duration, average power, and energy consumption per layer
- Supports both simple forward passes and text generation workflows (via `profile_generation()`)(reference example 1 and example 3)

**Metrics tracked:**
- `duration_ms`: Execution time for each layer
- `avg_power_w`: Average power draw during layer execution
- `energy_j`: Approximate energy consumption (power × time)
- `start_power_w` / `end_power_w`: Power readings at layer boundaries

**Use cases:**
- Identify energy-intensive layers in your model
- Compare power efficiency across different model architectures
- Analyze power consumption patterns during text generation
- Optimize model design for energy efficiency

The profiler automatically saves detailed results to CSV and generates a separate file with the top 10 energy-consuming layers for quick analysis.

### KernelPowerProfiler

The `KernelPowerProfiler` tracks GPU power consumption at the **CUDA kernel level**, providing insights into which specific GPU operations consume the most energy. It combines PyTorch's profiler for kernel timing with high-frequency power sampling to estimate energy consumption per kernel.

**How it works:**
- Uses `torch.profiler` to capture microsecond-accurate CUDA kernel execution timelines
- Runs a high-frequency power polling thread (1ms intervals) that continuously samples GPU power using NVML
- Integrates power samples over each kernel's duration to estimate energy consumption
- Groups results by kernel name/type to identify the most energy-intensive operations
- Supports profiling of forward passes (reference example 2 and example 4)

**Metrics tracked:**
- `kernel_name`: Name/type of the CUDA kernel
- `duration_ms`: Execution time for each kernel (from torch.profiler)
- `avg_power_w`: Average power draw during kernel execution (estimated)
- `energy_j`: Approximated energy consumption (integrated from power samples)

**Use cases:**
- Identify energy-intensive CUDA kernels in model
- Optimize kernel-level operations for power efficiency
- Compare power consumption across different kernel implementations
- Understand the relationship between kernel execution time and energy consumption
- Fine-tune model operations at the lowest level of abstraction

**Key differences from LayerPowerProfiler:**
- **Granularity**: Kernel-level (individual CUDA operations) vs. layer-level (PyTorch modules)
- **Sampling method**: High-frequency asynchronous polling vs. synchronous boundary readings
- **Accuracy**: Approximated energy through integration vs. direct power measurements at boundaries
- **Use case**: Low-level optimization vs. architectural analysis

The profiler automatically saves detailed results to CSV and generates a separate file with the top 10 energy-consuming kernel types for quick analysis.
