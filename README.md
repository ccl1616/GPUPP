# GPUPP - GPU Power Profiler

A clean, modular pipeline for monitoring and recording GPU power usage for each layer during PyTorch model inference.

## Features

✅ **Per-layer power consumption tracking** - Monitor power draw for each individual layer  
✅ **Execution time measurement** - Track how long each layer takes to execute  
✅ **Energy calculation** - Approximate energy consumption per layer  
✅ **CSV export** - Save detailed metrics for analysis  
✅ **Summary statistics** - Get aggregated stats per layer  
✅ **Easy-to-use API** - Simple interface for any PyTorch model  
✅ **Generation support** - Profile text generation tasks  

## Requirements

```bash
pip install torch transformers nvidia-ml-py pandas
```

**Note:** Requires NVIDIA GPU with NVML support

## Quick Start

### Installation

```bash
pip install torch transformers nvidia-ml-py pandas
```

### Execution Instructions

You have several options to run the profiler:

#### Option 1: Run Built-in Examples (Easiest to Get Started)

```bash
# See available examples
python examples.py

# Run a specific example (1-6)
python examples.py 6  # Minimal example
python examples.py 2  # LLaMA generation example
python examples.py 1  # Simple forward pass example
```

With sudo if you get permission errors:
```bash
sudo python examples.py 6
```

#### Option 2: Run the Main Script

The `layer_power_profiler.py` file has a complete LLaMA example in its `main()` function:

```bash
python layer_power_profiler.py
```

#### Option 3: Create Your Own Script (Recommended for Custom Use)

Create a new file (e.g., `my_profile.py`):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from layer_power_profiler import LayerPowerProfiler

# Load your model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Your prompt here", return_tensors="pt").to(model.device)

# Profile it!
profiler = LayerPowerProfiler(model)
profiler.profile_generation(inputs, max_new_tokens=50)
profiler.print_summary()
profiler.save_results("my_results.csv")
```

Then run:
```bash
python my_profile.py
```

#### Option 4: Use as a Library in Your Existing Code

Simply import and use:

```python
from layer_power_profiler import LayerPowerProfiler

# In your existing code where you have a model
profiler = LayerPowerProfiler(your_model)
profiler.profile(your_inputs)
profiler.save_results("profile.csv")
```

**Note about `sudo`:** Try running without `sudo` first. Only use `sudo` if you encounter permission errors when reading GPU power measurements.

### Minimal Example (3 lines!)

```python
from layer_power_profiler import LayerPowerProfiler

profiler = LayerPowerProfiler(model)
profiler.profile(inputs)
profiler.print_summary()
```

### Complete Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from layer_power_profiler import LayerPowerProfiler

# Load your model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare inputs
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)

# Profile the model
profiler = LayerPowerProfiler(model)
results = profiler.profile_generation(inputs, max_new_tokens=50)

# View and save results
profiler.print_summary()
profiler.save_results("power_profile.csv")
```

## API Reference

### LayerPowerProfiler

Main class for profiling GPU power consumption.

#### `__init__(model, device_index=0)`

Initialize the profiler.

**Parameters:**
- `model` (torch.nn.Module): The PyTorch model to profile
- `device_index` (int): GPU device index (default: 0)

#### `profile(inputs, num_runs=1, warmup=True)`

Profile the model with forward pass.

**Parameters:**
- `inputs`: Model inputs (tensor or dict)
- `num_runs` (int): Number of profiling runs
- `warmup` (bool): Whether to do warmup run

**Returns:**
- List[LayerMetrics]: Per-layer metrics for all runs

#### `profile_generation(inputs, max_new_tokens=50, warmup=True, **generate_kwargs)`

Profile the model during text generation.

**Parameters:**
- `inputs`: Model inputs
- `max_new_tokens` (int): Number of tokens to generate
- `warmup` (bool): Whether to do warmup run
- `**generate_kwargs`: Additional args for model.generate()

**Returns:**
- List[LayerMetrics]: Per-layer metrics

#### `save_results(filename="layer_power_profile.csv")`

Save detailed results to CSV.

#### `print_summary()`

Print formatted summary of profiling results.

#### `get_results_dataframe()`

Get results as pandas DataFrame.

**Returns:**
- pd.DataFrame: Results with all metrics

#### `get_summary_stats()`

Get summary statistics per layer.

**Returns:**
- pd.DataFrame: Aggregated statistics (mean, std, min, max)

### LayerMetrics

Data class storing per-layer measurements.

**Attributes:**
- `layer_name` (str): Name of the layer
- `exec_id` (int): Execution order ID
- `start_time` (float): Start timestamp
- `end_time` (float): End timestamp
- `duration_ms` (float): Execution duration in milliseconds
- `start_power_w` (float): Power at start in watts
- `end_power_w` (float): Power at end in watts
- `avg_power_w` (float): Average power in watts
- `energy_j` (float): Approximate energy consumed in joules

## Output Files

### Detailed Profile CSV

The main output file (`layer_power_profile.csv`) contains:

| Column | Description |
|--------|-------------|
| layer_name | Name of the layer |
| exec_id | Execution order (0, 1, 2, ...) |
| start_time | Unix timestamp when layer started |
| end_time | Unix timestamp when layer ended |
| duration_ms | How long the layer took (ms) |
| start_power_w | GPU power at layer start (watts) |
| end_power_w | GPU power at layer end (watts) |
| avg_power_w | Average power during layer (watts) |
| energy_j | Approximate energy consumed (joules) |

### Summary Statistics CSV

When you call `get_summary_stats().to_csv()`, you get aggregated statistics per layer:

- Mean, std, min, max for duration
- Mean, std, min, max for power
- Sum, mean, std for energy

## Usage Examples

### Example 1: Profile Simple Model

```python
from layer_power_profiler import LayerPowerProfiler
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 10)
).cuda()

inputs = torch.randn(32, 512).cuda()

profiler = LayerPowerProfiler(model)
profiler.profile(inputs, num_runs=3)
profiler.print_summary()
```

### Example 2: Profile Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from layer_power_profiler import LayerPowerProfiler

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Write a poem about", return_tensors="pt").to(model.device)

profiler = LayerPowerProfiler(model)
profiler.profile_generation(inputs, max_new_tokens=100, temperature=0.7)
profiler.save_results("generation_profile.csv")
```

### Example 3: Analyze by Layer Type

```python
from layer_power_profiler import LayerPowerProfiler
import pandas as pd

# ... load model and profile ...

df = profiler.get_results_dataframe()

# Categorize layers
def get_layer_type(name):
    if 'attention' in name.lower():
        return 'Attention'
    elif 'mlp' in name.lower():
        return 'MLP'
    elif 'norm' in name.lower():
        return 'Normalization'
    return 'Other'

df['layer_type'] = df['layer_name'].apply(get_layer_type)

# Analyze by type
type_summary = df.groupby('layer_type').agg({
    'duration_ms': 'sum',
    'avg_power_w': 'mean',
    'energy_j': 'sum'
})

print(type_summary)
```

### Example 4: Compare Batch Sizes

```python
batch_sizes = [1, 2, 4, 8]
results = {}

for batch_size in batch_sizes:
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", 
                      padding=True).to(model.device)
    
    profiler.profile(inputs, num_runs=1)
    
    df = profiler.get_results_dataframe()
    results[batch_size] = {
        'total_time_ms': df['duration_ms'].sum(),
        'total_energy_j': df['energy_j'].sum(),
        'avg_power_w': df['avg_power_w'].mean()
    }

print(pd.DataFrame(results).T)
```

### Example 5: Create Power Timeline

```python
profiler.profile_generation(inputs, max_new_tokens=50)

df = profiler.get_results_dataframe()
df = df.sort_values('start_time')

# Normalize time
df['time_from_start_ms'] = (df['start_time'] - df['start_time'].min()) * 1000

# Plot or save timeline
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['time_from_start_ms'], df['avg_power_w'])
plt.xlabel('Time (ms)')
plt.ylabel('Power (W)')
plt.title('GPU Power Consumption Timeline')
plt.savefig('power_timeline.png')
```

## Understanding the Output

### Summary Output

When you call `print_summary()`, you'll see:

```
================================================================================
PROFILING SUMMARY
================================================================================

GPU: NVIDIA H100 80GB HBM3
Memory Used: 14.23 GB / 80.00 GB

Total Execution Time: 2450.67 ms
Total Energy Consumed: 485.12 J
Average Power Draw: 198.05 W

Top 10 Power Consumers:
                    layer_name  avg_power_w  duration_ms  energy_j
model.layers.15.self_attn.q_proj    245.32       12.45      3.05
model.layers.15.self_attn.k_proj    243.87       11.89      2.90
...

Top 10 Time Consumers:
                    layer_name  duration_ms  avg_power_w  energy_j
model.layers.15.self_attn.q_proj       12.45       245.32      3.05
...
```

### Key Metrics Explained

- **Total Execution Time**: Sum of all layer execution times
- **Total Energy**: Sum of energy consumed by all layers (J = W × s)
- **Average Power**: Mean power draw across all layers
- **Energy per layer**: Power × Duration (approximate)

## Advanced Usage

### Custom GPU Device

```python
# Use second GPU
profiler = LayerPowerProfiler(model, device_index=1)
```

### Multiple Runs for Statistical Analysis

```python
# Run 10 times and analyze variance
results = profiler.profile(inputs, num_runs=10)
summary = profiler.get_summary_stats()
print(summary[('duration_ms', 'std')])  # See variance
```

### Integration with Your Training Loop

```python
profiler = LayerPowerProfiler(model)

for epoch in range(num_epochs):
    # Profile one batch per epoch
    if epoch % 10 == 0:
        profiler.profile(next(iter(dataloader)))
        profiler.save_results(f"profile_epoch_{epoch}.csv")
```

## Limitations

1. **Power measurement accuracy**: Power readings are sampled at layer boundaries. Very fast layers (<1ms) may not have accurate power measurements.

2. **Energy approximation**: Energy is calculated as (start_power + end_power) / 2 × duration. This is approximate.

3. **Overhead**: The profiler adds small overhead due to hook execution and power sampling.

4. **NVML requirement**: Requires NVIDIA GPU with NVML support (most modern NVIDIA GPUs).

## Troubleshooting

### "GPU monitoring not available"

Make sure:
1. You have NVIDIA GPU
2. pynvml is installed: `pip install pynvml`
3. NVIDIA drivers are properly installed

### "Could not get GPU handle"

Try:
```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
print(pynvml.nvmlDeviceGetName(handle))
```

If this fails, check your NVIDIA driver installation.

### Permission issues

On some systems, reading GPU power requires elevated privileges:
```bash
sudo python your_script.py
```

## Comparison with Original Script

This refactored pipeline offers:

- ✅ Cleaner, modular design
- ✅ Easier to use (3-line usage)
- ✅ Better organized code with classes
- ✅ More comprehensive documentation
- ✅ Additional features (summary stats, multiple runs)
- ✅ Can be used as a library
- ✅ Better error handling
- ✅ More output options

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Areas for improvement:
- More accurate energy modeling
- Support for multi-GPU profiling
- Visualization tools
- Integration with profiling tools like torch.profiler

