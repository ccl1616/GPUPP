# Tesla P100 GPU Power Profiler Environment Setup

This directory contains environment configuration files specifically designed for Tesla P100 GPU compatibility.

## Hardware Compatibility

- **GPU**: Tesla P100-PCIE-12GB (CUDA capability 6.0)
- **CUDA**: 11.7
- **Python**: 3.10

## Quick Setup Options

### Option 1: Conda Environment (Recommended)

```bash
# Create environment from YAML file
conda env create -f environment_p100.yml

# Activate environment
conda activate gpu-power-profiler-p100

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Pip Requirements

```bash
# Install from requirements file
pip install -r requirements_p100.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Package Versions

| Package | Version | Notes |
|---------|---------|-------|
| PyTorch | 2.0.1+cu117 | Tesla P100 compatible |
| Transformers | 4.35.0 | Compatible with PyTorch 2.0.1 |
| Accelerate | 0.24.0 | Compatible with Transformers 4.35.0 |
| Pandas | 2.3.3 | Data processing |
| NumPy | 2.2.6 | Numerical computing |
| nvidia-ml-py | 13.580.82 | GPU monitoring |

## Why These Specific Versions?

1. **PyTorch 2.0.1+cu117**: Latest version that supports Tesla P100 (CUDA 6.0)
2. **Transformers 4.35.0**: Compatible with PyTorch 2.0.1, supports LLaMA models
3. **Accelerate 0.24.0**: Compatible with Transformers 4.35.0 for device mapping

## Testing the Setup

After installation, test with the examples:

```bash
# Test simple forward pass
python examples.py 1

# Test LLaMA generation (if you have the model)
python examples.py 2
```

## Troubleshooting

### CUDA Compatibility Issues
If you get CUDA kernel errors, ensure you're using the Tesla P100 compatible versions:
- PyTorch 2.0.1+cu117 (not newer versions)
- CUDA 11.7 (not 12.x)

### Transformers Import Errors
If AutoModelForCausalLM fails to import:
- Ensure PyTorch >= 2.0.1 is installed
- Check transformers version is 4.35.0 (not newer)

### Memory Issues
Tesla P100 has 12GB VRAM. For large models:
- Use `device_map="auto"` for automatic memory management
- Consider model quantization if needed

## Environment Files

- `environment_p100.yml`: Complete conda environment specification
- `requirements_p100.txt`: Pip requirements for manual installation
- `README_p100_setup.md`: This setup guide
