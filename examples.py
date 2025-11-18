"""
Example Usage Scripts for Layer and Kernel Power Profiler
=========================================================
Various examples showing how to use the LayerPowerProfiler and KernelPowerProfiler
"""

import torch
import sys

# Import both profilers
from layer_power_profiler import LayerPowerProfiler
from kernel_power_profiler import KernelPowerProfiler


# Helper function for dummy model creation
def create_dummy_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 10)
    ).cuda()
    inputs = torch.randn(32, 512).cuda()
    return model, inputs

# Example 1: Profile a simple model with LayerPowerProfiler
def example_layer_forward():
    print("\n" + "="*60)
    print("Example 1: Layer Profiler (Simple Forward Pass)")
    print("="*60)
    
    model, inputs = create_dummy_model()
    
    # Profile
    profiler = LayerPowerProfiler(model)
    profiler.profile(inputs, num_runs=1, warmup=True)
    
    # Show results
    profiler.print_summary()
    profiler.save_results("layer_demo_profile.csv")


# Example 2: Profile the same model with KernelPowerProfiler
def example_kernel_forward():
    print("\n" + "#"*60)
    print("Example 2: Kernel Profiler (Simple Forward Pass)")
    print("#"*60)
    
    model, inputs = create_dummy_model()
    
    # Profile
    profiler = KernelPowerProfiler(model)
    profiler.profile(inputs, num_runs=1, warmup=True)
    
    # Show results
    profiler.print_summary()
    profiler.save_results("kernel_demo_profile.csv")

# Example 3: Profile LLaMA model during text generation with LayerPowerProfiler
def example_llama_generation_layer():
    """Profile LLaMA model during text generation"""
    print("\n" + "="*60)
    print("Example 3: LLaMA Text Generation with LayerPowerProfiler")
    print("="*60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Load model
    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Prepare input
    prompt = "Explain how neural networks work in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Profile
    profiler = LayerPowerProfiler(model)
    results = profiler.profile_generation(
        inputs,
        max_new_tokens=100,
        warmup=True,
        do_sample=True,
        temperature=0.7
    )
    
    # Show results
    profiler.print_summary()
    profiler.save_results("llama_generation_level_profile.csv")
    # This automatically creates llama_generation_profile_top10.csv
    
    # Get summary stats
    summary = profiler.get_summary_stats()
    summary.to_csv("llama_generation_level_summary.csv")

# Example 4: Profile LLaMA model during text generation with KernelPowerProfiler
def example_llama_generation_kernel():
    """Profile LLaMA model during text generation with the Kernel Profiler"""
    print("\n" + "#"*60)
    print("Example 4: LLaMA Text Generation with KernelPowerProfiler")
    print("#"*60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    try:
        model_id = "google/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        # Prepare input
        prompt = "Explain why optimizing kernels is important for power efficiency:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Profile (Note: We use the simple forward profile method for now)
        profiler = KernelPowerProfiler(model)
        # For full generation profiling, the profile method must be adapted 
        # to call model.generate inside the torch.profiler context.
        profiler.profile(inputs, num_runs=1, warmup=True) 
        
        # Show results
        profiler.print_summary()
        
    except Exception as e:
        print(f"Skipping LLaMA example due to load error (Requires GPU/HF setup): {e}")


if __name__ == "__main__":
    
    examples = {
        '1': ('Layer Profiler (Simple)', example_layer_forward),
        '2': ('Kernel Profiler (Simple)', example_kernel_forward),
        '3': ('LLaMA Layer Profiler (Text Generation)', example_llama_generation_layer),
        '4': ('LLaMA Kernel Profiler (Text Generation)', example_llama_generation_kernel),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            examples[choice][1]()
        else:
            print(f"Invalid example number. Choose from: {', '.join(examples.keys())}")
    else:
        print("Available Examples:")
        print("-" * 60)
        for num, (name, _) in examples.items():
            print(f"{num}. {name}")
        print("\nUsage: python examples.py <example_number>")
        
        # Run simple kernel example as default
        example_kernel_forward()