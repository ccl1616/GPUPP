"""
Example Usage Scripts for Layer Power Profiler
==============================================
Various examples showing how to use the LayerPowerProfiler
"""

import torch
from layer_power_profiler import LayerPowerProfiler


# Example 1: Profile a simple model with forward pass
def example_simple_forward():
    """Profile a simple CNN model"""
    print("\n" + "="*60)
    print("Example 1: Simple Forward Pass")
    print("="*60)
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 10)
    ).cuda()
    
    # Create random input
    inputs = torch.randn(32, 512).cuda()
    
    # Profile
    profiler = LayerPowerProfiler(model)
    results = profiler.profile(inputs, num_runs=3, warmup=True)
    
    # Show results
    profiler.print_summary()
    profiler.save_results("simple_forward_profile.csv")
    
    # Note: The save_results method automatically creates a top 10 CSV file
    # You can also manually save top 10 power consumers:
    # profiler.save_top10_power_consumers("simple_forward_profile.csv")


# Example 2: Profile LLaMA text generation
def example_llama_generation():
    """Profile LLaMA model during text generation"""
    print("\n" + "="*60)
    print("Example 2: LLaMA Text Generation")
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
    profiler.save_results("llama_generation_profile.csv")
    # This automatically creates llama_generation_profile_top10.csv
    
    # Get summary stats
    summary = profiler.get_summary_stats()
    summary.to_csv("llama_generation_summary.csv")


# Example 3: Compare different batch sizes
def example_batch_size_comparison():
    """Profile with different batch sizes"""
    print("\n" + "="*60)
    print("Example 3: Batch Size Comparison")
    print("="*60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    profiler = LayerPowerProfiler(model)
    
    batch_sizes = [1, 2, 4, 8]
    prompt = "The quick brown fox"
    
    for batch_size in batch_sizes:
        print(f"\n--- Profiling batch size: {batch_size} ---")
        
        # Create batched input
        prompts = [prompt] * batch_size
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        # Profile
        results = profiler.profile(inputs, num_runs=1, warmup=True)
        
        # Save results
        profiler.save_results(f"batch_{batch_size}_profile.csv")
        
        # Quick stats
        df = profiler.get_results_dataframe()
        total_energy = df['energy_j'].sum()
        total_time = df['duration_ms'].sum()
        print(f"Total Time: {total_time:.2f} ms")
        print(f"Total Energy: {total_energy:.2f} J")


# Example 4: Analyze specific layer types
def example_layer_type_analysis():
    """Analyze power consumption by layer type"""
    print("\n" + "="*60)
    print("Example 4: Layer Type Analysis")
    print("="*60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import pandas as pd
    
    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Profile
    profiler = LayerPowerProfiler(model)
    results = profiler.profile(inputs, warmup=True)
    
    # Analyze by layer type
    df = profiler.get_results_dataframe()
    
    # Extract layer type from layer name
    def get_layer_type(name):
        if 'attention' in name.lower():
            return 'Attention'
        elif 'mlp' in name.lower() or 'feed_forward' in name.lower():
            return 'MLP'
        elif 'norm' in name.lower():
            return 'Normalization'
        elif 'embed' in name.lower():
            return 'Embedding'
        else:
            return 'Other'
    
    df['layer_type'] = df['layer_name'].apply(get_layer_type)
    
    # Group by layer type
    type_summary = df.groupby('layer_type').agg({
        'duration_ms': ['sum', 'mean'],
        'avg_power_w': 'mean',
        'energy_j': 'sum',
    }).round(4)
    
    print("\nPower Consumption by Layer Type:")
    print(type_summary)
    
    type_summary.to_csv("layer_type_analysis.csv")


# Example 5: Track power over time
def example_power_timeline():
    """Create a timeline of power consumption"""
    print("\n" + "="*60)
    print("Example 5: Power Timeline")
    print("="*60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import pandas as pd
    
    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    prompt = "Write a story about"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Profile
    profiler = LayerPowerProfiler(model)
    results = profiler.profile_generation(inputs, max_new_tokens=50, warmup=True)
    
    # Create timeline
    df = profiler.get_results_dataframe()
    df = df.sort_values('start_time')
    
    # Normalize time to start at 0
    df['time_from_start_ms'] = (df['start_time'] - df['start_time'].min()) * 1000
    
    # Save timeline
    timeline = df[['time_from_start_ms', 'layer_name', 'duration_ms', 'avg_power_w', 'energy_j']]
    timeline.to_csv("power_timeline.csv", index=False)
    
    print("\nPower timeline saved to power_timeline.csv")
    print(f"Total execution time: {df['time_from_start_ms'].max():.2f} ms")


# Example 6: Minimal usage
def example_minimal():
    """Minimal example for quick profiling"""
    print("\n" + "="*60)
    print("Example 6: Minimal Usage")
    print("="*60)
    
    # Your model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 10)
    ).cuda()
    
    # Your input
    inputs = torch.randn(16, 100).cuda()
    
    # Profile - just 3 lines!
    profiler = LayerPowerProfiler(model)
    profiler.profile(inputs)
    profiler.print_summary()


if __name__ == "__main__":
    # Run the example you want
    import sys
    
    examples = {
        '1': ('Simple Forward Pass', example_simple_forward),
        '2': ('LLaMA Generation', example_llama_generation),
        '3': ('Batch Size Comparison', example_batch_size_comparison),
        '4': ('Layer Type Analysis', example_layer_type_analysis),
        '5': ('Power Timeline', example_power_timeline),
        '6': ('Minimal Usage', example_minimal),
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
        print("Or run all examples:")
        
        # Run minimal example as default
        example_minimal()
