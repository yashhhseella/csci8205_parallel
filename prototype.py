import openai
import json

openai.api_key = "Use your own API key"

### GPT Middleware Functions

def ir_to_json(ir_metadata: dict) -> str:
    """
    Convert IR metadata to a JSON string.
    """
    return json.dumps(ir_metadata)

def generate_prompt_from_ir(ir_metadata: dict) -> str:
    prompt = (
        "Matrix Multiplication Workload:\n"
        "Evaluate the following IR parameters and provide a cost estimation and tuning recommendations "
        "for optimal performance. Return a JSON object with keys 'predicted_cost' (numeric) and 'recommendation' (string):\n"
        f"shape: {ir_metadata.get('shape')}\n"
        f"dtype: {ir_metadata.get('dtype')}\n"
        f"layout: {ir_metadata.get('layout')}\n"
        f"op_type: {ir_metadata.get('op_type')}\n"
        f"gpu_arch: {ir_metadata.get('gpu_arch')}\n"
    )
    return prompt

def query_gpt(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def model_output_to_json(model_output: str) -> dict:
    try:
        return json.loads(model_output)
    except json.JSONDecodeError as e:
        print("Error decoding model output as JSON:", e)
        return {}

def get_cost_estimation(ir_metadata: dict) -> dict:
    prompt = generate_prompt_from_ir(ir_metadata)
    print("Generated GPT prompt:")
    print(prompt)
    model_response = query_gpt(prompt)
    print("GPT model response:")
    print(model_response)
    cost_info = model_output_to_json(model_response)
    return cost_info

def should_benchmark_configuration(ir_metadata: dict, threshold: float = 100.0) -> bool:
    cost_info = get_cost_estimation(ir_metadata)
    predicted_cost = cost_info.get("predicted_cost", None)
    if predicted_cost is None:
        print("No predicted cost returned; proceeding to benchmark.")
        return True
    print(f"GPT predicted cost: {predicted_cost}, recommendation: {cost_info.get('recommendation', '')}")
    return predicted_cost <= threshold

### Main Simulation

if __name__ == "__main__":
    # Example IR metadata extracted for a matrix multiplication workload.
    ir_metadata = {
        "shape": [1024, 1024],
        "dtype": "float16",
        "layout": "strided",
        "op_type": "mm.py",  # Typically extracted from kernel name in Inductor
        "gpu_arch": "A100"
    }
    
    print("IR Metadata:")
    print(ir_metadata)
    
    # Convert IR metadata to JSON (for logging or further processing)
    ir_json = ir_to_json(ir_metadata)
    print("\nIR Metadata as JSON:")
    print(ir_json)
    
    # Decide whether to benchmark this configuration based on GPT cost estimation
    print("\nEvaluating configuration with GPT-based cost estimation...")
    if should_benchmark_configuration(ir_metadata):
        print("Benchmarking configuration...")
        # Call your existing benchmarking function here.
    else:
        print("Skipping configuration based on cost estimation.")