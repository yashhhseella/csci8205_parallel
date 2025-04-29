import os, json, openai

openai.api_key = os.environ["OPENAI_API_KEY"]

def ir_to_json(ir):
    return json.dumps(ir)

def generate_prompt_from_ir(ir):
    return (
        "Matrix Multiplication Workload:\n"
        "Evaluate the following IR parameters and provide a cost estimation and tuning recommendations\n"
        "Return a JSON object with keys 'predicted_cost' (numeric) and 'recommendation' (object with BLOCK_M, BLOCK_N, BLOCK_K):\n"
        f"shape: {ir['shape']}\n"
        f"dtype: {ir['dtype']}\n"
        f"layout: {ir['layout']}\n"
        f"op_type: {ir['op_type']}\n"
        f"gpu_arch: {ir['gpu_arch']}\n"
    )

def query_gpt(prompt):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        max_tokens=150,
        temperature=0.3
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {}

def get_cost_estimation(ir):
    return query_gpt(generate_prompt_from_ir(ir))

def should_benchmark_configuration(ir, threshold=100.0):
    info = get_cost_estimation(ir)
    return info.get("predicted_cost", threshold+1) <= threshold
