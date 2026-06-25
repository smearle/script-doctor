import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD","spawn")
from vllm import LLM, SamplingParams
print("importing ok, loading model...", flush=True)
llm = LLM(model="Qwen/Qwen3-4B", max_model_len=4096, gpu_memory_utilization=0.6, enforce_eager=True)
out = llm.generate(["Write one short sentence about a cat."], SamplingParams(max_tokens=32, temperature=0.7))
print("OUTPUT:", out[0].outputs[0].text.strip()[:200], flush=True)
print("OFFLINE_VLLM_OK", flush=True)
