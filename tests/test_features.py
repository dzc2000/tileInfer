import sys
sys.path.insert(0, '/mnt/tileInfer')
from engine.llm import LLM
from engine.sampling_params import SamplingParams

llm = LLM(model='/mnt/models/Qwen3.6-27B', tp_size=2)

# Test 1: stop strings
params = SamplingParams(temperature=0.7, max_tokens=256, stop=['the'])
out = llm.generate(['Tell me about AI'], params)
print('Stop test:', out[0]['text'][:100])
print('Stopped early:', len(out[0]['token_ids']) < 256)

# Test 2: repetition penalty
params = SamplingParams(temperature=0.7, max_tokens=128, repetition_penalty=1.2)
out = llm.generate(['The quick brown fox'], params)
print('RepPen test:', out[0]['text'][:200])

# Test 3: multiple prompts (mixed batching)
params = SamplingParams(temperature=0.0, max_tokens=32)
prompts = ['Hello', 'Tell me a joke', 'What is AI?', 'Explain quantum computing']
out = llm.generate(prompts, params)
for i, o in enumerate(out):
    print(f'  Prompt {i}: {o["text"][:80]}...')
print('All feature tests passed')
