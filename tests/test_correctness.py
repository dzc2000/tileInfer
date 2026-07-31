import sys
sys.path.insert(0, '/mnt/tileInfer')
from engine.llm import LLM
from engine.sampling_params import SamplingParams

llm = LLM(model='/mnt/models/Qwen3.6-27B', tp_size=2)
params = SamplingParams(temperature=0.0, max_tokens=64)

# 基本生成
out = llm.generate(['What is 2+2? Answer briefly.'], params)
print('Output:', out[0]['text'])

# Streaming
print('Streaming: ', end='')
for idx, text, done in llm.generate_stream(['Hello'], params):
    print(text, end='', flush=True)
print()
print('All tests passed')
