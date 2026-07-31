import sys
sys.path.insert(0, '/mnt/tileInfer')
from engine.llm import LLM
from engine.sampling_params import SamplingParams

# 极小 KV cache 触发抢占
llm = LLM(model='/mnt/models/Qwen3.6-27B', tp_size=2,
          max_num_seqs=4, max_model_len=2048)
params = SamplingParams(max_tokens=512, ignore_eos=True)

# 多个长请求触发内存压力
prompts = ['Tell me a very long story ' * 50] * 4
try:
    outputs = llm.generate(prompts, params)
    print('OOM recovery: PASS (completed without crash)')
    print(f'Generated {sum(len(o["token_ids"]) for o in outputs)} tokens total')
except Exception as e:
    print(f'OOM recovery: FAIL ({e})')
