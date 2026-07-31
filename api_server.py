"""OpenAI 兼容的 HTTP API 服务器（基于标准库 http.server）。
单卡: python api_server.py --model /path/to/model --port 8000
多卡: torchrun --nproc_per_node=2 api_server.py --model /path/to/model --tp-size 2
"""
import os, argparse, json, time, uuid, threading
os.environ.setdefault("TILELANG_CACHE_DIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tilelang_cache"))
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch, torch.distributed as dist
from engine.llm import LLM
from engine.sampling_params import SamplingParams
from engine.parallel import get_tp_rank, is_tp_active

llm: LLM = None          # 全局引擎
MODEL_NAME = ""          # 模型名（用于响应）
infer_lock = threading.Lock()  # 串行化推理（TP 下必须）

def build_prompt(messages, response_format=None):
    """用 tokenizer 的 chat template 拼接 prompt。"""
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if response_format and isinstance(response_format, dict) \
            and response_format.get("type") == "json_object":
        prompt += "\n请以JSON格式输出。"  # 结构化输出指令
    return prompt

def make_sampling_params(data):
    """从请求体构造 SamplingParams。"""
    stop = data.get("stop")
    stop = [stop] if isinstance(stop, str) else (stop or [])
    return SamplingParams(
        temperature=float(data.get("temperature", 0.6)),
        top_p=float(data.get("top_p", 1.0)),
        top_k=int(data.get("top_k", -1)),
        max_tokens=int(data.get("max_tokens", 256)),
        repetition_penalty=float(data.get("repetition_penalty", 1.0)),
        stop=stop)

def broadcast_work(prompt, params):
    """rank 0 广播 prompt 与参数，唤醒 worker 参与本轮推理。"""
    pb = prompt.encode("utf-8")
    length = torch.tensor([len(pb)], dtype=torch.long, device="cuda")
    dist.broadcast(length, src=0)
    if pb:
        buf = torch.frombuffer(bytearray(pb), dtype=torch.uint8).to("cuda")
        dist.broadcast(buf, src=0)
    p = torch.tensor([params.temperature, params.top_p, params.top_k,
                      params.max_tokens, params.repetition_penalty],
                     dtype=torch.float32, device="cuda")
    dist.broadcast(p, src=0)

def worker_loop():
    """非 rank 0 的 worker：循环接收广播并参与推理。"""
    while True:
        length = torch.tensor([0], dtype=torch.long, device="cuda")
        dist.broadcast(length, src=0)
        if int(length.item()) == -1:
            break  # 退出信号
        n, prompt = int(length.item()), ""
        if n > 0:
            buf = torch.empty(n, dtype=torch.uint8, device="cuda")
            dist.broadcast(buf, src=0)
            prompt = buf.cpu().numpy().tobytes().decode("utf-8")
        p = torch.zeros(5, dtype=torch.float32, device="cuda")
        dist.broadcast(p, src=0)
        llm.generate([prompt], SamplingParams(
            temperature=p[0].item(), top_p=p[1].item(), top_k=int(p[2].item()),
            max_tokens=int(p[3].item()), repetition_penalty=p[4].item()))

def run_inference(prompt, params):
    """非流式推理。"""
    with infer_lock:
        if is_tp_active():
            broadcast_work(prompt, params)
        return llm.generate([prompt], params)[0]

def run_inference_stream(prompt, params):
    """流式推理（生成器）。"""
    with infer_lock:
        if is_tp_active():
            broadcast_work(prompt, params)
        for _i, text, _f in llm.generate_stream([prompt], params):
            yield text

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, message):
        self._send_json(code, {"error": {"message": message, "type": "invalid_request_error"}})

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "created": int(time.time())}]})
        else:
            self._err(404, f"Not found: {self.path}")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._err(404, f"Not found: {self.path}")
            return
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
        except Exception as e:
            self._err(400, f"Invalid JSON: {e}")
            return
        if not data.get("messages"):
            self._err(400, "messages is required")
            return
        try:
            prompt = build_prompt(data["messages"], data.get("response_format"))
            params = make_sampling_params(data)
        except Exception as e:
            self._err(400, f"Invalid params: {e}")
            return
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        if data.get("stream", False):
            self._stream(rid, created, prompt, params)
        else:
            self._nonstream(rid, created, prompt, params)

    def _nonstream(self, rid, created, prompt, params):
        try:
            out = run_inference(prompt, params)
        except Exception as e:
            self._err(500, f"Internal error: {e}")
            return
        pt, ct = len(out["prompt_token_ids"]), len(out["token_ids"])
        self._send_json(200, {
            "id": rid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out["text"]},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})

    def _stream(self, rid, created, prompt, params):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            for text in run_inference_stream(prompt, params):
                send({"id": rid, "object": "chat.completion.chunk", "created": created,
                      "model": MODEL_NAME,
                      "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
            send({"id": rid, "object": "chat.completion.chunk", "created": created,
                  "model": MODEL_NAME,
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as e:
            send({"error": {"message": str(e), "type": "internal_error"}})

    def log_message(self, *a):
        pass  # 静默默认日志

def main():
    global llm, MODEL_NAME
    ap = argparse.ArgumentParser(description="OpenAI 兼容 API 服务器")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True, help="模型路径")
    ap.add_argument("--tp-size", type=int, default=1, help="张量并行大小")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    MODEL_NAME = os.path.basename(os.path.normpath(args.model)) or args.model
    print(f"初始化引擎: model={args.model}, tp_size={args.tp_size}")
    llm = LLM(model=args.model, max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=args.enforce_eager, tp_size=args.tp_size)

    # TP>1 时：rank 0 启动 HTTP 服务器，其余 rank 进入 worker 循环参与推理
    if args.tp_size > 1 and get_tp_rank() != 0:
        print(f"[worker] rank {get_tp_rank()} 进入推理循环")
        worker_loop()
    else:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"API 服务器: http://{args.host}:{args.port} (rank {get_tp_rank()})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        if is_tp_active():  # 通知 worker 退出
            dist.broadcast(torch.tensor([-1], dtype=torch.long, device="cuda"), src=0)

if __name__ == "__main__":
    main()
