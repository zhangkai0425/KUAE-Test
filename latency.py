# TODO: Test the latency of d = 3, 5, 7 (latency for 3*d rounds or latency per round)
import time
import torch
import torch.nn as nn
import numpy as np
from model import MaskedBlockDecoder

def benchmark(model, events, n_warmup=50, n_iter=1000, use_amp=False):
    model.eval()
    latencies = []
    with torch.no_grad():
        # warmup
        for _ in range(n_warmup):
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    _ = model(events)
            else:
                _ = model(events)
                
        for _ in range(n_iter):
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    _ = model(events)
            else:
                _ = model(events)
            latencies.append(model.latency_us)
        
    latencies = np.array(latencies)
    mean_us = latencies.mean()
    std_us = latencies.std(ddof=1)
    return mean_us, std_us


def run_latency_test(d, mode="fp32"):
    B = 1
    d_model = 192
    N = d + d + d

    events = torch.randint(0, 2, (B, N, d + 1, d + 1), dtype=torch.float)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = MaskedBlockDecoder(d=d, core_size=d, buffer_size=d, d_model=d_model, nhead=4, num_layers=3).to(device)
    start = time.perf_counter_ns()
    events = events.to(device)
    end = time.perf_counter_ns()
    print(f"#### batchsize = {B}, data shape = {events.shape}, communication-latency = {(end-start) / 1000:.4f} us")
    
    model = torch.compile(model, mode="max-autotune")

    if mode == "fp16":
        model = model.half()
        events = events.half()
        mean_us, std_us = benchmark(model, events, use_amp=True)
    elif mode == "fp32":
        mean_us, std_us = benchmark(model, events)
    else:
        raise ValueError("mode must be one of ['fp32', 'fp16']")
    
    cycles = N * B
    output = f"d = {d}, mode = {mode}, latency = {mean_us:.2f} ± {std_us:.2f} us, [Throughput] latency per cycle = {mean_us / cycles:.2f} ± {std_us / cycles:.2f} us"
    
    
    print(f"d = {d}, mode = {mode}, latency = {mean_us:.2f} ± {std_us:.2f} us, [Throughput] latency per cycle = {mean_us / cycles:.2f} ± {std_us / cycles:.2f} us")
    return output
    


if __name__ == "__main__":
    outputs = []
    for d in [3, 5, 7]:
        for mode in ["fp16"]:
            try:
                output = run_latency_test(d, mode)
                outputs.append(output)
            except Exception as e:
                print(f"Failed at d={d}, mode={mode}, error: {e}")
    
    for output in outputs:
        print(output)
        