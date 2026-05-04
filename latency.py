# TODO: Test the latency of d = 3, 5, 7
import time
import torch
import argparse
import numpy as np
import torch_musa
from model import MaskedBlockDecoder


def get_device():
    if torch.musa.is_available():
        return torch.device("musa:0")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def sync_device(device):
    if device.type == "musa":
        torch.musa.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(model, events, n_warmup=50, n_iter=1000):
    model.eval()
    latencies = []

    with torch.no_grad():
        # warmup
        for _ in range(n_warmup):
            _ = model(events)

        sync_device(events.device)

        for _ in range(n_iter):
            _ = model(events)
            latencies.append(model.latency_us)

    latencies = np.array(latencies)
    mean_us = latencies.mean()
    std_us = latencies.std(ddof=1)
    return mean_us, std_us


def run_latency_test(d, mode="fp32", B=1, torch_compile=False):
    B = B
    d_model = 192
    N = d + d + d

    device = get_device()
    print(f"Using device: {device}")

    # 数据
    events = torch.randint(
        0, 2,
        (B, N, d + 1, d + 1),
        dtype=torch.float32
    )

    # 模型
    model = MaskedBlockDecoder(
        d=d,
        core_size=d,
        buffer_size=d,
        d_model=d_model,
        nhead=4
    ).to(device)

    # 数据搬运
    start = time.perf_counter_ns()
    events = events.to(device)
    sync_device(device)
    end = time.perf_counter_ns()

    print(
        f"#### batchsize = {B}, data shape = {events.shape}, "
        f"communication-latency = {(end-start)/1000:.4f} us"
    )
    if torch_compile:
        model = torch.compile(model, mode="max-autotune")
    # 🔥 FP16（不使用 AMP，只用 half）
    if mode == "fp16":
        model = model.half()
        events = events.half()

    elif mode != "fp32":
        raise ValueError("mode must be one of ['fp32', 'fp16']")

    # 🚀 跑 benchmark
    mean_us, std_us = benchmark(model, events)

    cycles = N * B / 3

    output = (
        f"d = {d}, mode = {mode}, "
        f"B = {B}, cycles = {N} * {B} = {cycles}, "
        f"latency = {mean_us:.2f} ± {std_us:.2f} us, "
        f"[Throughput] latency per cycle = "
        f"{mean_us / cycles:.2f} ± {std_us / cycles:.2f} us"
    )

    print(output)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--d", type=int, nargs="+", default=[3], help="list of d values, e.g. --d 3 5 7")
    parser.add_argument("--B", type=int, nargs="+", default=[1], help="list of batch sizes, e.g. --B 1 4")
    parser.add_argument("--mode", type=str, nargs="+", default=["fp16"], choices=["fp32", "fp16"], help="precision mode")
    parser.add_argument("--torch-compile", action="store_true", help="whether to use torch.compile for optimization")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = []
    for d in args.d:
        for B in args.B:
            for mode in args.mode:
                try:
                    output = run_latency_test(d, mode, B, torch_compile=args.torch_compile)
                    outputs.append(output)
                except Exception as e:
                    print(f"Failed at d={d}, B={B}, mode={mode}, error: {e}")

    print("\n===== FINAL RESULTS =====")
    for output in outputs:
        print(output)