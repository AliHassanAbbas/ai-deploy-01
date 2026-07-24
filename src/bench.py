"""The ONE benchmarking harness every experiment in this tutorial uses.

Fair comparisons require identical measurement. All scripts import
benchmark() and report() from here, so a number from experiment 01 can
be compared with a number from experiment 07.
"""
import statistics
import time

import torch


def benchmark(fn, warmup: int = 20, iters: int = 200) -> dict:
    """Time fn() and return latency/throughput statistics.

    warmup: untimed runs first. The first calls of any model LIE about
    speed - caches are cold, lazy allocations happen, JIT compilers are
    still compiling. We let all of that finish before the stopwatch starts.

    GPU caveat: CUDA runs asynchronously - python returns before the GPU
    finished. torch.cuda.synchronize() blocks until the GPU is truly done,
    so we time real work, not the launching of work.
    """
    use_cuda = torch.cuda.is_available()

    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    return {
        "mean_ms": statistics.mean(times_ms),
        "p50_ms": times_ms[len(times_ms) // 2],
        "p99_ms": times_ms[min(int(len(times_ms) * 0.99), len(times_ms) - 1)],
    }


def report(name: str, stats: dict, batch_size: int = 1):
    """Print one aligned result row. Throughput = items finished per second."""
    throughput = batch_size * 1000.0 / stats["mean_ms"]
    print(
        f"{name:<34} "
        f"mean {stats['mean_ms']:8.3f} ms | "
        f"p50 {stats['p50_ms']:8.3f} ms | "
        f"p99 {stats['p99_ms']:8.3f} ms | "
        f"{throughput:10.1f} items/s"
    )