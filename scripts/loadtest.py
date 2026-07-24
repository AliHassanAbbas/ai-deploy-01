r"""Load test: hammer a server with concurrent requests, report the truth.

Run from the project root (server running in another terminal):
    python scripts\loadtest.py --port 8000
"""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

SENTENCES = [
    "this movie was great fun and the acting was excellent",
    "boring predictable plot and terrible awful dialogue",
    "a wonderful touching story with a brilliant cast",
    "what a dreadful lifeless disaster of a film",
]


def one_request(session, url, i):
    payload = {"text": SENTENCES[i % len(SENTENCES)]}
    t0 = time.perf_counter()
    r = session.post(url, json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, r.text
    return latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/predict"

    # warmup (first requests lie - same rule as src\bench.py)
    with requests.Session() as s:
        for i in range(10):
            one_request(s, url, i)

    sessions = [requests.Session() for _ in range(args.concurrency)]
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        latencies = list(
            pool.map(
                lambda i: one_request(sessions[i % args.concurrency], url, i),
                range(args.requests),
            )
        )
    wall = time.perf_counter() - t_start

    latencies.sort()
    print(f"requests    : {args.requests}  (concurrency {args.concurrency})")
    print(f"wall time   : {wall:.2f} s")
    print(f"throughput  : {args.requests / wall:.1f} req/s")
    print(f"latency p50 : {latencies[len(latencies) // 2]:.1f} ms")
    print(f"latency p99 : {latencies[int(len(latencies) * 0.99)]:.1f} ms")


if __name__ == "__main__":
    main()