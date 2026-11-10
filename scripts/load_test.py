"""Test de charge asynchrone de l'API : débit, latences p50/p95/p99, taux d'erreur."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


async def worker(client, url, questions, n, lat, errors):
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/v1/ask", json={"question": random.choice(questions)})
            if r.status_code != 200:
                errors.append(r.status_code)
        except httpx.HTTPError as e:
            errors.append(type(e).__name__)
        lat.append((time.perf_counter() - t0) * 1000)


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--requests", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "reports" / "load_test.json"))
    a = ap.parse_args()
    questions = [json.loads(l)["question"] for l in (ROOT / "data/eval/eval_set.jsonl").read_text(encoding="utf-8").splitlines()]
    random.seed(42)
    lat, errors = [], []
    per = a.requests // a.concurrency
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=a.concurrency)) as c:
        await c.post(f"{a.url}/v1/ask", json={"question": "échauffement"})
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(c, a.url, questions, per, lat, errors) for _ in range(a.concurrency)))
        dur = time.perf_counter() - t0
    res = {
        "requests": len(lat),
        "concurrency": a.concurrency,
        "duration_s": round(dur, 2),
        "throughput_rps": round(len(lat) / dur, 1),
        "error_rate": round(len(errors) / len(lat), 4),
        "latency_ms": {
            "p50": round(statistics.median(lat), 1),
            "p95": round(pct(lat, 95), 1),
            "p99": round(pct(lat, 99), 1),
            "max": round(max(lat), 1),
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
