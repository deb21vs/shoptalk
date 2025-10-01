import time, requests, statistics

def probe(n=20):
    t = []
    for _ in range(n):
        s = time.perf_counter()
        _ = requests.post("http://localhost:8000/search", json={"query": "red men shirt under 50", "top_k": 20}).json()
        t.append(time.perf_counter() - s)
    print("n =", n)
    print("p50:", statistics.median(t))
    print("p95:", sorted(t)[int(0.95*len(t))-1])
    print("p99:", sorted(t)[int(0.99*len(t))-1])

if __name__ == "__main__":
    probe()
