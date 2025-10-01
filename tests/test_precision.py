import json, numpy as np, requests, sys

K = 10
def main(path="tests/queries.jsonl"):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("[WARN] tests/queries.jsonl not found; add labeled queries to evaluate.")
        return
    acc = []
    for line in lines:
        rec = json.loads(line)
        r = requests.post("http://localhost:8000/search", json={"query": rec["query"], "top_k": K}).json()
        ids = [it["product_id"] for it in r["items"]]
        tp = len(set(ids) & set(rec["positives"]))
        acc.append(tp / K)
    print("Precision@K:", float(np.mean(acc)))

if __name__ == "__main__":
    main(*sys.argv[1:])
