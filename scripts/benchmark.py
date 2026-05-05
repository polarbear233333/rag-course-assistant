import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.benchmark import BenchmarkRunner


def load_questions(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append(obj["question"])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare pure LLM and source-grounded RAG.")
    parser.add_argument("--dataset", default="datasets/eval_examples.jsonl")
    parser.add_argument("--output", default="outputs/benchmark_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions(ROOT / args.dataset)
    result = BenchmarkRunner().compare_dataset(questions, top_k=args.top_k)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
