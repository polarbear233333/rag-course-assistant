import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.rag_evaluator import RAGEvaluator


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run custom RAG evaluation metrics.")
    parser.add_argument("--dataset", default="datasets/eval_examples.jsonl")
    parser.add_argument("--output", default="outputs/eval_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl(dataset_path)
    result = RAGEvaluator().evaluate_dataset(dataset, top_k=args.top_k)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
