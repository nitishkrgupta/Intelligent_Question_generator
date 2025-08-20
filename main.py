import argparse
import json
from pathlib import Path

from utils import generate_from_text

def main():
    parser = argparse.ArgumentParser(description="Intelligent Question Generator (CLI)")
    parser.add_argument("--in", dest="inp", required=True, help="Path to input .txt file")
    parser.add_argument("--out", dest="out", default="questions.json", help="Output JSON/CSV path")
    parser.add_argument("--max-sents", type=int, default=8, help="Max sentences to consider")
    parser.add_argument("--num-qs", type=int, default=3, help="Num Q candidates per sentence")
    parser.add_argument("--types", type=str, default="mcq,fill_blank,true_false,short_answer",
                        help="Comma-separated types to include")
    args = parser.parse_args()

    in_path = Path(args.inp)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    text = in_path.read_text(encoding="utf-8")
    include = tuple(t.strip() for t in args.types.split(",") if t.strip())

    questions = generate_from_text(
        text,
        max_sents=args.max_sents,
        num_q_per_sent=args.num_qs,
        include_types=include,
    )

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    elif out_path.suffix.lower() == ".csv":
        import csv
        keys = sorted(set().union(*[q.keys() for q in questions]))
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for q in questions:
                writer.writerow(q)
    else:
        raise ValueError("Output must be .json or .csv")

    print(f"Saved {len(questions)} questions to {out_path}")

if __name__ == "__main__":
    main()
