
import argparse, io, json, os, sys
from pathlib import Path

try:
    from utils.postprocess import bio_postprocess
except ModuleNotFoundError:
    # добавить корень проекта в sys.path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from utils.postprocess import bio_postprocess  # type: ignore

def read_jsonl(path: str):
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser("BIO → spans")
    ap.add_argument("--input", required=True, help="JSONL: id, text, tokens, offsets, labels")
    ap.add_argument("--output", required=True, help="JSONL: id, spans[]")
    ap.add_argument("--bio-prefix", action="store_true", help="entity как 'B-XXX' вместо 'XXX'")
    args = ap.parse_args()

    out_rows = []
    for row in read_jsonl(args.input):
        spans = bio_postprocess(
            labels=row["labels"],
            offsets=row["offsets"],
            text=row["text"],
            return_bio_prefix=args.bio_prefix,
        )
        out_rows.append({"id": row["id"], "spans": spans})

    write_jsonl(args.output, out_rows)

if __name__ == "__main__":
    main()
