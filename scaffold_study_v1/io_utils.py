"""Strict, portable report I/O. Never unpickle a model or trajectory."""
import csv
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    data = Path(path).read_text(encoding="utf-8-sig")
    require(not data.startswith("version https://git-lfs"), f"Git LFS pointer, not data: {path}. Fetch this file first.")
    return json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {x}")))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def boolean(value):
    if value in ("", None, "None", "null"):
        return None
    if type(value) is bool:
        return value
    require(value in ("True", "False", "true", "false"), f"Invalid boolean: {value!r}")
    return value.lower() == "true"
