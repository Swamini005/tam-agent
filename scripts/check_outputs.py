import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tam_agent.schema import validate


def check(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Invalid JSON: {exc}"]

    if not isinstance(data, dict):
        return ["Root element must be a JSON object"]

    return validate(data)


def main():
    paths = sys.argv[1:] or ["outputs"]
    files = []
    for arg in paths:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            files.append(p)

    if not files:
        print("No output JSON files found to validate.")
        sys.exit(1)

    failed = 0
    for file in files:
        problems = check(file)
        if problems:
            failed += 1
            print(f"FAIL  {file.name}")
            for p in problems:
                print(f"      - {p}")
        else:
            print(f"PASS  {file.name}")

    print(f"\n{len(files) - failed}/{len(files)} valid")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
