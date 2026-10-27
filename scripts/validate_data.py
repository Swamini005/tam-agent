import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    kb_dir = root / "knowledge-base"

    print("Checking dataset integrity...")
    errors = []

    # 1. Check accounts.json
    acc_path = data_dir / "accounts.json"
    if not acc_path.exists():
        errors.append(f"Missing accounts file: {acc_path}")
    else:
        try:
            accounts = json.loads(acc_path.read_text(encoding="utf-8"))
            if not isinstance(accounts, list) or len(accounts) == 0:
                errors.append("accounts.json must contain a non-empty list")
            else:
                print(f"[OK] accounts.json: {len(accounts)} accounts loaded")
        except Exception as e:
            errors.append(f"Invalid accounts.json: {e}")

    # 2. Check tickets.json
    tkt_path = data_dir / "tickets.json"
    if not tkt_path.exists():
        errors.append(f"Missing tickets file: {tkt_path}")
    else:
        try:
            tickets = json.loads(tkt_path.read_text(encoding="utf-8"))
            if not isinstance(tickets, list) or len(tickets) == 0:
                errors.append("tickets.json must contain a non-empty list")
            else:
                print(f"[OK] tickets.json: {len(tickets)} tickets loaded")
        except Exception as e:
            errors.append(f"Invalid tickets.json: {e}")

    # 3. Check knowledge base
    if not kb_dir.exists():
        errors.append(f"Missing knowledge-base folder: {kb_dir}")
    else:
        kb_files = list(kb_dir.rglob("*.md"))
        if not kb_files:
            errors.append("No markdown articles found in knowledge-base/")
        else:
            print(f"[OK] knowledge-base: {len(kb_files)} articles found")

    if errors:
        print("\nData validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("\nDataset integrity confirmed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
