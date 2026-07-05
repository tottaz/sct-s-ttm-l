#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "version.json"


def main():
    data = json.loads(VERSION_FILE.read_text(encoding="utf-8"))
    build_number = int(data.get("build_number", 0)) + 1
    data["build_number"] = build_number
    data["build"] = f"{datetime.utcnow().strftime('%Y.%m.%d')}.{build_number}"
    VERSION_FILE.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{data.get('version', '0.0.0')}+{data['build']}")


if __name__ == "__main__":
    main()
