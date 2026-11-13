"""Assemble the Case 3 MAT file and verify its checksum."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / "case3"
    spec = json.loads((root / "data_parts.json").read_text())
    target = root / spec["filename"]
    digest = hashlib.sha256()
    if target.exists():
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    else:
        with target.open("xb") as output:
            for part in spec["parts"]:
                with (root / part).open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError("Case 3 data checksum mismatch")
    print("Case 3 data checksum passed")


if __name__ == "__main__":
    main()
