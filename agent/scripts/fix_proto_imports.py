"""Rewrite generated protobuf absolute imports to package-relative.

``protoc`` emits ``import events_pb2 as events__pb2`` for sibling files.
Inside a Python package, that import resolves to the top of sys.path
which is not what we want — we want
``from byzminds_agent.proto_gen import events_pb2 as events__pb2``.

Run this once per regeneration. Idempotent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SIBLINGS = ("events", "ledger", "view", "kernel", "handler")
PACKAGE = "byzminds_agent.proto_gen"


def rewrite(path: Path) -> bool:
    src = path.read_text()
    orig = src
    for s in SIBLINGS:
        # `import events_pb2 as events__pb2` →
        # `from byzminds_agent.proto_gen import events_pb2 as events__pb2`
        src = re.sub(
            rf"^import {s}_pb2 as ({s}__pb2)$",
            rf"from {PACKAGE} import {s}_pb2 as \1",
            src,
            flags=re.M,
        )
        # `import events_pb2` (no alias) →
        # `from byzminds_agent.proto_gen import events_pb2`
        src = re.sub(
            rf"^import {s}_pb2$",
            rf"from {PACKAGE} import {s}_pb2",
            src,
            flags=re.M,
        )
        # gRPC stub side: `import kernel_pb2 as kernel__pb2` already covered.
        # Also rewrite cross-file gRPC stub imports:
        # `import kernel_pb2_grpc as kernel__pb2__grpc` →
        # `from byzminds_agent.proto_gen import kernel_pb2_grpc as kernel__pb2__grpc`
        src = re.sub(
            rf"^import {s}_pb2_grpc as ({s}__pb2__grpc)$",
            rf"from {PACKAGE} import {s}_pb2_grpc as \1",
            src,
            flags=re.M,
        )
    if src != orig:
        path.write_text(src)
        return True
    return False


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: fix_proto_imports.py <dir>", file=sys.stderr)
        return 2
    target = Path(argv[1])
    if not target.is_dir():
        print(f"not a directory: {target}", file=sys.stderr)
        return 2
    changed = 0
    for p in sorted(target.glob("*_pb2.py")) + sorted(target.glob("*_pb2_grpc.py")):
        if rewrite(p):
            changed += 1
    print(f"fix_proto_imports: rewrote imports in {changed} files under {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
