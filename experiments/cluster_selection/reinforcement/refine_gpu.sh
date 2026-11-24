#!/usr/bin/env bash
# Shared-filesystem continuation; each invocation writes a new segment.
set -euo pipefail
if (( $# < 3 )); then
    printf 'Usage: %s PYTHON RUN_ROOT DATA [refine options]\n' "$0" >&2
    exit 2
fi
cluster_python=$1
cluster_run_root=$2
shift 2
exec "$cluster_python" -u - "$cluster_run_root" "$@" <<'PY'
import fcntl
from pathlib import Path
import sys
from uuid import uuid4

root, arguments = Path(sys.argv[1]).expanduser().resolve(), sys.argv[2:]
for argument in arguments:
    option = argument.split("=", 1)[0]
    if option.startswith("--") and len(option) > 2 and any(
            reserved.startswith(option) for reserved in ("--output", "--resume")):
        raise SystemExit("The wrapper owns --output and --resume; pass only the other refine options")
root.mkdir(parents=True, exist_ok=True)
with (root / ".lock").open("a") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"A training process already owns {root}", file=sys.stderr)
        raise SystemExit(75)
    checkpoints = [path for path in root.glob("segment*/resume.pt") if path.is_file()]
    resume = max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, str(path))) if checkpoints else None
    if resume is not None:
        remaining, index = [], 0
        while index < len(arguments):
            option = arguments[index].split("=", 1)[0]
            if option.startswith("--") and len(option) > 2 and "--initial".startswith(option):
                index += 1 if "=" in arguments[index] else 2
            else:
                remaining.append(arguments[index])
                index += 1
        arguments = [*remaining, "--resume", str(resume)]
    segment = root / f"segment_{uuid4().hex}"
    print(f"Training segment: {segment}; resume: {resume or 'none (fresh initialization)'}", flush=True)
    sys.argv = ["refine", *arguments, "--output", str(segment)]
    # Exact source/settings/optimizer/RNG checks are owned by refine.fit.
    from experiments.cluster_selection.reinforcement import refine
    refine.main()
PY
