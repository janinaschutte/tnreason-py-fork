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
cluster_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
exec "$cluster_python" -u - "$cluster_repo_root" "$cluster_run_root" "$@" <<'PY'
from contextlib import ExitStack
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
from uuid import uuid4

repo = Path(sys.argv[1])
root, arguments = Path(sys.argv[2]).expanduser().resolve(), sys.argv[3:]
sys.path.insert(0, str(repo))
for argument in arguments:
    option = argument.split("=", 1)[0]
    if option.startswith("--") and len(option) > 2 and any(
            reserved.startswith(option) for reserved in ("--output", "--resume")):
        raise SystemExit("The wrapper owns --output and --resume; pass only the other refine options")
root.mkdir(parents=True, exist_ok=True)
with (root / ".lock").open("a") as lock, ExitStack() as resources:
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
        # Preserve the exact code of ongoing runs when the shared checkout changes.
        legacy = repo / "experiments/cluster_selection/reinforcement/results/compact_training_mcmfsdfa"
        manifest = legacy / "previous_source_hashes.json"
        saved_sources = json.loads((resume.parent / "protocol.json").read_text())["source_hashes"]
        if manifest.is_file() and saved_sources == json.loads(manifest.read_text()):
            files = {}
            with tarfile.open(legacy / "previous_training_sources.tar.gz") as archive:
                for member in archive.getmembers():
                    path = Path(member.name)
                    if (not member.isfile() or path.is_absolute() or ".." in path.parts
                            or str(path) != member.name or member.name in files):
                        raise ValueError("Invalid archived source path")
                    content = archive.extractfile(member).read()
                    if sha256(content).hexdigest() != saved_sources.get(member.name):
                        raise ValueError("Archived source hash mismatch")
                    files[member.name] = content
            if set(files) != set(saved_sources):
                raise ValueError("Archived source set mismatch")
            frozen = Path(resources.enter_context(TemporaryDirectory(prefix="tnreason-resume-")))
            for name, content in files.items():
                path = frozen / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            (frozen / "experiments/__init__.py").touch()
            sys.path.insert(0, str(frozen))
            print("Resuming with verified archived sources", flush=True)
    segment = root / f"segment_{uuid4().hex}"
    print(f"Training segment: {segment}; resume: {resume or 'none (fresh initialization)'}", flush=True)
    sys.argv = ["refine", *arguments, "--output", str(segment)]
    # Exact source/settings/optimizer/RNG checks are owned by refine.fit.
    from experiments.cluster_selection.reinforcement import refine
    refine.main()
PY
