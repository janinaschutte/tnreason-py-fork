"""Retry one recorded wall timeout without changing its original experiment."""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tie", choices=("name", "oldest_pruned"))
    parser.add_argument("allowance", type=int)
    parser.add_argument("puzzle", type=int)
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("allowance_frozen_base", folder / "run.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    original = base.read(folder / "protocol.json")
    case = [args.tie, args.allowance, args.puzzle]
    assert case in original["cases"], "Case absent from the frozen experiment"
    assert args.seconds > original["guards"]["seconds"], "Retry must increase the wall guard"
    name = f"{args.tie}_{args.allowance}_{args.puzzle}.json"
    original_row = folder / "rows" / name
    record = base.read(original_row)
    assert record["protocol_sha256"] == base.digest(folder / "protocol.json")
    assert record["status"] == "wall_timeout", "Only actual wall timeouts can be retried"
    assert (record["tie"], record["allowance"], record["puzzle"]) == tuple(case)
    assert base.digest(original["data"]) == original["data_sha256"]
    assert all(base.digest(base.ROOT / path) == expected
               for path, expected in original["source_sha256"].items())
    assert base.ORIGINAL_ATTACH.__module__ == base.runtime.__name__, "Use a fresh process"
    destination = folder / f"retries_{args.seconds}" / name.removesuffix(".json")
    protocol = dict(original, cases=[case], workers=1,
                    guards={**original["guards"], "seconds": args.seconds},
                    source_sha256={**original["source_sha256"],
                        str(Path(__file__).resolve().relative_to(base.ROOT)):
                        base.digest(Path(__file__))},
                    retry_of=dict(folder=str(folder),
                        protocol_sha256=base.digest(folder / "protocol.json"),
                        row_sha256=base.digest(original_row)),
                    wall_guard_note="Retry only a recorded wall timeout; all other limits are unchanged.")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "protocol.json"
    if path.exists():
        assert base.read(path) == protocol, "Frozen retry protocol changed"
    else:
        base.write(path, protocol)
    (destination / "rows").mkdir(exist_ok=True)
    base.OUT = destination
    if not (destination / "rows" / name).exists():
        base.init_worker()
    row = base.run_case(case)
    print(row["tie"], row["allowance"], row["puzzle"], row["status"], row["nodes"],
          f"{row['solve_s']:.2f}s", str(destination), flush=True)


if __name__ == "__main__":
    main()
