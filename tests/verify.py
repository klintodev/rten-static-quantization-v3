"""Run RTen's pinned regression cases and the verifier-only ONNX models."""
import re
import subprocess
from collections import defaultdict
from pathlib import Path

TESTS = Path(__file__).resolve().parent
LOGS = Path("/logs/verifier")


def run_cases(suite, package, names):
    if not names:
        raise ValueError(f"Empty {suite} suite for {package}")
    command = ["cargo", "test", "--locked", "--offline", "-p", package]
    command.extend(["--bin", "rten"] if package == "rten-cli" else ["--lib"])
    if package == "rten":
        command.extend(["--features", "all-ops,rten_format"])
    # libtest combines multiple filters with OR. Exact names ensure that
    # similarly named tests cannot replace missing or newly ignored cases.
    command.extend(["--", "--exact", "--format", "pretty", *names])
    log = LOGS / f"{suite}-{package}.log"
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    output = log.read_text(errors="replace")
    passed = set(re.findall(r"^test (\S+) \.\.\. ok$", output, re.MULTILINE))
    missing = set(names) - passed
    print(f"{suite} {package}: {len(names) - len(missing)} / {len(names)} passed")
    if missing or result.returncode:
        print(output[-8000:])
        print(f"See {log} for the complete output.")
        return False
    return True


def main():
    by_package = defaultdict(list)
    for line in (TESTS / "upstream-tests.txt").read_text().splitlines():
        package, name = line.split()
        by_package[package].append(name)
    results = [run_cases("p2p", package, by_package[package])
               for package in ("rten", "rten-shape-inference", "rten-cli")]

    # A child of model can inspect the optimized graph using the baseline
    # model builder. This file is uploaded only in Harbor's verifier phase.
    fixture = TESTS / "qlinear.rs"
    tests = re.findall(r"^#\[test\]\nfn (f2p_\w+)\(", fixture.read_text(), re.MULTILINE)
    module = Path("src/model.rs")
    original = module.read_bytes()
    try:
        with module.open("a") as stream:
            stream.write('\n#[cfg(test)]\n#[path = "/tests/qlinear.rs"]\nmod harbor_qlinear;\n')
        names = [f"model::harbor_qlinear::{test}" for test in tests]
        results.append(run_cases("f2p", "rten", names))
    finally:
        module.write_bytes(original)
    results.append(subprocess.run(["python3", str(TESTS / "cli_cases.py")]).returncode == 0)
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
