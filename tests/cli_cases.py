"""Black-box CLI checks. No dependency on the solution or its profile format."""

import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from onnx_fixtures import (
    branches,
    data,
    graph,
    inspect,
    interface,
    members,
    model,
    op,
    ops,
    safetensors,
    tensor,
    vi,
)

BIN = Path(os.environ.get("CARGO_TARGET_DIR", "target")).resolve() / "release/rten"


def command(*args, ok=True):
    res = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    if (res.returncode == 0) != ok:
        raise AssertionError(
            f"{args[0]} exited {res.returncode}\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}"
        )
    return res.stdout + res.stderr


def bounded(*args):
    limit = 256 * 1024
    with tempfile.TemporaryFile() as log:
        p = subprocess.Popen([BIN, *map(str, args)], stdout=log, stderr=log, start_new_session=True)
        peak = 0
        while p.poll() is None:
            todo = [p.pid]
            total = 0
            while todo:
                pid = todo.pop()
                try:
                    proc = Path("/proc") / str(pid)
                    status = (proc / "status").read_text()
                    total += int(re.search(r"^VmRSS:\s+(\d+)", status, re.M)[1])
                    todo.extend(map(int, (proc / f"task/{pid}/children").read_text().split()))
                except (OSError, TypeError):
                    pass
            peak = max(peak, total)
            if peak > limit:
                import signal

                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                raise AssertionError(f"{args[0]} exceeded 256 MiB RSS ({peak / 1024:.1f} MiB)")
            time.sleep(0.008)
        log.seek(0)
        assert p.returncode == 0, log.read().decode(errors="replace")[-1000:]
        label = args[0] + (" " + args[args.index("--format") + 1] if "--format" in args else "")
        print(f"512 MiB external fixture / {label} peak RSS {peak / 1024:.1f} MiB", flush=True)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / "source"
        self.src.mkdir()
        self.out = self.root / "out"
        self.out.mkdir()
        self.index = 0

    def file(self, suffix=".st"):
        self.index += 1
        return self.root / (str(self.index) + suffix)

    def export(self, action, original, *args):
        out = self.out / self.file(".onnx").name
        command(action, original, "-o", out, *args)
        actual = inspect(out)
        for f in (11, 12):
            self.assertEqual(interface(inspect(original), f), interface(actual, f))
        return out

    def move(self, path):
        dest = self.root / ("relocated-" + path.stem)
        dest.mkdir()
        for member in members(path):
            to = dest / member.relative_to(path.parent)
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(member, to)
        return dest / path.name

    def check(self, path, feed, expected, tol):
        ref = safetensors(self.file(), **expected)
        subprocess.run(
            [BIN.parent / "examples/harbor_readback", path, feed, ref, str(tol)], check=True
        )

    def simple(self, w=1):
        dims = ["batch", 2]
        return model(
            self.src / "model.onnx",
            graph(
                [op("MatMul", ["x", "w"], ["out"])],
                [vi("x", 1, dims)],
                [vi("out", 1, dims)],
                [tensor("w", 1, [2, 2], [w, 0, 0, w])],
            ),
        )

    def test_profiles_and_formats(self):
        original = self.simple()

        def feed(v):
            return safetensors(self.file(), x=(1, [1, 2], v))

        a, b = feed([-3, 0]), feed([0, 5])

        def calibrate(*args):
            out = self.file(".profile")
            command("calibrate", original, "-o", out, *args)
            return out

        p, q = calibrate("--sample", a), calibrate("--sample", b)
        joined = [
            calibrate("--sample", a, "--sample", b),
            calibrate("--profile", p, "--profile", q),
            calibrate("--profile", q, "--profile", p),
        ]
        for fmt in ("qdq", "qoperator"):
            for source in [("--profile", p) for p in joined] + [("--sample", a, "--sample", b)]:
                m = self.move(self.export("quantize", original, "--format", fmt, *source))
                kinds = ops(inspect(m))
                if fmt == "qdq":
                    self.assertTrue({"QuantizeLinear", "DequantizeLinear", "MatMul"} <= set(kinds))
                    self.assertNotIn("QLinearMatMul", kinds)
                else:
                    self.assertIn("QLinearMatMul", kinds)
                self.check(m, feed([-2.86, 4.81]), {"out": (1, [1, 2], [-2.86, 4.81])}, 0.09)
        other = self.simple(w=2)
        for action in ["calibrate", "quantize"]:
            fmt = ["--format", "qoperator"] if action == "quantize" else []
            command(action, other, "-o", self.file(), *fmt, "--profile", p, ok=False)

    def test_profiles_merge_after_sample_removal(self):
        import random

        original = self.simple()
        rng = random.Random(4471)
        count = 32768

        def observe(low, high):
            values = [low, high] + [rng.uniform(low, high) for _ in range(count - 2)]
            return safetensors(self.file(), x=(1, [count // 2, 2], values))

        negative = [observe(-5.0, 0.0) for _ in range(6)]
        positive = [observe(0.0, 3.0) for _ in range(6)]

        def profile(files):
            directory = self.root / ("profile-" + self.file().stem)
            directory.mkdir()
            p = directory / "ranges"
            args = [item for f in files for item in ("--sample", f)]
            command("calibrate", original, "-o", p, *args)
            return p, sum(x.stat().st_size for x in directory.rglob("*") if x.is_file())

        _, base_size = profile(negative[:1])
        a, large_size = profile(negative)
        b, _ = profile(positive)
        # Broad allowance for metadata and auxiliary files. A new observation
        # of the same fixed graph must not append its calibration tensors.
        self.assertLess(large_size - base_size, negative[0].stat().st_size)
        for f in negative + positive:
            f.unlink()
        merged = self.file(".profile")
        command("calibrate", original, "-o", merged, "--profile", b, "--profile", a)
        q = self.move(
            self.export("quantize", original, "--profile", merged, "--format", "qoperator")
        )
        held = safetensors(self.file(), x=(1, [1, 2], [-4.8, 2.85]))
        self.check(q, held, {"out": (1, [1, 2], [-4.8, 2.85])}, 0.09)

    def test_optimize_conv_bias_and_shared_outputs(self):
        for exact, shared in [(True, False), (False, False), (True, True)]:
            weights = [
                tensor("xs", 1, [], [0.5]),
                tensor("xz", 2, [], [128]),
                tensor("ws", 1, [2], [0.25, 0.5]),
                tensor("wz", 3, [2], [0, 0]),
                tensor("w", 3, [2, 1, 1, 1], [3, 5]),
                tensor("ys", 1, [], [0.125]),
                tensor("yz", 2, [], [128]),
                tensor("bias", 1, [2], [0.25 if exact else 0.30, -0.5]),
            ]
            shape = [1, 2, 1, 2]
            nodes = [
                op("DequantizeLinear", ["x", "xs", "xz"], ["dx"]),
                op("DequantizeLinear", ["w", "ws", "wz"], ["dw"], axis=-4),
                op("Conv", ["dx", "dw", "bias"], ["f"], kernel_shape=[1, 1], strides=[1, 1]),
                op("QuantizeLinear", ["f", "ys", "yz"], ["q"]),
                op("DequantizeLinear", ["q", "ys", "yz"], ["out"]),
            ]
            outs = [vi("out", 1, shape)] + ([vi("f", 1, shape)] if shared else [])
            m = model(
                self.src / "conv.onnx", graph(nodes, [vi("x", 2, [1, 1, 1, 2])], outs, weights)
            )
            export = self.move(self.export("optimize", m))
            self.assertEqual("QLinearConv" in ops(inspect(export)), exact and not shared)
            x = safetensors(self.file(), x=(2, [1, 1, 1, 2], [129, 126]))
            floats = [
                0.375 + (0.25 if exact else 0.30),
                -0.75 + (0.25 if exact else 0.30),
                0.75,
                -3.0,
            ]
            result = {"out": (1, shape, [round(z / 0.125) * 0.125 for z in floats])}
            if shared:
                result["f"] = (1, shape, floats)
            self.check(export, x, result, 1e-5)

    def test_nested_if_loop_captures_and_unseen_regions(self):
        width = 128
        dims = [1, width]
        body = graph(
            [op("MatMul", ["state", "weight"], ["next"]), op("Identity", ["continue"], ["again"])],
            [vi("iteration", 7, []), vi("continue", 9, []), vi("state", 1, dims)],
            [vi("again", 9, []), vi("next", 1, dims)],
            name="body",
        )
        yes = graph(
            [op("Loop", ["trip", "active", "x"], ["repeated"], body=body)],
            [],
            [vi("repeated", 1, dims)],
            name="yes",
        )
        no = graph([op("MatMul", ["x", "weight"], ["once"])], [], [vi("once", 1, dims)], name="no")
        bank = self.src / "weights.data"
        mat = [float(i == j) for i in range(width) for j in range(width)]
        bank.write_bytes(data(1, mat))
        g = graph(
            [
                op("Greater", ["selector", "zero"], ["active"]),
                op("If", ["active"], ["out"], then_branch=yes, else_branch=no),
            ],
            [vi("selector", 6, []), vi("x", 1, dims)],
            [vi("out", 1, dims)],
            [
                tensor("weight", 1, [width, width], external=(bank.name, 0, bank.stat().st_size)),
                tensor("trip", 7, [], [2]),
                tensor("zero", 6, [], [0]),
            ],
        )
        original = model(self.src / "nested.onnx", g)

        def feed(which, vals):
            return safetensors(self.file(), selector=(6, [], [which]), x=(1, dims, vals))

        v = [(i % 15 - 7) / 8 for i in range(width)]
        observed = feed(1, v)
        held = [x * 0.79 for x in v]
        for fmt in ["qdq", "qoperator"]:
            expected_op = "QuantizeLinear" if fmt == "qdq" else "QLinearMatMul"
            for both in (False, True):
                samples = ["--sample", observed]
                if both:
                    samples += ["--sample", feed(-1, v)]
                m = self.move(self.export("quantize", original, "--format", fmt, *samples))
                b = branches(inspect(m), "If")
                self.assertIn(expected_op, ops(branches(b["then_branch"], "Loop")["body"]))
                self.assertEqual(expected_op in ops(b["else_branch"]), both)
                if not both:
                    self.assertIn("MatMul", ops(b["else_branch"]))
                if both:
                    self.shared_payloads(m, width * width)
                # Source data are inaccessible while checking the relocated export.
                hidden = bank.with_suffix(".hidden")
                bank.rename(hidden)
                try:
                    for selector in [1, -1]:
                        self.check(m, feed(selector, held), {"out": (1, dims, held)}, 0.035)
                finally:
                    hidden.rename(bank)

    def shared_payloads(self, path, n):
        from onnx_fixtures import unpack
        from hashlib import sha256

        seen = {}

        def visit(g):
            for t in map(unpack, g.get(5, [])):
                if t.get(2) not in ([3], [2]):
                    continue
                where = None
                if 13 in t:
                    attrs = {e[1][0]: e[2][0] for e in map(unpack, t[13])}
                    src = path.parent / attrs[b"location"].decode()
                    off = int(attrs.get(b"offset", b"0"))
                    size = int(attrs.get(b"length", str(src.stat().st_size - off)))
                    with src.open("rb") as f:
                        f.seek(off)
                        raw = f.read(size)
                    where = (src, off, size)
                elif 9 in t:
                    raw = t[9][0]
                elif 5 in t:
                    vals = []
                    for p in t[5]:
                        if isinstance(p, int):
                            vals.append(p & 255)
                        else:
                            z = s = 0
                            for b in p:
                                z |= (b & 127) << s
                                if b & 128:
                                    s += 7
                                else:
                                    vals.append(z & 255)
                                    z = s = 0
                    raw = bytes(vals)
                else:
                    continue
                if len(raw) < n:
                    continue
                key = sha256(raw).digest()
                self.assertTrue(
                    key not in seen or (where is not None and seen[key] == where),
                    "the same 8-bit weight payload occupies multiple storage ranges",
                )
                seen[key] = where
            for nd in map(unpack, g.get(1, [])):
                for a in map(unpack, nd.get(5, [])):
                    for sg in a.get(6, []):
                        visit(unpack(sg))

        visit(inspect(path))

    def test_float_conv_with_runtime_weights(self):
        g = graph(
            [
                op(
                    "Conv",
                    ["image", "filter", "bias"],
                    ["features"],
                    kernel_shape=[1, 1],
                    strides=[1, 1],
                )
            ],
            [vi("image", 1, [1, 1, 1, 3]), vi("filter", 1, [2, 1, 1, 1])],
            [vi("features", 1, [1, 2, 1, 3])],
            [tensor("bias", 1, [2], [0.25, -0.5])],
        )
        m = model(self.src / "dynamic.onnx", g)

        def sample(x, w):
            return safetensors(self.file(), image=(1, [1, 1, 1, 3], x), filter=(1, [2, 1, 1, 1], w))

        a = sample([-2.0, 0.0, 2.0], [1.0, -1.5])
        b = sample([-1.25, 0.25, 0.75], [0.5, -1.0])
        expected = {"features": (1, [1, 2, 1, 3], [-0.375, 0.375, 0.625, 0.75, -0.75, -1.25])}
        for fmt in ["qdq", "qoperator"]:
            out = self.move(self.export("quantize", m, "--format", fmt, "--sample", a))
            self.assertIn("Conv" if fmt == "qdq" else "QLinearConv", ops(inspect(out)))
            self.check(out, b, expected, 0.09)

    def test_ranges_cover_repeated_loop_iterations(self):
        dims = [1, 2]
        body = graph(
            [op("MatMul", ["carry", "double"], ["next"]), op("Identity", ["cond"], ["again"])],
            [vi("i", 7, []), vi("cond", 9, []), vi("carry", 1, dims)],
            [vi("again", 9, []), vi("next", 1, dims)],
            name="repeated",
        )
        m = model(
            self.src / "carried.onnx",
            graph(
                [op("Loop", ["count", "yes", "x"], ["out"], body=body)],
                [vi("x", 1, dims)],
                [vi("out", 1, dims)],
                [
                    tensor("double", 1, [2, 2], [2, 0, 0, 2]),
                    tensor("count", 7, [], [3]),
                    tensor("yes", 9, [], [1]),
                ],
            ),
        )
        obs = safetensors(self.file(), x=(1, dims, [-1.0, 1.0]))
        test = safetensors(self.file(), x=(1, dims, [-0.75, 0.75]))
        prof = self.file(".profile")
        command("calibrate", m, "--sample", obs, "-o", prof)
        obs.unlink()
        for fmt in ["qdq", "qoperator"]:
            out = self.move(self.export("quantize", m, "--format", fmt, "--profile", prof))
            self.assertIn(
                "QLinearMatMul" if fmt == "qoperator" else "QuantizeLinear",
                ops(branches(inspect(out), "Loop")["body"]),
            )
            self.check(out, test, {"out": (1, dims, [-6.0, 6.0])}, 0.19)

    def test_mixed_precision_and_infeasible_limits(self):
        import random

        rng = random.Random(6193)
        n = 96
        dims = [300, 1]
        ins = [vi("a", 1, dims), vi("b", 1, dims), vi("c", 1, [4, n])]
        outs = [vi("hard1", 1, dims), vi("hard2", 1, dims), vi("easy", 1, [4, n])]
        g = graph(
            [
                op("MatMul", [x, w], [y])
                for x, w, y in [
                    ("a", "unit", "hard1"),
                    ("b", "unit", "hard2"),
                    ("c", "matrix", "easy"),
                ]
            ],
            ins,
            outs,
            [
                tensor("unit", 1, [1, 1], [1]),
                tensor("matrix", 1, [n, n], [rng.uniform(-1.0, 1.0) for _ in range(n * n)]),
            ],
        )
        m = model(self.src / "mixed.onnx", g)
        high = [i / 299 for i in range(300)]
        zero = [0.0] * 300

        def sample(a, b, c=None):
            return safetensors(
                self.file(),
                a=(1, dims, a),
                b=(1, dims, b),
                c=(1, [4, n], c if c is not None else [0.0] * (4 * n)),
            )

        one, two = sample(high, zero), sample(zero, high)
        extra = sample(zero, zero, [rng.uniform(-1.0, 1.0) for _ in range(4 * n)])
        for fmt in ["qoperator", "qdq"]:
            args = ["--format", fmt, "--sample", one, "--sample", two, "--sample", extra]
            budget = 20000
            out = self.move(
                self.export(
                    "quantize",
                    m,
                    *args,
                    "--max-bytes",
                    budget,
                    "--max-error",
                    "0.00001",
                    "--validate",
                    one,
                    "--validate",
                    two,
                )
            )
            self.assertLessEqual(sum(p.stat().st_size for p in members(out)), budget)
            for file, a, b in [(one, high, zero), (two, zero, high)]:
                self.check(
                    out,
                    file,
                    {
                        "hard1": (1, dims, a),
                        "hard2": (1, dims, b),
                        "easy": (1, [4, n], [0.0] * (4 * n)),
                    },
                    0.00001,
                )
            self.assertIn(
                "QLinearMatMul" if fmt == "qoperator" else "QuantizeLinear", ops(inspect(out))
            )
            command(
                "quantize",
                m,
                "-o",
                self.file(".onnx"),
                *args,
                "--max-bytes",
                32,
                "--max-error",
                "0",
                "--validate",
                one,
                ok=False,
            )

            # Both limits are individually attainable, but quantizing the only
            # large matrix collapses at least two distinct validation outputs.
            hard_easy = sample(zero, zero, [i / (4 * n - 1) for i in range(4 * n)])
            diagnostic = command(
                "quantize",
                m,
                "-o",
                self.file(".onnx"),
                *args,
                "--max-bytes",
                budget,
                "--max-error",
                "0.00001",
                "--validate",
                hard_easy,
                ok=False,
            )
            self.assertTrue(diagnostic.strip(), "infeasible search returned no diagnostic")

    def test_streaming_external_model(self):
        # Each rank-one dense matrix maps one Walsh vector exactly to the next.
        # All 32 distinct matrices contribute to the public output.
        basic = self.simple()
        probe = safetensors(self.file(), x=(1, [1, 2], [0, 1]))
        command("calibrate", basic, "--sample", probe, "-o", self.file())
        n, layers = 2048, 32
        dims = [1, n]
        bank = self.src / "bank.data"

        def sign(i, k):
            return -1 if (i & k).bit_count() % 2 else 1

        weights = []
        nodes = []
        with bank.open("wb") as f:
            for k in range(layers):
                rows = {s: data(1, [s * sign(j, k + 1) / n for j in range(n)]) for s in [1, -1]}
                w = "w" + str(k)
                weights.append(tensor(w, 1, [n, n], external=(bank.name, f.tell(), n * n * 4)))
                for i in range(n):
                    f.write(rows[sign(i, k)])
                nodes.append(
                    op(
                        "MatMul",
                        ["x" if k == 0 else "v" + str(k), w],
                        ["out" if k + 1 == layers else "v" + str(k + 1)],
                    )
                )
        self.assertEqual(bank.stat().st_size, 512 * 1024**2)
        m = model(
            self.src / "stream.onnx",
            graph(nodes, [vi("x", 1, dims)], [vi("out", 1, dims)], weights),
        )
        obs = safetensors(self.file(), x=(1, dims, [1.0] * n))
        prof = self.file(".profile")
        bounded("calibrate", m, "--sample", obs, "-o", prof)
        optimized = self.out / "optimized.onnx"
        bounded("optimize", m, "-o", optimized)
        self.check(
            self.move(optimized), obs, {"out": (1, dims, [sign(i, layers) for i in range(n)])}, 1e-5
        )
        for fmt in ["qoperator", "qdq"]:
            out = self.out / (fmt + ".onnx")
            bounded("quantize", m, "--profile", prof, "--format", fmt, "-o", out)
            out = self.move(out)
            kinds = ops(inspect(out))
            self.assertIn("QLinearMatMul" if fmt == "qoperator" else "QuantizeLinear", kinds)
            feed = safetensors(self.file(), x=(1, dims, [0.5] * n))
            self.check(
                out, feed, {"out": (1, dims, [0.5 * sign(i, layers) for i in range(n)])}, 0.03
            )


def build_readback():
    src = Path(__file__).with_name("readback.rs")
    example = Path("/app/rten-cli/examples/harbor_readback.rs")
    example.parent.mkdir(exist_ok=True)
    with example.open("xb") as f:
        f.write(src.read_bytes())
    try:
        subprocess.run(
            [
                "cargo",
                "build",
                "--locked",
                "--offline",
                "--release",
                "-p",
                "rten-cli",
                "--example",
                "harbor_readback",
            ],
            check=True,
        )
    finally:
        example.unlink()


if __name__ == "__main__":
    subprocess.run(
        ["cargo", "build", "--locked", "--offline", "--release", "-p", "rten-cli"], check=True
    )
    build_readback()
    unittest.main(verbosity=2)
