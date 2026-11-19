# Static quantization for RTen: a sample Harbor task

Human authored LLM assisted harbor task. For high quality coding evals, [contact us](https://www.klinto.dev)

The brief asks an agent to add the ONNX `QLinearConv` and `QLinearMatMul` operators to [RTen](https://github.com/robertknight/rten), a Rust ONNX runtime. The agent also has to fuse quantize/dequantize patterns into them and add `calibrate` and `quantize` commands. Those commands must stay within 256 MiB of memory on a model with 512 MiB of weights. The full brief is in [instruction.md](../instruction.md).

## Run it

You need Docker and Harbor. These runs used Harbor 0.25.0.

```bash
git clone https://github.com/klintodev/rten-static-quantization-v3
harbor run -p rten-static-quantization-v3 -a oracle   # reference solution
harbor run -p rten-static-quantization-v3 -a nop      # does nothing
```

The Docker image builds RTen at commit `e77a3b6` from GitHub. The agent then gets two hours with no network. The verifier runs in a separate environment, rebuilds the code from source and runs three groups of tests:

- 139 of RTen's existing tests, which must keep passing (pass-to-pass). They're listed in [tests/upstream-tests.txt](../tests/upstream-tests.txt).
- 14 new unit tests in [tests/qlinear.rs](../tests/qlinear.rs) (fail-to-pass).
- 8 command-line cases in [tests/cli_cases.py](../tests/cli_cases.py) (fail-to-pass).

The reward is 1 only if every test passes.

## Results

| Agent | Reward | Pass-to-pass (139) | Unit tests (14) | Command-line cases (8) | Verifier output |
| --- | --- | --- | --- | --- | --- |
| Reference solution (`oracle`) | 1 | 139 | 14 | 8 | [test-stdout.txt](../runs/rten-v3-public-oracle/rten-static-quantization-v3__x6a35pe/verifier/test-stdout.txt) |
| Do-nothing agent (`nop`) | 0 | 139 | 0 | 0 | [test-stdout.txt](../runs/rten-v3-public-nop/rten-static-quantization-v3__oHA9DRt/verifier/test-stdout.txt) |
| Claude Sonnet 5.5, high effort, in Claude Code 2.1.327 | 0 | 139 | 14 | 7 | [test-stdout.txt](../runs/2026-11-19__00-22-27/rten-static-quantization-v3__QXL5eGr/verifier/test-stdout.txt) |

Each row is one run on 19 November 2026. The reference and do-nothing runs used the commands above on a fresh clone of this repository.

All three runs used the same task files. Harbor's content digest for them is `sha256:65277d5f511bff5270d03e6c768b4c81e9ea5de98bbdc45645d91b702a6f1a34`. Folders outside the task files, such as this one, `runs/` and `trajectories/`, aren't part of that digest.

### Where Claude Sonnet 5.5 failed

It worked for 56 minutes, then reported that everything was in and that the full test suite passed. The hidden command-line case `test_float_conv_with_runtime_weights` failed. In that case the Conv's weights are a model input, supplied at run time, rather than stored in the model. The brief says to accept runtime weights. The agent's `quantize` command only picks up Conv and MatMul layers whose weights are stored, so its QOperator export left this layer as a float `Conv`. Every other test passed.

From the verifier's terminal output, [runs/2026-11-19__00-22-27/rten-static-quantization-v3__QXL5eGr/verifier/test-stdout.txt](../runs/2026-11-19__00-22-27/rten-static-quantization-v3__QXL5eGr/verifier/test-stdout.txt):

```text
======================================================================
FAIL: test_float_conv_with_runtime_weights (__main__.ExportTests.test_float_conv_with_runtime_weights)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tests/cli_cases.py", line 372, in test_float_conv_with_runtime_weights
    self.assertIn("Conv" if fmt == "qdq" else "QLinearConv", ops(inspect(out)))
AssertionError: 'QLinearConv' not found in ['Conv']

----------------------------------------------------------------------
Ran 8 tests in 8.195s

FAILED (failures=1)
p2p rten: 118 / 118 passed
p2p rten-shape-inference: 9 / 9 passed
p2p rten-cli: 12 / 12 passed
f2p rten: 14 / 14 passed
```

Its step-by-step trajectory is in [trajectories/](../trajectories/2026-11-19__00-22-27/rten-static-quantization-v3__QXL5eGr/trajectory.json).

## What's in this repository

| Path | Contents |
| --- | --- |
| `task.toml`, `instruction.md` | Harbor task configuration and the brief the agent sees |
| `environment/` | The Dockerfile for the agent's environment |
| `solution/` | The reference solution: `reference.patch` and `solve.sh` |
| `tests/` | The hidden tests and the verifier |
| `trajectories/<job>/<trial>/trajectory.json` | The Claude Sonnet 5.5 run, step by step |
| `runs/<job>/<trial>/` | Each run's `summary.json` and verifier output. `verifier/test-stdout.txt` is the terminal output with any test failures. |

## Licence

Some files here contain material from RTen, which is licensed under MIT or Apache-2.0. [NOTICE](../NOTICE) lists them, and RTen's licence texts are in [licenses/rten/](../licenses/rten/). This repository is not affiliated with or endorsed by the RTen project.
