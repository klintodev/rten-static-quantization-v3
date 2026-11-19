Work offline with the provided Rust toolchain and locked Cargo dependencies.

# Static quantization and mixed-precision ONNX export in RTen

Implement ONNX `QLinearConv` and `QLinearMatMul`. Preserve convolution attributes, MatMul
vector and batch broadcasting, and shape inference for batched and symbolic inputs.

## Operators and fusion

Support all `int8`/`uint8` combinations for activations, weights, and outputs. Zero points
must match their tensors’ dtypes. The output zero point determines the output dtype.

Scales must be positive, finite `float32`. Activation and output parameters are per-tensor
(scalars or one-element vectors). Weight scale and zero point must both be per-tensor or both
be vectors along Conv’s output-channel axis or MatMul’s final axis. Rank-one MatMul weights
have one implicit output column and accept only per-tensor parameters. Reject invalid parameters.

Accumulate centered products in `int32`, adding optional per-channel `int32` Conv bias before
scaling. Round ties to even, add the output zero point, then saturate.

Fuse dequantize–Conv/MatMul–quantize patterns for compatible constant scales and zero points.
Accept runtime weights and equivalent negative axes. Conv bias may be absent or a constant float
tensor exactly representable as `int32` at each channel’s accumulator scale. Skip fusion if an
intermediate is shared or is a graph output. Preserve quantized results.

## CLI

```text
rten optimize INPUT.onnx -o OUTPUT.onnx
rten calibrate INPUT.onnx -o PROFILE [--sample FILE]... [--profile PROFILE]...
rten quantize INPUT.onnx -o OUTPUT.onnx --format {qdq,qoperator} [--sample FILE]... [--profile PROFILE]...
```

`optimize` exports eligible fusions. `calibrate` records activation minima and maxima from
SafeTensors samples keyed by model input name. Merging profiles from the same model in any
order must reproduce single-run ranges without retaining samples. Reject incompatible profiles.

`quantize` accepts samples or saved profiles. Calibrate and rewrite nested `If` branches and
`Loop` bodies, including captured weights and loop-carried values. Leave unobserved float
regions unchanged. Preserve public input and output names, shapes, and dtypes. Exports must
reload with optimization disabled after moving the model and its external tensor files together.

With `--max-bytes N --max-error E` and repeatable `--validate FILE`, choose float or calibrated
8-bit execution per eligible region. Count the ONNX file and all required sidecars toward `N`.
Bound the maximum absolute output error against the original float model over all supplied
validation samples by `E`. Report infeasible limits if no assignment satisfies both.

Calibration and export must run within 256 MiB peak RSS on a model with 512 MiB of external
weights whose individual layers and activations fit in memory. Preserve shared weight storage
across graph scopes for each distinct encoding.

## Verification

Test shape inference and fusion eligibility. Check arithmetic against an independent
dequantize–compute–quantize reference, exactly for exact binary-scale fixtures and within
1 LSB for floating-point roundoff. Compare both exported formats on held-out inputs.
Document usage and limitations in `docs/quantization.md`.

Existing kernel limitations for packed asymmetric `int8` MatMul weights, padded asymmetric
Conv inputs, and multi-image Conv prepacking are out of scope.
