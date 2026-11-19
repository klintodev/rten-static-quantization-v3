//! Hidden behavioral tests, compiled as a child of `model` by the runner.
//! Uses existing APIs only, so the unmodified base compiles and fails assertions.
use rten_onnx::onnx;
use rten_tensor::Tensor;
use rten_tensor::prelude::*;

use super::onnx_builder::{
    GraphProtoExt, ModelProtoExt, NodeProtoExt, ValueInfoProtoExt, create_node,
    create_tensor_from_view, create_value_info,
};
use super::{Model, ModelOptions};
use crate::{DataType, Dimension, Value, ValueType};

#[derive(Clone, Copy, Debug)]
enum Q {
    U8,
    I8,
}

impl Q {
    fn dtype(self) -> onnx::DataType {
        match self {
            Self::U8 => onnx::DataType::UINT8,
            Self::I8 => onnx::DataType::INT8,
        }
    }
    fn zero(self) -> i32 {
        match self {
            Self::U8 => 121,
            Self::I8 => -3,
        }
    }
    fn bounds(self) -> (i32, i32) {
        match self {
            Self::U8 => (0, 255),
            Self::I8 => (-128, 127),
        }
    }
    fn tensor(self, shape: &[usize], data: &[i32]) -> Value {
        match self {
            Self::U8 => Tensor::from_data(
                shape,
                data.iter()
                    .map(|&v| u8::try_from(v).unwrap())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Self::I8 => Tensor::from_data(
                shape,
                data.iter()
                    .map(|&v| i8::try_from(v).unwrap())
                    .collect::<Vec<_>>(),
            )
            .into(),
        }
    }
}

fn floats(shape: &[usize], values: &[f32]) -> Value {
    Tensor::from_data(shape, values.to_vec()).into()
}

fn initializer(name: &str, value: &Value) -> onnx::TensorProto {
    match value {
        Value::FloatTensor(t) => create_tensor_from_view(name, t.view()),
        Value::Int8Tensor(t) => create_tensor_from_view(name, t.view()),
        Value::UInt8Tensor(t) => create_tensor_from_view(name, t.view()),
        Value::Int32Tensor(t) => create_tensor_from_view(name, t.view()),
        _ => panic!("unsupported fixture dtype"),
    }
}

fn integers(value: &Value) -> Vec<i32> {
    match value {
        Value::Int8Tensor(t) => t.iter().map(|&x| x as i32).collect(),
        Value::UInt8Tensor(t) => t.iter().map(|&x| x as i32).collect(),
        _ => panic!("expected quantized output, got {value:?}"),
    }
}

fn shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Int8Tensor(t) => t.shape().to_vec(),
        Value::UInt8Tensor(t) => t.shape().to_vec(),
        _ => panic!("expected quantized output"),
    }
}

fn assert_quantized(actual: &Value, q: Q, dims: &[usize], expected: &[i32], tolerance: i32) {
    assert!(
        matches!(
            (q, actual),
            (Q::U8, Value::UInt8Tensor(_)) | (Q::I8, Value::Int8Tensor(_))
        ),
        "wrong output dtype"
    );
    assert_eq!(shape(actual), dims);
    let values = integers(actual);
    assert_eq!(values.len(), expected.len());
    for (i, (&a, &b)) in values.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= tolerance,
            "element {i}: got {a}, expected {b}, tolerance {tolerance}"
        );
    }
}

#[derive(Clone)]
struct Case {
    conv: bool,
    xq: Q,
    wq: Q,
    yq: Q,
    xshape: Vec<usize>,
    wshape: Vec<usize>,
    x: Vec<i32>,
    w: Vec<i32>,
    xs: f32,
    ws: Vec<f32>,
    ys: f32,
    xz: i32,
    wz: Vec<i32>,
    yz: i32,
    bias: Option<Vec<i32>>,
    groups: usize,
    strides: [usize; 2],
    dilations: [usize; 2],
    pads: [usize; 4],
}

impl Case {
    fn name(&self) -> &'static str {
        if self.conv {
            "QLinearConv"
        } else {
            "QLinearMatMul"
        }
    }
    fn params(&self) -> Vec<(&'static str, Value)> {
        let ws_shape = if self.ws.len() == 1 {
            vec![]
        } else {
            vec![self.ws.len()]
        };
        let wz_shape = if self.wz.len() == 1 {
            vec![]
        } else {
            vec![self.wz.len()]
        };
        let mut values = vec![
            ("x", self.xq.tensor(&self.xshape, &self.x)),
            ("xs", floats(&[], &[self.xs])),
            ("xz", self.xq.tensor(&[], &[self.xz])),
            ("w", self.wq.tensor(&self.wshape, &self.w)),
            ("ws", floats(&ws_shape, &self.ws)),
            ("wz", self.wq.tensor(&wz_shape, &self.wz)),
            ("ys", floats(&[], &[self.ys])),
            ("yz", self.yq.tensor(&[], &[self.yz])),
        ];
        if let Some(bias) = &self.bias {
            values.push((
                "bias",
                Tensor::from_data(&[bias.len()], bias.clone()).into(),
            ));
        }
        values
    }
    fn attrs(&self, node: onnx::NodeProto) -> onnx::NodeProto {
        if !self.conv {
            return node;
        }
        node.with_attr("group", self.groups as i64)
            .with_attr("strides", self.strides.map(|x| x as i64).to_vec())
            .with_attr("dilations", self.dilations.map(|x| x as i64).to_vec())
            .with_attr("pads", self.pads.map(|x| x as i64).to_vec())
    }
    fn graph(&self) -> onnx::GraphProto {
        let mut node = self
            .attrs(create_node(self.name()))
            .with_name("compute")
            .with_output("y");
        let mut graph = onnx::GraphProto::default();
        for (name, value) in self.params() {
            node = node.with_input(name);
            if name == "x" {
                graph = graph.with_input(
                    create_value_info(name)
                        .with_dtype(self.xq.dtype())
                        .with_shape(
                            &self
                                .xshape
                                .iter()
                                .copied()
                                .map(Dimension::Fixed)
                                .collect::<Vec<_>>(),
                        ),
                );
            } else {
                graph = graph.with_initializer(initializer(name, &value));
            }
        }
        graph.with_node(node).with_output(create_value_info("y"))
    }
    fn quantize(&self, real: f64) -> i32 {
        let (lo, hi) = self.yq.bounds();
        ((real / self.ys as f64).round_ties_even() as i64 + self.yz as i64)
            .clamp(lo as i64, hi as i64) as i32
    }
    // Independent scalar f64 dequantize -> compute -> quantize reference.
    // All exact cases use binary scales and small centered operands.
    fn reference(&self) -> (Vec<usize>, Vec<i32>) {
        let mut out = vec![];
        if !self.conv {
            let k = *self.xshape.last().unwrap();
            let n = *self.wshape.last().unwrap();
            for row in self.x.chunks(k) {
                for col in 0..n {
                    let ws = self.ws[col % self.ws.len()] as f64;
                    let wz = self.wz[col % self.wz.len()];
                    let real = row
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            (x - self.xz) as f64
                                * self.xs as f64
                                * (self.w[i * n + col] - wz) as f64
                                * ws
                        })
                        .sum();
                    out.push(self.quantize(real));
                }
            }
            let mut dims = self.xshape.clone();
            *dims.last_mut().unwrap() = n;
            return (dims, out);
        }
        let [batch, channels, h, w]: [usize; 4] = self.xshape.clone().try_into().unwrap();
        let [oc, ic, kh, kw]: [usize; 4] = self.wshape.clone().try_into().unwrap();
        let oh = (h + self.pads[0] + self.pads[2] - (kh - 1) * self.dilations[0] - 1)
            / self.strides[0]
            + 1;
        let ow = (w + self.pads[1] + self.pads[3] - (kw - 1) * self.dilations[1] - 1)
            / self.strides[1]
            + 1;
        for b in 0..batch {
            for c in 0..oc {
                for y in 0..oh {
                    for x in 0..ow {
                        let ws = self.ws[c % self.ws.len()] as f64;
                        let wz = self.wz[c % self.wz.len()];
                        let mut real = self
                            .bias
                            .as_ref()
                            .map_or(0., |bias| bias[c] as f64 * self.xs as f64 * ws);
                        for i in 0..ic {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (y * self.strides[0] + ky * self.dilations[0])
                                        as isize
                                        - self.pads[0] as isize;
                                    let ix = (x * self.strides[1] + kx * self.dilations[1])
                                        as isize
                                        - self.pads[1] as isize;
                                    if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                        continue;
                                    }
                                    let chan = (c / (oc / self.groups)) * ic + i;
                                    let xv = self.x[((b * channels + chan) * h + iy as usize) * w
                                        + ix as usize];
                                    let wv = self.w[((c * ic + i) * kh + ky) * kw + kx];
                                    real += (xv - self.xz) as f64
                                        * self.xs as f64
                                        * (wv - wz) as f64
                                        * ws;
                                }
                            }
                        }
                        out.push(self.quantize(real));
                    }
                }
            }
        }
        (vec![batch, oc, oh, ow], out)
    }
}

fn matmul(xq: Q, wq: Q, yq: Q, channel: bool) -> Case {
    let (xz, wz, yz) = (xq.zero(), wq.zero(), yq.zero());
    let z = if channel {
        vec![wz, wz + 2, wz - 1]
    } else {
        vec![wz]
    };
    Case {
        conv: false,
        xq,
        wq,
        yq,
        xshape: vec![2, 4],
        wshape: vec![4, 3],
        x: [-4, 2, 7, -3, 6, -1, 3, 5].iter().map(|x| x + xz).collect(),
        w: [-2, 4, 3, 5, -3, 2, 1, 6, -4, -3, 1, 5]
            .iter()
            .enumerate()
            .map(|(i, w)| w + z[(i % 3) % z.len()])
            .collect(),
        xs: 0.5,
        ws: if channel {
            vec![0.25, 0.5, 0.125]
        } else {
            vec![0.25]
        },
        ys: 0.25,
        xz,
        wz: z,
        yz,
        bias: None,
        groups: 1,
        strides: [1, 1],
        dilations: [1, 1],
        pads: [0; 4],
    }
}

fn conv(xq: Q, wq: Q, yq: Q, channel: bool, bias: bool) -> Case {
    let mut c = matmul(xq, wq, yq, channel);
    c.conv = true;
    // Multi-image general Conv uses an existing prepacked-A path with incorrect
    // asymmetric zero-point handling. Test spatial attributes on one image;
    // cover multiple images separately with pointwise Conv below.
    c.xshape = vec![1, 2, 5, 6];
    c.wshape = vec![4, 1, 2, 2];
    c.groups = 2;
    c.strides = [2, 1];
    c.dilations = [1, 2];
    c.pads = [1, 1, 0, 2];
    // Im2Col padding on the base also assumes zero in the internal i8 domain.
    // Pointwise fixtures retain asymmetric activation zero-point coverage.
    c.xz = match xq {
        Q::I8 => 0,
        Q::U8 => 128,
    };
    c.ws = if channel {
        vec![0.25, 0.5, 0.125, 1.]
    } else {
        vec![0.25]
    };
    c.wz = if channel {
        vec![wq.zero(), wq.zero() + 2, wq.zero() - 1, wq.zero() + 3]
    } else {
        vec![wq.zero()]
    };
    c.x = (0..60).map(|i| i % 13 - 6 + c.xz).collect();
    c.w = (0..16)
        .map(|i| (i % 7) as i32 - 3 + c.wz[(i / 4) % c.wz.len()])
        .collect();
    c.bias = bias.then(|| vec![3, -7, 9, -2]);
    c
}

fn conv_batched_pointwise(xq: Q, wq: Q, yq: Q, channel: bool, bias: bool) -> Case {
    let mut c = conv(xq, wq, yq, channel, bias);
    c.xshape = vec![2, 2, 2, 3];
    c.xz = xq.zero();
    c.x = (0..24).map(|i| i % 13 - 6 + c.xz).collect();
    c.wshape = vec![4, 2, 1, 1];
    c.w = (0..8)
        .map(|i| (i % 7) as i32 - 3 + c.wz[(i / 2) % c.wz.len()])
        .collect();
    c.groups = 1;
    c.strides = [1, 1];
    c.dilations = [1, 1];
    c.pads = [0; 4];
    c
}

fn packable_matmul(mut c: Case) -> Case {
    // The base MatMulInteger's packed-B path drops nonzero i8 weight zero
    // points. Keep asymmetric weights in unpacked checks; packed checks still
    // exercise nonzero activation/output zero points and per-column scales.
    assert!(!c.conv);
    if matches!(c.wq, Q::I8) {
        let cols = *c.wshape.last().unwrap();
        for (i, w) in c.w.iter_mut().enumerate() {
            *w -= c.wz[(i % cols) % c.wz.len()];
        }
        c.wz.fill(0);
    }
    c
}

fn load(graph: onnx::GraphProto, optimize: bool, prepack: bool) -> Model {
    let bytes = graph.into_model().with_opset("", 18).write_buf().unwrap();
    ModelOptions::with_all_ops()
        .enable_optimization(optimize)
        .prepack_weights(prepack)
        .load(bytes)
        .expect("valid quantized ONNX model must load")
}

fn run(model: &Model, c: &Case) -> Value {
    model
        .run_one(c.xq.tensor(&c.xshape, &c.x).into(), None)
        .expect("quantized model must execute")
}

fn check(c: &Case, prepack: bool, tolerance: i32) {
    let m = load(c.graph(), false, prepack);
    let (dims, expected) = c.reference();
    assert_quantized(&run(&m, c), c.yq, &dims, &expected, tolerance);
}

#[test]
fn f2p_matmul_signedness_and_channels() {
    for x in [Q::I8, Q::U8] {
        for w in [Q::I8, Q::U8] {
            for y in [Q::I8, Q::U8] {
                for channels in [false, true] {
                    let c = matmul(x, w, y, channels);
                    check(&c, false, 0);
                    if matches!(w, Q::I8) {
                        check(&packable_matmul(c), true, 0);
                    }
                }
            }
        }
    }
}

#[test]
fn f2p_conv_channels_bias_and_attributes() {
    for x in [Q::I8, Q::U8] {
        for w in [Q::I8, Q::U8] {
            for y in [Q::I8, Q::U8] {
                for channels in [false, true] {
                    for bias in [false, true] {
                        check(&conv(x, w, y, channels, bias), true, 0);
                        check(&conv_batched_pointwise(x, w, y, channels, bias), true, 0);
                    }
                }
            }
        }
    }
}

#[test]
fn f2p_rounding_zero_point_and_saturation() {
    for is_conv in [false, true] {
        for yq in [Q::I8, Q::U8] {
            for xs in [0.5, 8.] {
                let mut c = matmul(Q::I8, Q::I8, yq, false);
                c.conv = is_conv;
                c.xz = 0;
                c.wz = vec![0];
                c.yz = if matches!(yq, Q::U8) { 121 } else { -3 };
                c.x = vec![-127, -7, -5, -3, -1, 0, 1, 3, 5, 7, 127];
                c.w = vec![1];
                c.xshape = if is_conv {
                    vec![1, 1, 1, 11]
                } else {
                    vec![11, 1]
                };
                c.wshape = if is_conv {
                    vec![1, 1, 1, 1]
                } else {
                    vec![1, 1]
                };
                c.xs = xs;
                c.ws = vec![1.];
                c.ys = 1.;
                check(&c, false, 0);
            }
        }
    }
}

#[test]
fn f2p_conv_1d_auto_padding() {
    let mut c = conv(Q::U8, Q::I8, Q::I8, true, false);
    c.xshape = vec![1, 1, 6];
    c.wshape = vec![2, 1, 2];
    c.x = vec![122, 124, 120, 126, 119, 123];
    c.w = vec![-1, -4, 1, -1];
    c.ws = vec![0.25, 0.5];
    c.wz = vec![-3, -2];
    for (mode, start, length) in [("SAME_UPPER", 0isize, 3), ("VALID", 0, 2), ("NOTSET", 1, 3)] {
        let mut graph = c.graph();
        graph.node[0].attribute.clear();
        graph.node[0] = graph.node[0]
            .clone()
            .with_attr("kernel_shape", vec![2i64])
            .with_attr("strides", vec![2i64])
            .with_attr("dilations", vec![2i64])
            .with_attr("auto_pad", mode.to_string());
        if mode == "NOTSET" {
            graph.node[0] = graph.node[0].clone().with_attr("pads", vec![1i64, 0]);
        }
        let mut expected = vec![];
        for channel in 0..2 {
            for out in 0..length {
                let mut real = 0.;
                for kernel in 0..2 {
                    let index = (2 * out + 2 * kernel) as isize - start;
                    if (0..6).contains(&index) {
                        real += (c.x[index as usize] - c.xz) as f64
                            * c.xs as f64
                            * (c.w[channel * 2 + kernel] - c.wz[channel]) as f64
                            * c.ws[channel] as f64;
                    }
                }
                expected.push(c.quantize(real));
            }
        }
        assert_quantized(
            &run(&load(graph, false, true), &c),
            c.yq,
            &[1, 2, length],
            &expected,
            0,
        );
    }
}

#[test]
fn f2p_matmul_batch_and_lhs_vector() {
    let mut c = packable_matmul(matmul(Q::U8, Q::I8, Q::I8, true));
    c.xshape = vec![2, 1, 2, 4];
    let second_batch = c.x.iter().rev().copied().collect::<Vec<_>>();
    c.x.extend(second_batch);
    check(&c, true, 0);
    // Each RHS batch broadcasts over both distinct LHS matrices. Derive the
    // expected outputs one RHS matrix at a time with the scalar reference.
    c.xshape = vec![1, 2, 2, 4];
    c.wshape = vec![3, 1, 4, 3];
    let first_weights = c.w.clone();
    c.w.extend(first_weights.iter().map(|w| -w));
    c.w.extend(first_weights.iter().rev().copied());
    let mut expected = Vec::new();
    for weights in c.w.chunks_exact(12) {
        let mut reference = c.clone();
        reference.w = weights.to_vec();
        reference.wshape = vec![4, 3];
        expected.extend(reference.reference().1);
    }
    assert_quantized(
        &run(&load(c.graph(), false, true), &c),
        c.yq,
        &[3, 2, 2, 3],
        &expected,
        0,
    );
    c.wshape = vec![4, 3];
    c.w.truncate(12);
    // Runtime vector promotion, independent of the existing shape inferencer's limitations.
    c.xshape = vec![4];
    c.x.truncate(4);
    check(&c, false, 0);
}

fn replace(graph: &mut onnx::GraphProto, name: &str, value: Value) {
    let t = graph
        .initializer
        .iter_mut()
        .find(|t| t.name.as_deref() == Some(name))
        .unwrap();
    *t = initializer(name, &value);
}

fn assert_rejected(graph: onnx::GraphProto, c: &Case, parameter: &str) {
    let bytes = graph.into_model().with_opset("", 18).write_buf().unwrap();
    // Both load-time and run-time validation are valid.
    if let Ok(model) = ModelOptions::with_all_ops()
        .enable_optimization(false)
        .load(bytes)
    {
        assert!(
            model
                .run_one(c.xq.tensor(&c.xshape, &c.x).into(), None)
                .is_err(),
            "{} accepted invalid {parameter}",
            c.name()
        );
    }
}

#[test]
fn f2p_matmul_rhs_vector_has_one_output_column() {
    let mut c = matmul(Q::U8, Q::I8, Q::I8, false);
    c.wshape = vec![4];
    c.w = vec![-2, 4, 1, 3];
    let expected: Vec<_> =
        c.x.chunks_exact(c.w.len())
            .map(|row| {
                let dot = row
                    .iter()
                    .zip(&c.w)
                    .map(|(&a, &b)| {
                        (a - c.xz) as f64 * c.xs as f64 * (b - c.wz[0]) as f64 * c.ws[0] as f64
                    })
                    .sum();
                c.quantize(dot)
            })
            .collect();

    for scale_shape in [&[][..], &[1][..]] {
        for zero_shape in [&[][..], &[1][..]] {
            let mut graph = c.graph();
            replace(&mut graph, "ws", floats(scale_shape, &c.ws));
            replace(&mut graph, "wz", c.wq.tensor(zero_shape, &c.wz));
            assert_quantized(
                &run(&load(graph, false, false), &c),
                c.yq,
                &[2],
                &expected,
                0,
            );
        }
    }

    // These four values describe the contracted dimension, not output columns.
    c.ws = vec![0.25, 0.5, 0.125, 1.];
    c.wz = vec![-3, -1, 0, 2];
    assert_rejected(c.graph(), &c, "per-element vector weights");
    for constant_weights in [false, true] {
        assert_not_fused(qdq(&c, constant_weights, -1), &c);
    }
}

#[test]
fn f2p_validation() {
    for c in [
        matmul(Q::U8, Q::I8, Q::U8, true),
        conv(Q::U8, Q::I8, Q::U8, true, true),
    ] {
        check(&c, false, 0); // Missing operators are a failure, never a successful rejection.
        let mut invalid = vec![
            ("xz", Q::I8.tensor(&[], &[0])),
            ("wz", Q::U8.tensor(&[c.wz.len()], &vec![0; c.wz.len()])),
            ("xs", Tensor::from(1i32).into()),
            ("ws", Tensor::from(1i32).into()),
            ("ys", Tensor::from(1i32).into()),
            ("xs", floats(&[2], &[0.5, 0.25])),
            ("ys", floats(&[2], &[0.5, 0.25])),
            ("ws", floats(&[2], &[0.25, 0.5])),
            ("wz", Q::I8.tensor(&[2], &[0, 1])),
            ("ws", floats(&[], &[0.25])),
            ("wz", Q::I8.tensor(&[], &[0])),
            ("ws", floats(&[1, c.ws.len()], &c.ws)),
            ("xz", Q::U8.tensor(&[2], &[121, 121])),
            ("yz", Q::U8.tensor(&[2], &[121, 121])),
            ("yz", Tensor::from(0i32).into()),
        ];
        for name in ["xs", "ws", "ys"] {
            for bad in [0., -1., f32::NAN, f32::INFINITY] {
                let count = if name == "ws" { c.ws.len() } else { 1 };
                let dims = if count == 1 { vec![] } else { vec![count] };
                invalid.push((name, floats(&dims, &vec![bad; count])));
            }
        }
        if c.conv {
            invalid.push(("bias", floats(&[4], &[1.; 4])));
            invalid.push(("bias", Tensor::from_data(&[2], vec![1i32, 2]).into()));
        }
        for (name, value) in invalid {
            let mut graph = c.graph();
            replace(&mut graph, name, value);
            assert_rejected(graph, &c, name);
        }
        let mut scalar = c.clone();
        scalar.ws = vec![0.25];
        scalar.wz = vec![-3];
        let mut graph = scalar.graph();
        for name in ["xs", "ws", "ys"] {
            let value = match name {
                "xs" => scalar.xs,
                "ws" => scalar.ws[0],
                _ => scalar.ys,
            };
            replace(&mut graph, name, floats(&[1], &[value]));
        }
        for (name, q, z) in [
            ("xz", scalar.xq, scalar.xz),
            ("wz", scalar.wq, scalar.wz[0]),
            ("yz", scalar.yq, scalar.yz),
        ] {
            replace(&mut graph, name, q.tensor(&[1], &[z]));
        }
        let (dims, expected) = scalar.reference();
        assert_quantized(
            &run(&load(graph, false, false), &scalar),
            scalar.yq,
            &dims,
            &expected,
            0,
        );
    }
}

#[test]
fn f2p_shapes_and_output_types() {
    for c in [
        matmul(Q::U8, Q::I8, Q::I8, true),
        conv(Q::U8, Q::I8, Q::I8, true, false),
    ] {
        let mut graph = c.graph();
        let mut input_dims: Vec<_> = c.xshape.iter().copied().map(Dimension::Fixed).collect();
        input_dims[0] = Dimension::Symbolic("batch".into());
        graph.input[0] = create_value_info("x")
            .with_dtype(c.xq.dtype())
            .with_shape(&input_dims);
        let m = load(graph, true, false);
        let info = m.node_info(m.node_id("y").unwrap()).unwrap();
        let (dims, _) = c.reference();
        let mut expected: Vec<_> = dims.into_iter().map(Dimension::Fixed).collect();
        expected[0] = Dimension::Symbolic("batch".into());
        assert_eq!(info.shape(), Some(expected));
        assert_eq!(info.dtype(), Some(ValueType::Tensor(DataType::Int8)));
    }
    let mut c = matmul(Q::U8, Q::I8, Q::U8, false);
    c.xshape = vec![2, 1, 2, 4];
    c.x = c.x.repeat(2);
    let m = load(c.graph(), true, false);
    assert_eq!(
        m.node_info(m.node_id("y").unwrap()).unwrap().shape(),
        Some(vec![2, 1, 2, 3].into_iter().map(Dimension::Fixed).collect())
    );
}

// QDQ fixtures keep all quantization parameters constant unless a negative
// test explicitly promotes one to a graph input.
fn qdq(c: &Case, constant_weights: bool, axis: i64) -> onnx::GraphProto {
    let mut graph = c.graph();
    graph.node.clear();
    if !constant_weights {
        graph.initializer.retain(|t| t.name.as_deref() != Some("w"));
        graph = graph.with_input(
            create_value_info("w").with_dtype(c.wq.dtype()).with_shape(
                &c.wshape
                    .iter()
                    .copied()
                    .map(Dimension::Fixed)
                    .collect::<Vec<_>>(),
            ),
        );
    }
    let dx = create_node("DequantizeLinear")
        .with_input("x")
        .with_input("xs")
        .with_input("xz")
        .with_output("dx");
    let dw = create_node("DequantizeLinear")
        .with_input("w")
        .with_input("ws")
        .with_input("wz")
        .with_attr("axis", axis)
        .with_output("dw");
    let compute = c
        .attrs(create_node(if c.conv { "Conv" } else { "MatMul" }))
        .with_input("dx")
        .with_input("dw")
        .with_output("float_y");
    let q = create_node("QuantizeLinear")
        .with_input("float_y")
        .with_input("ys")
        .with_input("yz")
        .with_output("y");
    graph
        .with_node(dx)
        .with_node(dw)
        .with_node(compute)
        .with_node(q)
}

fn plan_uses(m: &Model, op_name: &str) -> bool {
    let graph = m.graph();
    graph
        .execution_plan(m.input_ids(), m.output_ids(), Default::default())
        .unwrap()
        .into_iter()
        .filter_map(|id| graph.get_node(id)?.as_operator())
        .any(|node| node.operator().name() == op_name)
}

fn run_qdq(m: &Model, c: &Case) -> Vec<Value> {
    let params = c.params();
    let inputs = m
        .input_ids()
        .iter()
        .map(|&id| {
            let name = m.node_info(id).unwrap().name().unwrap().to_string();
            let value = params.iter().find(|(n, _)| *n == name).unwrap().1.clone();
            (id, value.into())
        })
        .collect();
    m.run(inputs, m.output_ids(), None).unwrap()
}

fn fusion_case(c: &Case, constant_weights: bool, axis: i64) {
    for prepack in [false, true] {
        let c = if prepack && !c.conv {
            packable_matmul(c.clone())
        } else {
            c.clone()
        };
        let c = &c;
        let graph = qdq(c, constant_weights, axis);
        let plain = load(graph.clone(), false, false);
        let expected = run_qdq(&plain, c);
        let fused = load(graph.clone(), true, prepack);
        assert!(plan_uses(&fused, c.name()), "expected {} fusion", c.name());
        let output = fused.node_id("y").expect("fusion lost the named output");
        assert_eq!(fused.output_ids(), &[output]);
        let actual = run_qdq(&fused, c);
        assert_quantized(
            &actual[0],
            c.yq,
            &shape(&expected[0]),
            &integers(&expected[0]),
            0,
        );
    }
}

#[test]
fn f2p_qdq_matmul_fusion() {
    for channel in [false, true] {
        for constant in [false, true] {
            for axis in [1, -1] {
                fusion_case(&matmul(Q::U8, Q::I8, Q::I8, channel), constant, axis);
            }
        }
    }
}

#[test]
fn f2p_qdq_conv_fusion() {
    for channel in [false, true] {
        for constant in [false, true] {
            for axis in [0, -4] {
                fusion_case(&conv(Q::U8, Q::I8, Q::U8, channel, false), constant, axis);
            }
        }
    }
}

fn assert_not_fused(graph: onnx::GraphProto, c: &Case) {
    let plain = load(graph.clone(), false, false);
    let optimized = load(graph, true, true);
    assert!(
        !plan_uses(&optimized, c.name()),
        "unsafe QDQ pattern was fused"
    );
    let before = run_qdq(&plain, c);
    let after = run_qdq(&optimized, c);
    assert_eq!(
        before, after,
        "optimizer changed outputs of an ineligible pattern"
    );
}

fn check_dynamic_quantization_parameters() {
    for c in [
        matmul(Q::U8, Q::I8, Q::U8, false),
        conv(Q::U8, Q::I8, Q::U8, false, false),
    ] {
        for name in ["xs", "ws", "ys", "xz", "wz", "yz"] {
            let mut graph = qdq(&c, false, 0);
            let t = graph
                .initializer
                .iter()
                .find(|t| t.name.as_deref() == Some(name))
                .unwrap()
                .clone();
            graph
                .initializer
                .retain(|t| t.name.as_deref() != Some(name));
            graph = graph.with_input(
                create_value_info(name)
                    .with_dtype(t.data_type.unwrap())
                    .with_shape(&[]),
            );
            assert_not_fused(graph, &c);
        }
    }
}

fn check_reused_intermediates() {
    for c in [
        matmul(Q::U8, Q::I8, Q::U8, false),
        conv(Q::U8, Q::I8, Q::U8, false, false),
    ] {
        for intermediate in ["dx", "dw", "float_y"] {
            for graph_output in [false, true] {
                let mut graph = qdq(&c, false, 0);
                if graph_output {
                    graph = graph.with_output(create_value_info(intermediate));
                } else {
                    graph = graph
                        .with_node(
                            create_node("Neg")
                                .with_input(intermediate)
                                .with_output("side"),
                        )
                        .with_output(create_value_info("side"));
                }
                assert_not_fused(graph, &c);
            }
        }
    }
}

fn check_incompatible_axes_and_granularity() {
    // Square dimensions make an incorrect axis valid for DQ but different
    // from the output-channel axis required by QLinearMatMul.
    let mut c = matmul(Q::U8, Q::I8, Q::U8, true);
    c.xshape = vec![3, 3];
    c.wshape = vec![3, 3];
    c.x = vec![120, 124, 118, 126, 119, 125, 123, 117, 122];
    c.w.truncate(9);
    assert_not_fused(qdq(&c, false, 0), &c);
    let mut graph = qdq(&c, false, 1);
    replace(&mut graph, "xs", floats(&[3], &[0.25, 0.5, 1.]));
    replace(&mut graph, "xz", Q::U8.tensor(&[3], &[121, 121, 121]));
    assert_not_fused(graph, &c);
    let mut graph = qdq(&c, false, 1);
    replace(&mut graph, "ys", floats(&[3], &[0.25, 0.5, 1.]));
    replace(&mut graph, "yz", Q::U8.tensor(&[3], &[121, 121, 121]));
    assert_not_fused(graph, &c);

    let mut c = conv(Q::U8, Q::I8, Q::U8, true, false);
    c.groups = 1;
    c.xshape = vec![1, 2, 3, 3];
    c.x.truncate(18);
    c.wshape = vec![2, 2, 2, 2];
    c.ws = vec![0.25, 0.5];
    c.wz = vec![-3, -1];
    // Both axis 0 and axis 1 have length two, but only axis 0 is output channels.
    assert_not_fused(qdq(&c, false, 1), &c);
}

#[test]
fn f2p_conservative_qdq_fusion() {
    // First establish that each optimizer recognizes the eligible version.
    // Then change one condition at a time and require the graph to stay unfused.
    fusion_case(&matmul(Q::U8, Q::I8, Q::U8, false), false, 1);
    fusion_case(&conv(Q::U8, Q::I8, Q::U8, false, false), false, 0);
    check_dynamic_quantization_parameters();
    check_reused_intermediates();
    check_incompatible_axes_and_granularity();
}

#[test]
fn f2p_nonbinary_scales() {
    for mut c in [
        matmul(Q::I8, Q::U8, Q::I8, true),
        conv_batched_pointwise(Q::U8, Q::I8, Q::U8, true, true),
    ] {
        c.xs = 0.17;
        c.ys = 0.11;
        for (i, scale) in c.ws.iter_mut().enumerate() {
            *scale = 0.13 + 0.07 * i as f32;
        }
        check(&c, false, 1);
    }
}

#[test]
fn f2p_wide_accumulation() {
    // Each dot product is +/-65536 before requantization. Small operands keep
    // the intermediate pairwise GEMM sums within the signed 16-bit range.
    for is_conv in [false, true] {
        let mut c = matmul(Q::U8, Q::I8, Q::U8, false);
        c.conv = is_conv;
        c.xz = 128;
        c.wz = vec![0];
        c.xs = 1.;
        c.ws = vec![1.];
        c.ys = 2048.;
        c.x = vec![160; 128];
        if is_conv {
            c.xshape = vec![1, 128, 1, 1];
            c.wshape = vec![2, 128, 1, 1];
            c.w = [vec![16; 128], vec![-16; 128]].concat();
        } else {
            c.xshape = vec![1, 128];
            c.wshape = vec![128, 2];
            c.w = [16, -16].repeat(128);
        }
        check(&c, false, 0);
    }
}

#[test]
fn f2p_qdq_conv_bias_eligibility() {
    let c = conv(Q::U8, Q::I8, Q::U8, true, false);
    for constant in [false,true] {
        for exact in [false,true] {
            let mut g = qdq(&c, constant, -4);
            let mut bias: Vec<_> = c.ws.iter().enumerate().map(|(i,&s)| (i as f32 - 2.) * c.xs * s).collect();
            if !exact { bias[2] += 0.5 * c.xs * c.ws[2]; }
            g.initializer.push(initializer("conv_bias", &floats(&[bias.len()], &bias)));
            g.node[2].input.push("conv_bias".into());
            let plain = load(g.clone(), false, false);
            let opt = load(g, true, false);
            assert_eq!(plan_uses(&opt, c.name()), exact);
            let expected = run_qdq(&plain,&c);
            let actual = run_qdq(&opt,&c);
            assert_quantized(&actual[0],c.yq,&shape(&expected[0]),&integers(&expected[0]),0);
        }
    }
}
