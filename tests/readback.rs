//! Reload the submitted ONNX with optimizations disabled and compare public outputs.
use rten::{ModelOptions, Value};
use rten_serialize::Value as Saved;
use rten_tensor::prelude::*;
use std::error::Error;

fn input_value(v: Saved) -> Result<Value, Box<dyn Error>> {
    Ok(match v {
        Saved::Float32(t) => t.into(),
        Saved::Int32(t) => t.into(),
        Saved::UInt8(t) => t.into(),
        Saved::Int8(t) => t.into(),
        v => return Err(format!("unexpected fixture input {:?}", v.dtype()).into()),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let tolerance: f64 = args[3].to_str().ok_or("invalid tolerance")?.parse()?;
    let mut opts = ModelOptions::with_all_ops();
    opts.enable_optimization(false);
    opts.prepack_weights(false);
    let model = opts.load_file(&args[0])?;
    let mut feeds = Vec::new();
    for (name, value) in rten_serialize::safetensors::read_from_file(&args[1])? {
        feeds.push((model.node_id(&name)?, input_value(value)?));
    }
    let inputs = feeds
        .iter()
        .map(|(id, v)| (*id, v.as_view().into()))
        .collect();
    let got = model.run(inputs, model.output_ids(), None)?;
    let mut expected = rten_serialize::safetensors::read_from_file(&args[2])?;
    for (&id, value) in model.output_ids().iter().zip(got) {
        let name = model
            .node_info(id)
            .and_then(|n| n.name())
            .ok_or("unnamed public output")?;
        let Some(Saved::Float32(reference)) = expected.remove(name) else {
            return Err(format!("unexpected output {name}").into());
        };
        let Value::FloatTensor(actual) = value else {
            return Err(format!("incorrect dtype for {name}").into());
        };
        if actual.shape() != reference.shape() {
            return Err(format!(
                "{name} shape {:?}, expected {:?}",
                actual.shape(),
                reference.shape()
            )
            .into());
        }
        for (i, (&a, &b)) in actual.iter().zip(reference.iter()).enumerate() {
            let error = (f64::from(a) - f64::from(b)).abs();
            if !error.is_finite() || error > tolerance {
                return Err(
                    format!("{name}[{i}] = {a}, expected {b}, tolerance {tolerance}").into(),
                );
            }
        }
    }
    if !expected.is_empty() {
        return Err("missing public outputs".into());
    }
    Ok(())
}
