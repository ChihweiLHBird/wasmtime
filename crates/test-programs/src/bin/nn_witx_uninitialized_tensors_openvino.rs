//! Check that the OpenVINO backend never hands the guest uninitialized host
//! memory through tensors that inference has not written yet:
//!
//! 1. `get_output` before `compute` must either fail or return all zeros.
//! 2. `compute` without `set_input` must behave exactly like `compute` with an
//!    all-zero input tensor.
//!
//! See https://github.com/bytecodealliance/wasmtime/issues/14437.

use anyhow::{Context, Result, ensure};
use std::fs;
use test_programs::nn::witx::{self, TensorType};
use wasi_nn::Graph;

/// The byte length of the `[1, 3, 224, 224]` `f32` input of MobileNet.
const INPUT_BYTES: usize = 1 * 3 * 224 * 224 * 4;
/// The byte length of the `[1, 1001]` `f32` output of MobileNet.
const OUTPUT_BYTES: usize = 1001 * 4;

pub fn main() -> Result<()> {
    let xml = fs::read("fixture/model.xml")
        .context("the model file to be mapped to the fixture directory")?;
    let weights = fs::read("fixture/model.bin")
        .context("the weights file to be mapped to the fixture directory")?;
    let graph = witx::load(
        &[&xml, &weights],
        witx::GraphEncoding::Openvino,
        witx::ExecutionTarget::CPU,
    )?;

    // 1. Read an output tensor without ever calling `set_input` or `compute`.
    // Use a buffer larger than the output and pre-fill it with a non-zero
    // pattern so that only bytes actually written by the host count.
    let context = graph.init_execution_context()?;
    let mut buffer = vec![0xffu8; OUTPUT_BYTES * 2];
    match context.get_output(0, &mut buffer) {
        Err(e) => println!("[nn] `get_output` before `compute` failed: {e}"),
        Ok(num_bytes) => {
            println!("[nn] `get_output` before `compute` returned {num_bytes} bytes");
            ensure!(
                num_bytes == OUTPUT_BYTES,
                "expected {OUTPUT_BYTES} output bytes, got {num_bytes}"
            );
            ensure!(
                buffer[..num_bytes].iter().all(|&b| b == 0),
                "output tensor read before `compute` contains non-zero bytes; \
                 the host leaked uninitialized memory"
            );
            ensure!(
                buffer[num_bytes..].iter().all(|&b| b == 0xff),
                "the host wrote past the reported output length"
            );
        }
    }

    // 2. Inference without any input set must run over zero-initialized input
    // tensors, i.e. produce the same output as an explicit all-zero input.
    let without_input = infer(&graph, None)?;
    let with_zero_input = infer(&graph, Some(&vec![0u8; INPUT_BYTES]))?;
    ensure!(
        without_input == with_zero_input,
        "`compute` without `set_input` ran over uninitialized input memory"
    );
    println!("[nn] `compute` without `set_input` matches an all-zero input");
    Ok(())
}

/// Run one inference on a fresh execution context and return the raw output.
fn infer(graph: &Graph, input: Option<&[u8]>) -> Result<Vec<u8>> {
    let mut context = graph.init_execution_context()?;
    if let Some(input) = input {
        context.set_input(0, TensorType::F32, &[1, 3, 224, 224], input)?;
    }
    context.compute()?;
    let mut buffer = vec![0u8; OUTPUT_BYTES];
    let num_bytes = context.get_output(0, &mut buffer)?;
    buffer.truncate(num_bytes);
    Ok(buffer)
}
