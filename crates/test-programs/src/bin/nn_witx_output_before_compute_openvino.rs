//! Retrieve an output tensor from an OpenVINO execution context without ever
//! calling `compute`. The host must not hand the guest uninitialized host
//! memory: the call must either fail or return all-zero bytes.
//!
//! See https://github.com/bytecodealliance/wasmtime/issues/14437.

use anyhow::{Context, Result, ensure};
use std::fs;
use test_programs::nn::witx;

/// The byte length of the `[1, 1001]` `f32` output of MobileNet.
const OUTPUT_BYTES: usize = 1001 * std::mem::size_of::<f32>();

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
    let context = graph.init_execution_context()?;

    // Deliberately skip `set_input` and `compute`. Pre-fill the buffer with a
    // non-zero pattern so that only bytes actually written by the host count.
    let mut output_buffer = vec![0xffu8; OUTPUT_BYTES];
    match context.get_output(0, &mut output_buffer) {
        Err(e) => println!("[nn] `get_output` before `compute` failed: {e}"),
        Ok(num_bytes) => {
            println!("[nn] `get_output` before `compute` returned {num_bytes} bytes");
            ensure!(
                num_bytes == OUTPUT_BYTES,
                "expected {OUTPUT_BYTES} output bytes, got {num_bytes}"
            );
            ensure!(
                output_buffer.iter().all(|&b| b == 0),
                "output tensor read before `compute` contains non-zero bytes; \
                 the host leaked uninitialized memory"
            );
        }
    }
    Ok(())
}
