//! Send input tensors whose byte length does not match their dimensions to
//! the OpenVINO backend, including dimensions whose byte size overflows.
//! Check that the backend returns an error and does not trap.

use anyhow::{Context, Result, ensure};
use std::fs;
use test_programs::nn::witx::{self, TensorType};

/// The byte length of the `[1, 3, 224, 224]` `f32` input of MobileNet.
const INPUT_BYTES: usize = 1 * 3 * 224 * 224 * 4;

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
    let mut context = graph.init_execution_context()?;

    for len in [INPUT_BYTES - 1, INPUT_BYTES + 1] {
        let result = context.set_input(0, TensorType::F32, &[1, 3, 224, 224], &vec![0u8; len]);
        ensure!(
            result.is_err(),
            "expected an error for an input tensor of {len} bytes"
        );
        println!("[nn] rejected an input tensor of {len} bytes");
    }

    // Dimensions whose byte size overflows must be rejected as well.
    let result = context.set_input(0, TensorType::F32, &[u32::MAX as usize; 4], &[0u8; 4]);
    ensure!(
        result.is_err(),
        "expected an error for dimensions whose byte size overflows"
    );
    println!("[nn] rejected overflowing dimensions");

    // A huge tensor with far too little data must be rejected; the host checks
    // the length before creating the tensor, so this never allocates 4 GiB.
    let result = context.set_input(0, TensorType::F32, &[1 << 30, 1], &[0u8; 4]);
    ensure!(
        result.is_err(),
        "expected an error for a 4 GiB tensor with 4 bytes of data"
    );
    println!("[nn] rejected a 4 GiB tensor with 4 bytes of data");
    Ok(())
}
