//! Send input tensors whose byte length does not match their dimensions to
//! the OpenVINO backend, including dimensions whose byte size overflows.
//! Check that the backend returns an error and does not trap.

use anyhow::{Context, Result, anyhow, ensure};
use std::fs;
use test_programs::nn::wit::{self, Tensor, TensorType};

/// The byte length of the `[1, 3, 224, 224]` `f32` input of MobileNet.
const INPUT_BYTES: usize = 1 * 3 * 224 * 224 * 4;

pub fn main() -> Result<()> {
    let xml = fs::read("fixture/model.xml")
        .context("the model file to be mapped to the fixture directory")?;
    let weights = fs::read("fixture/model.bin")
        .context("the weights file to be mapped to the fixture directory")?;
    let graph = wit::load(
        &[xml, weights],
        wit::GraphEncoding::Openvino,
        wit::ExecutionTarget::Cpu,
    )?;
    let context = graph
        .init_execution_context()
        .map_err(|e| anyhow!("{e:?}"))?;

    for len in [INPUT_BYTES - 1, INPUT_BYTES + 1] {
        let tensor = Tensor::new(&[1, 3, 224, 224], TensorType::Fp32, &vec![0u8; len]);
        let result = context.compute(vec![("input".to_string(), tensor)]);
        ensure!(
            result.is_err(),
            "expected an error for an input tensor of {len} bytes"
        );
        println!("[nn] rejected an input tensor of {len} bytes");
    }

    // Dimensions whose byte size overflows must be rejected as well.
    let tensor = Tensor::new(&[u32::MAX; 4], TensorType::Fp32, &[0u8; 4]);
    let result = context.compute(vec![("input".to_string(), tensor)]);
    ensure!(
        result.is_err(),
        "expected an error for dimensions whose byte size overflows"
    );
    println!("[nn] rejected overflowing dimensions");
    Ok(())
}
