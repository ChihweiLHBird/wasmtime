//! Define the Rust interface a backend must implement in order to be used by
//! this crate. The `Box<dyn ...>` types returned by these interfaces allow
//! implementations to maintain backend-specific state between calls.

#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(all(feature = "openvino", target_pointer_width = "64"))]
pub mod openvino;
#[cfg(feature = "pytorch")]
pub mod pytorch;
#[cfg(all(feature = "winml", target_os = "windows"))]
pub mod winml;

#[cfg(feature = "onnx")]
use self::onnx::OnnxBackend;
#[cfg(all(feature = "openvino", target_pointer_width = "64"))]
use self::openvino::OpenvinoBackend;
#[cfg(feature = "pytorch")]
use self::pytorch::PytorchBackend;
#[cfg(all(feature = "winml", target_os = "windows"))]
use self::winml::WinMLBackend;

use crate::wit::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{Backend, ExecutionContext, Graph};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;
use wiggle::GuestError;

/// Return a list of all available backend frameworks.
pub fn list() -> Vec<Backend> {
    let mut backends = vec![];
    let _ = &mut backends; // silence warnings if none are enabled
    #[cfg(all(feature = "openvino", target_pointer_width = "64"))]
    {
        backends.push(Backend::from(OpenvinoBackend::default()));
    }
    #[cfg(all(feature = "winml", target_os = "windows"))]
    {
        backends.push(Backend::from(WinMLBackend::default()));
    }
    #[cfg(feature = "onnx")]
    {
        backends.push(Backend::from(OnnxBackend::default()));
    }
    #[cfg(feature = "pytorch")]
    {
        backends.push(Backend::from(PytorchBackend::default()));
    }
    backends
}

/// A [Backend] contains the necessary state to load [Graph]s.
pub trait BackendInner: Send + Sync {
    fn encoding(&self) -> GraphEncoding;
    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError>;
    fn as_dir_loadable<'a>(&'a mut self) -> Option<&'a mut dyn BackendFromDir>;
}

/// Some [Backend]s support loading a [Graph] from a directory on the
/// filesystem; this is not a general requirement for backends but is useful for
/// the Wasmtime CLI.
pub trait BackendFromDir: BackendInner {
    fn load_from_dir(
        &mut self,
        builders: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError>;
}

/// A [BackendGraph] can create [BackendExecutionContext]s; this is the backing
/// implementation for the user-facing graph.
pub trait BackendGraph: Send + Sync {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError>;
}

/// A [BackendExecutionContext] performs the actual inference; this is the
/// backing implementation for a user-facing execution context.
pub trait BackendExecutionContext: Send + Sync {
    // WITX functions
    fn set_input(&mut self, id: Id, tensor: &Tensor) -> Result<(), BackendError>;
    fn get_output(&mut self, id: Id) -> Result<Tensor, BackendError>;

    // Functions which work for both WIT and WITX
    fn compute(
        &mut self,
        inputs: Option<Vec<NamedTensor>>,
    ) -> Result<Option<Vec<NamedTensor>>, BackendError>;
}

/// An identifier for a tensor in a [Graph].
#[derive(Debug)]
pub enum Id {
    Index(u32),
    Name(String),
}
impl Id {
    pub fn index(&self) -> Option<u32> {
        match self {
            Id::Index(i) => Some(*i),
            Id::Name(_) => None,
        }
    }
    pub fn name(&self) -> Option<&str> {
        match self {
            Id::Index(_) => None,
            Id::Name(n) => Some(n),
        }
    }
}

/// Errors returned by a backend; [BackendError::BackendAccess] is a catch-all
/// for failures interacting with the ML library.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Failed while accessing backend")]
    BackendAccess(#[from] wasmtime::Error),
    #[error("Failed while accessing guest module")]
    GuestAccess(#[from] GuestError),
    #[error("The backend expects {0} buffers, passed {1}")]
    InvalidNumberOfBuilders(usize, usize),
    #[error("Not enough memory to copy tensor data of size: {0}")]
    NotEnoughMemory(usize),
    #[error("Unsupported tensor type: {0}")]
    UnsupportedTensorType(String),
}

/// Read a file into a byte vector.
#[allow(dead_code, reason = "not used on all platforms")]
fn read(path: &Path) -> wasmtime::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = vec![];
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

pub struct NamedTensor {
    pub name: String,
    pub tensor: Tensor,
}

/// Return the size in bytes of one element of type `ty`.
#[allow(dead_code, reason = "not used by every backend feature combination")]
pub(crate) fn bytes_per_element(ty: TensorType) -> usize {
    match ty {
        TensorType::U8 => 1,
        TensorType::Fp16 | TensorType::Bf16 => 2,
        TensorType::Fp32 | TensorType::I32 => 4,
        TensorType::Fp64 | TensorType::I64 => 8,
    }
}

/// Return the number of bytes in a tensor of type `ty` with the given
/// dimensions, or an error if that number overflows `usize`.
#[allow(dead_code, reason = "not used by every backend feature combination")]
pub(crate) fn byte_count(dimensions: &[u32], ty: TensorType) -> wasmtime::Result<usize> {
    dimensions
        .iter()
        .try_fold(bytes_per_element(ty), |count, d| {
            count.checked_mul(usize::try_from(*d).ok()?)
        })
        .ok_or_else(|| wasmtime::format_err!("tensor dimensions are too large: {dimensions:?}"))
}

/// Check that a guest-provided tensor carries exactly the number of bytes its
/// dimensions and element type call for.
///
/// Backends call this before handing the data to their ML library, because
/// the libraries either copy as many bytes as the dimensions imply without
/// looking at the data length or allocate (and zero-fill) the whole tensor
/// up front.
#[allow(dead_code, reason = "not used by every backend feature combination")]
pub(crate) fn check_data_length(tensor: &Tensor) -> wasmtime::Result<()> {
    let expected = byte_count(&tensor.dimensions, tensor.ty)?;
    if tensor.data.len() != expected {
        return Err(wasmtime::format_err!(
            "input tensor has {} bytes of data but dimensions {:?} of type {:?} need {expected} bytes",
            tensor.data.len(),
            tensor.dimensions,
            tensor.ty
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_count_handles_every_element_type_and_overflow() {
        assert_eq!(byte_count(&[2, 3], TensorType::Fp32).unwrap(), 24);
        assert_eq!(byte_count(&[5], TensorType::U8).unwrap(), 5);
        assert_eq!(byte_count(&[5], TensorType::Fp16).unwrap(), 10);
        assert_eq!(byte_count(&[5], TensorType::Bf16).unwrap(), 10);
        assert_eq!(byte_count(&[5], TensorType::I32).unwrap(), 20);
        assert_eq!(byte_count(&[5], TensorType::Fp64).unwrap(), 40);
        assert_eq!(byte_count(&[5], TensorType::I64).unwrap(), 40);
        assert_eq!(byte_count(&[], TensorType::Fp32).unwrap(), 4);
        assert_eq!(byte_count(&[0, 7], TensorType::Fp32).unwrap(), 0);
        assert!(byte_count(&[u32::MAX; 4], TensorType::I64).is_err());
        assert!(byte_count(&[1 << 31, 1 << 31, 4], TensorType::Fp32).is_err());
    }

    #[test]
    fn check_data_length_rejects_mismatches() {
        let exact = Tensor::new(vec![2, 3], TensorType::Fp32, vec![0; 24]);
        assert!(check_data_length(&exact).is_ok());
        let short = Tensor::new(vec![4096], TensorType::Fp32, vec![0; 4]);
        assert!(check_data_length(&short).is_err());
        let long = Tensor::new(vec![2], TensorType::Fp16, vec![0; 8]);
        assert!(check_data_length(&long).is_err());
        let huge = Tensor::new(vec![u32::MAX; 4], TensorType::I64, vec![]);
        assert!(check_data_length(&huge).is_err());
    }
}
